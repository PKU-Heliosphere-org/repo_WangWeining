import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import struct
import scipy.fft
from scipy.fft import fftshift
from scipy.fft import fftfreq
from scipy.io import loadmat,savemat
from scipy.interpolate import griddata
matplotlib.rcParams['font.size'] = 16
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
######### loadinfo function


def decompose_vector(v, u, w, epsilon=1e-4):
    """
    将三维向量v分解为u和w的线性组合v = a*u + b*w
    
    参数：
        v: 目标三维向量
        u, w: 用于分解的两个三维向量
        epsilon: 残差阈值，用于判断是否有解
    
    返回：
        若有解，返回(a, b)；否则返回None
    """
    # 构造系数矩阵A（3行2列，每列对应u和w的分量）
    A = np.column_stack((u, w))  # 等价于[[u1, w1], [u2, w2], [u3, w3]]
    
    # 求解超定方程组A·[a,b]^T = v（最小二乘法）
    x, residuals, rank, _ = np.linalg.lstsq(A, v, rcond=None)
    a, b = x  # 候选解
    # print(residuals[0])
    # 验证残差：若残差小于阈值，说明解有效
    if residuals.size == 0 or residuals[0] < epsilon:
        return (a, b)
    else:
        return None


def loadinfo(dir):
    fstr = dir + "info"
    fd = open(fstr,"rb")
    infocontent = fd.read()
    fd.close
    arr = struct.unpack("fIIIffffff", infocontent[:40])
    infoarr=np.zeros(6)
    infoarr[0] = arr[1]
    infoarr[1] = arr[2]
    infoarr[2] = arr[3]
    infoarr[3] = arr[6]
    infoarr[4] = arr[7]
    infoarr[5] = arr[8]
    print(infoarr)
    return infoarr
######### end loadSlice


######### loadSlice function
def loadSlice(dir,q,sl,nx,ny,nz,interval=11):
    fstr = dir + q + f"_{sl*interval}.gda"
    fd = open(fstr,"rb")
    fd.seek(0*4*sl*nx*ny*nz,1)
    arr = np.fromfile(fd,dtype=np.float32,count=nx*ny*nz)
    fd.close()
    arr = np.reshape(arr,(nz, ny, nx))
    arr = np.transpose(arr, axes=(2, 1, 0))
    return arr
######### end loadSlice

def get_psd(array,dx=0.5,dy=0.5,dt=2.56):
    dim = array.ndim
    if dim == 2:
        FS = (scipy.fft.fftn(array, axes=(0,1)))
        psd = np.abs(FS)**2/(FS.size * dx * dy)
    elif dim == 3:
        FS = (scipy.fft.fftn(array, axes=(0,1,2)))
        psd = np.abs(FS)**2/(FS.size * dx * dy * dt)
    return psd  


######### Make a gif
def makeGIF(imdir, basename, slicenums, imageext):
    images = [(imdir + basename + '_' + str(index) + imageext) for index in slicenums]
    filename = imdir+'../'+basename+'.gif'
    with open(imdir + basename+'_list.txt','w') as fil:
        for item in images:
            fil.write("%s\n" % item)
    os.chdir(imdir)
    os.system('convert @'+basename+'_list.txt '+filename)
########
def get_parameter_index_in_np_array(array, sub_array):
    sub_idx = np.array([], dtype=int)
    for i in sub_array:
        indices = np.where(array == i)[0]  # 获取所有匹配索引
        sub_idx = np.concatenate((sub_idx, indices))
    return sub_idx
cmap = plt.get_cmap("Spectral")

Q = {}
Q_3d = {}
qs = ["ni","bx","by","bz","uix","uiy","uiz","pi-xx","pi-yy","pi-zz","ex","ey","ez"]
qs = ["bx","by","bz","uix","uiy","uiz","ni"]
# qs = ["bx","by","bz","ex","ey","ez"]
dir = "data_imbalanced_highTimeResolution/"
# dir = "/Users/chuanpeng/research/alfven_turbulence/data_2kx_amp002/"
infoarr = loadinfo(dir)
nx = int(infoarr[0])
ny = int(infoarr[1])
nz = int(infoarr[2])
Lx = int(infoarr[3])
Ly = int(infoarr[4])
Lz = int(infoarr[5])
print(infoarr)

dx = Lx/nx
dy = Ly/ny
dz = Lz/nz
dt = 11/32/np.sqrt(3)#2.56


#mark
t_idx_start =200
t_idx_end = 222
t_idx_step = 2

calculate_w_k = 1
use_cwt_2DT_CPU = 0
use_cwt_2DT_GPU = 1
re_calc = 0

W_thermal_arr = np.array([])

xv = np.linspace(0,Lx,nx)-Lx/2.0
yv = np.linspace(0,Ly,ny)-Ly/2.0
zv = np.linspace(0,Lz,nz)-Lz/2.0
slicenums = []

for q in qs:
    slices = []
    for slice_ in range(t_idx_start, t_idx_end, t_idx_step):
            print(slice_)
            tmp = loadSlice(dir,q,slice_,nx,ny,nz)
            slices.append(tmp[:,:,:])
    Q_3d[q] = np.stack(slices, axis=-1)

variable = Q_3d['bx'][:,:,:,0]
vari_rms = np.sqrt(np.mean(variable**2) - np.mean(variable)**2)
print("vari_rms: ", vari_rms)

import math
def find_nearest_lattice(r1, direction, l):
    """
    在三维格点中，找到沿指定方向、距离起始格点r1为l的最近格点。
    
    参数:
        r1 (tuple/list): 起始格点坐标，需为3个整数，如(1, 2, 3)
        direction (tuple/list): 方向向量，需为3个数字，如(1.0, 1.0, 0.0)
        l (float/int): 距离，需为正数
    
    返回:
        tuple: 最近的格点坐标（3个整数）
    
    异常:
        ValueError: 输入参数不符合要求时抛出
    """
    # 验证起始格点r1的有效性（3个整数）
    if not (isinstance(r1, (tuple, list)) and len(r1) == 3 and all(isinstance(x, int) for x in r1)):
        raise ValueError("r1必须是包含3个整数的元组或列表")
    
    # 验证方向向量的有效性（3个数字）
    if not (isinstance(direction, (tuple, list)) and len(direction) == 3):
        raise ValueError("direction必须是包含3个数字的元组或列表")
    
    # 验证距离l的有效性（正数）
    if not (isinstance(l, (int, float)) and l > 0):
        raise ValueError("l必须是正数")
    
    # 解析方向向量分量
    vx, vy, vz = direction
    
    # 计算方向向量的模长（避免零向量）
    mod = math.sqrt(vx**2 + vy**2 + vz**2)
    if mod < 1e-10:  # 考虑浮点数精度误差
        raise ValueError("方向向量不能是零向量")
    
    # 归一化方向向量（得到单位向量）
    nx = vx / mod
    ny = vy / mod
    nz = vz / mod
    
    # 计算沿方向移动距离l后的目标点坐标（非格点）
    px = r1[0] + l * nx
    py = r1[1] + l * ny
    pz = r1[2] + l * nz
    
    # 四舍五入得到最近的格点（各分量取最近整数）
    nearest_lattice = (round(px), round(py), round(pz))
    
    return nearest_lattice

def calculate_curl(Bx, By, Bz, x, y, z):
    # 计算偏导数
    dBz_dy = np.gradient(Bz, y, axis=1)
    dBy_dz = np.gradient(By, z, axis=2)
    dBx_dz = np.gradient(Bx, z, axis=2)
    dBz_dx = np.gradient(Bz, x, axis=0)
    dBy_dx = np.gradient(By, x, axis=0)
    dBx_dy = np.gradient(Bx, y, axis=1)

    # 计算旋度
    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy

    return curl_x, curl_y, curl_z
# import numpy as np
# from numba import njit, prange
# from tqdm import tqdm
# from numba import get_num_threads, set_num_threads
# # set_num_threads(256)
# print(f"Numba默认线程数：{get_num_threads()}")

# @njit(fastmath=True, parallel=True)
# def _numba_batch_eig(jacobian_batch, use_symmetric):
#     n_total = jacobian_batch.shape[0]
#     # 预定义复数类型数组（固定为complex128）
#     eigenvals = np.zeros((n_total, 3), dtype=np.complex128)
#     eigenvecs = np.zeros((n_total, 3, 3), dtype=np.complex128)
    
#     for idx in prange(n_total):
#         jac = jacobian_batch[idx]
#         # 显式转为复数矩阵，避免类型混乱
#         jac_complex = jac.astype(np.complex128)
        
#         if use_symmetric:
#             # 处理对称矩阵：eigh返回实数，需转为复数
#             jac_sym = 0.5 * (jac_complex + jac_complex.T.conj())
#             evals_real, evecs_real = np.linalg.eigh(jac_sym)
#             # 关键：强制转换为complex128，与预定义数组类型匹配
#             evals = evals_real.astype(np.complex128)
#             evecs = evecs_real.astype(np.complex128)
#         else:
#             # 处理非对称矩阵：eig直接返回复数，无需额外转换
#             evals, evecs = np.linalg.eig(jac_complex)
        
#         # 赋值（此时类型完全匹配）
#         eigenvals[idx] = evals
#         eigenvecs[idx] = evecs
#     return eigenvals, eigenvecs


# def calc_vector_field_jacobian_eigen_numba(vector_field, hx, hy, hz, use_symmetric=False):
#     # 输入校验
#     if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
#         raise ValueError(f"矢量场必须是 (nx, ny, nz, 3)，当前形状 {vector_field.shape}")
#     if hx <= 0 or hy <= 0 or hz <= 0:
#         raise ValueError(f"空间步长必须为正数，当前 hx={hx}, hy={hy}, hz={hz}")
    
#     nx, ny, nz, _ = vector_field.shape
#     n_total = nx * ny * nz
    
#     # 一阶偏导数计算（确保输入为float64）
#     def first_deriv(arr, axis, h):
#         # 强制arr为float64，避免混合类型
#         arr_float = arr.astype(np.float64)
#         deriv = np.zeros_like(arr_float, dtype=np.float64)
#         n = arr_float.shape[axis]
        
#         # 内部点：中心差分
#         slices = [slice(None)] * 3
#         slices[axis] = slice(1, n-1)
#         slices_plus = [slice(None)] * 3
#         slices_plus[axis] = slice(2, n)
#         slices_minus = [slice(None)] * 3
#         slices_minus[axis] = slice(0, n-2)
#         deriv[tuple(slices)] = (arr_float[tuple(slices_plus)] - arr_float[tuple(slices_minus)]) / (2 * h)
        
#         # 首边界：向前差分
#         slices_first = [slice(None)] * 3
#         slices_first[axis] = 0
#         slices_first_plus = [slice(None)] * 3
#         slices_first_plus[axis] = 1
#         deriv[tuple(slices_first)] = (arr_float[tuple(slices_first_plus)] - arr_float[tuple(slices_first)]) / h
        
#         # 尾边界：向后差分
#         slices_last = [slice(None)] * 3
#         slices_last[axis] = -1
#         slices_last_minus = [slice(None)] * 3
#         slices_last_minus[axis] = -2
#         deriv[tuple(slices_last)] = (arr_float[tuple(slices_last)] - arr_float[tuple(slices_last_minus)]) / h
        
#         return deriv
    
#     # 计算偏导数（统一为float64）
#     d_Bx_dx = first_deriv(vector_field[..., 0], axis=0, h=hx)
#     d_Bx_dy = first_deriv(vector_field[..., 0], axis=1, h=hy)
#     d_Bx_dz = first_deriv(vector_field[..., 0], axis=2, h=hz)
#     d_By_dx = first_deriv(vector_field[..., 1], axis=0, h=hx)
#     d_By_dy = first_deriv(vector_field[..., 1], axis=1, h=hy)
#     d_By_dz = first_deriv(vector_field[..., 1], axis=2, h=hz)
#     d_Bz_dx = first_deriv(vector_field[..., 2], axis=0, h=hx)
#     d_Bz_dy = first_deriv(vector_field[..., 2], axis=1, h=hy)
#     d_Bz_dz = first_deriv(vector_field[..., 2], axis=2, h=hz)
    
#     # 构建批量雅可比矩阵（统一为float64）
#     d_Bx_dx_flat = d_Bx_dx.reshape(-1, 1)
#     d_Bx_dy_flat = d_Bx_dy.reshape(-1, 1)
#     d_Bx_dz_flat = d_Bx_dz.reshape(-1, 1)
#     d_By_dx_flat = d_By_dx.reshape(-1, 1)
#     d_By_dy_flat = d_By_dy.reshape(-1, 1)
#     d_By_dz_flat = d_By_dz.reshape(-1, 1)
#     d_Bz_dx_flat = d_Bz_dx.reshape(-1, 1)
#     d_Bz_dy_flat = d_Bz_dy.reshape(-1, 1)
#     d_Bz_dz_flat = d_Bz_dz.reshape(-1, 1)
    
#     jacobian_batch = np.stack([
#         np.hstack([d_Bx_dx_flat, d_Bx_dy_flat, d_Bx_dz_flat]),
#         np.hstack([d_By_dx_flat, d_By_dy_flat, d_By_dz_flat]),
#         np.hstack([d_Bz_dx_flat, d_Bz_dy_flat, d_Bz_dz_flat])
#     ], axis=1).astype(np.float64)  # 强制float64，避免类型混合
    
#     # 计算特征值/向量（带进度条）
#     with tqdm(total=1, desc="Numba批量计算特征值（类型统一）") as pbar:
#         eigenvals_batch, eigenvecs_batch = _numba_batch_eig(jacobian_batch, use_symmetric)
#         pbar.update(1)
    
#     # 重塑回原三维形状
#     eigenvals = eigenvals_batch.reshape(nx, ny, nz, 3)
#     eigenvecs = eigenvecs_batch.reshape(nx, ny, nz, 3, 3)
    
#     return eigenvals, eigenvecs
import numpy as np
from tqdm import tqdm


def calc_vector_field_jacobian_eigen_vectorized(vector_field, hx, hy, hz, use_symmetric=False):
    """
    向量化版本：计算三维矢量场的雅可比矩阵特征值和特征向量（无Python循环）。
    
    参数:
        vector_field (np.ndarray): 三维矢量场，形状 (nx, ny, nz, 3)。
        hx, hy, hz (float): 各方向空间步长（正数）。
        use_symmetric (bool): 若为True，假设雅可比矩阵对称，用eigh加速（快2~3倍）。
    
    返回:
        eigenvals (np.ndarray): 特征值数组 (nx, ny, nz, 3)（复数）。
        eigenvecs (np.ndarray): 特征向量数组 (nx, ny, nz, 3, 3)（复数）。
    """
    # 输入校验
    if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
        raise ValueError(f"矢量场必须是 (nx, ny, nz, 3)，当前形状 {vector_field.shape}")
    if hx <= 0 or hy <= 0 or hz <= 0:
        raise ValueError(f"空间步长必须为正数，当前 hx={hx}, hy={hy}, hz={hz}")
    
    nx, ny, nz, _ = vector_field.shape
    n_total = nx * ny * nz  # 总网格点数
    eigenvals = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    eigenvecs = np.zeros((nx, ny, nz, 3, 3), dtype=np.complex128)
    
    # --------------------------
    # 1. 向量化计算一阶偏导数（同原逻辑）
    # --------------------------
    def first_deriv(arr, axis, h):
        """向量化计算沿指定轴的一阶偏导数（中心/向前/向后差分）"""
        deriv = np.zeros_like(arr, dtype=np.float64)
        n = arr.shape[axis]
        # 内部点：中心差分 (f[i+1] - f[i-1])/(2h)
        slices = [slice(None)] * 3
        slices[axis] = slice(1, n-1)
        slices_plus = [slice(None)] * 3
        slices_plus[axis] = slice(2, n)
        slices_minus = [slice(None)] * 3
        slices_minus[axis] = slice(0, n-2)
        deriv[tuple(slices)] = (arr[tuple(slices_plus)] - arr[tuple(slices_minus)]) / (2 * h)
        # 首边界：向前差分 (f[1] - f[0])/h
        slices_first = [slice(None)] * 3
        slices_first[axis] = 0
        slices_first_plus = [slice(None)] * 3
        slices_first_plus[axis] = 1
        deriv[tuple(slices_first)] = (arr[tuple(slices_first_plus)] - arr[tuple(slices_first)]) / h
        # 尾边界：向后差分 (f[-1] - f[-2])/h
        slices_last = [slice(None)] * 3
        slices_last[axis] = -1
        slices_last_minus = [slice(None)] * 3
        slices_last_minus[axis] = -2
        deriv[tuple(slices_last)] = (arr[tuple(slices_last)] - arr[tuple(slices_last_minus)]) / h
        return deriv
    
    # 计算9个偏导数（向量化操作，无循环）
    d_Bx_dx = first_deriv(vector_field[..., 0], axis=0, h=hx)
    d_Bx_dy = first_deriv(vector_field[..., 0], axis=1, h=hy)
    d_Bx_dz = first_deriv(vector_field[..., 0], axis=2, h=hz)
    d_By_dx = first_deriv(vector_field[..., 1], axis=0, h=hx)
    d_By_dy = first_deriv(vector_field[..., 1], axis=1, h=hy)
    d_By_dz = first_deriv(vector_field[..., 1], axis=2, h=hz)
    d_Bz_dx = first_deriv(vector_field[..., 2], axis=0, h=hx)
    d_Bz_dy = first_deriv(vector_field[..., 2], axis=1, h=hy)
    d_Bz_dz = first_deriv(vector_field[..., 2], axis=2, h=hz)
    
    # --------------------------
    # 2. 向量化构建批量雅可比矩阵
    # --------------------------
    # 将所有偏导数从 (nx, ny, nz) 重塑为 (n_total, 1)，便于堆叠
    d_Bx_dx_flat = d_Bx_dx.reshape(-1, 1)  # 形状 (n_total, 1)
    d_Bx_dy_flat = d_Bx_dy.reshape(-1, 1)
    d_Bx_dz_flat = d_Bx_dz.reshape(-1, 1)
    d_By_dx_flat = d_By_dx.reshape(-1, 1)
    d_By_dy_flat = d_By_dy.reshape(-1, 1)
    d_By_dz_flat = d_By_dz.reshape(-1, 1)
    d_Bz_dx_flat = d_Bz_dx.reshape(-1, 1)
    d_Bz_dy_flat = d_Bz_dy.reshape(-1, 1)
    d_Bz_dz_flat = d_Bz_dz.reshape(-1, 1)
    
    # 堆叠成 (n_total, 3, 3) 的批量雅可比矩阵：每个元素是一个网格点的3×3矩阵
    # 结构：[[∂Bx/∂x, ∂Bx/∂y, ∂Bx/∂z],
    #        [∂By/∂x, ∂By/∂y, ∂By/∂z],
    #        [∂Bz/∂x, ∂Bz/∂y, ∂Bz/∂z]]
    jacobian_batch = np.stack([
        np.hstack([d_Bx_dx_flat, d_Bx_dy_flat, d_Bx_dz_flat]),  # 第0行：Bx的三个偏导数
        np.hstack([d_By_dx_flat, d_By_dy_flat, d_By_dz_flat]),  # 第1行：By的三个偏导数
        np.hstack([d_Bz_dx_flat, d_Bz_dy_flat, d_Bz_dz_flat])   # 第2行：Bz的三个偏导数
    ], axis=1)  # 最终形状：(n_total, 3, 3)
    
    # --------------------------
    # 3. 向量化批量计算特征值/向量
    # --------------------------
    # 进度条（监控批量计算进度，可选）
    with tqdm(total=1, desc="批量计算特征值") as pbar:
        if use_symmetric:
            # 若矩阵对称，用eigh加速（速度快2~3倍，数值更稳定）
            # 对称化处理：J_sym = (J + J.T)/2
            jacobian_batch = 0.5 * (jacobian_batch + np.transpose(jacobian_batch, axes=(0, 2, 1)))
            eigenvals_batch, eigenvecs_batch = np.linalg.eigh(jacobian_batch)
        else:
            # 通用非对称矩阵，用eig计算
            eigenvals_batch, eigenvecs_batch = np.linalg.eig(jacobian_batch)
        pbar.update(1)
    
    # --------------------------
    # 4. 重塑回原三维网格形状
    # --------------------------
    eigenvals = eigenvals_batch.reshape(nx, ny, nz, 3)  # 从 (n_total, 3) → (nx, ny, nz, 3)
    eigenvecs = eigenvecs_batch.reshape(nx, ny, nz, 3, 3)  # 从 (n_total, 3, 3) → (nx, ny, nz, 3, 3)
    
    return eigenvals, eigenvecs
# import numpy as np
# import concurrent.futures
# from concurrent.futures import ProcessPoolExecutor
# import itertools
# import os
# from tqdm import tqdm  # 导入进度条库


# def calc_vector_field_jacobian_eigen(vector_field, hx, hy, hz, max_workers=None):
#     """
#     计算三维矢量场的雅可比矩阵的特征值和特征向量（多线程版本，带进度条）。
    
#     参数:
#         vector_field (np.ndarray): 三维矢量场数据，形状为 (nx, ny, nz, 3)。
#         hx, hy, hz (float): 各方向空间步长（正数）。
#         max_workers (int, optional): 线程池最大线程数，默认使用CPU核心数。
    
#     返回:
#         eigenvals (np.ndarray): 特征值数组 (nx, ny, nz, 3)（可能为复数）。
#         eigenvecs (np.ndarray): 特征向量数组 (nx, ny, nz, 3, 3)（可能为复数）。
#     """
#     # 输入校验
#     if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
#         raise ValueError(f"矢量场必须是形状为 (nx, ny, nz, 3) 的4维数组，当前形状为 {vector_field.shape}")
#     if hx <= 0 or hy <= 0 or hz <= 0:
#         raise ValueError(f"空间步长必须为正数，当前 hx={hx}, hy={hy}, hz={hz}")
    
#     nx, ny, nz, _ = vector_field.shape
#     total_points = nx * ny * nz  # 总网格点数（用于进度条）
#     eigenvals = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
#     eigenvecs = np.zeros((nx, ny, nz, 3, 3), dtype=np.complex128)
    
#     # 一阶偏导数计算函数（同原逻辑）
#     def first_deriv(arr, axis, h):
#         deriv = np.zeros_like(arr, dtype=np.float64)
#         n = arr.shape[axis]
#         # 内部点：中心差分
#         slices = [slice(None)] * 3
#         slices[axis] = slice(1, n-1)
#         slices_plus = [slice(None)] * 3
#         slices_plus[axis] = slice(2, n)
#         slices_minus = [slice(None)] * 3
#         slices_minus[axis] = slice(0, n-2)
#         deriv[tuple(slices)] = (arr[tuple(slices_plus)] - arr[tuple(slices_minus)]) / (2 * h)
#         # 首边界：向前差分
#         slices_first = [slice(None)] * 3
#         slices_first[axis] = 0
#         slices_first_plus = [slice(None)] * 3
#         slices_first_plus[axis] = 1
#         deriv[tuple(slices_first)] = (arr[tuple(slices_first_plus)] - arr[tuple(slices_first)]) / h
#         # 尾边界：向后差分
#         slices_last = [slice(None)] * 3
#         slices_last[axis] = -1
#         slices_last_minus = [slice(None)] * 3
#         slices_last_minus[axis] = -2
#         deriv[tuple(slices_last)] = (arr[tuple(slices_last)] - arr[tuple(slices_last_minus)]) / h
#         return deriv
    
#     # 计算所有偏导数（同原逻辑）
#     d_Bx_dx = first_deriv(vector_field[..., 0], axis=0, h=hx)
#     d_Bx_dy = first_deriv(vector_field[..., 0], axis=1, h=hy)
#     d_Bx_dz = first_deriv(vector_field[..., 0], axis=2, h=hz)
#     d_By_dx = first_deriv(vector_field[..., 1], axis=0, h=hx)
#     d_By_dy = first_deriv(vector_field[..., 1], axis=1, h=hy)
#     d_By_dz = first_deriv(vector_field[..., 1], axis=2, h=hz)
#     d_Bz_dx = first_deriv(vector_field[..., 2], axis=0, h=hx)
#     d_Bz_dy = first_deriv(vector_field[..., 2], axis=1, h=hy)
#     d_Bz_dz = first_deriv(vector_field[..., 2], axis=2, h=hz)
    
#     # 单个网格点的处理函数（供线程调用）
#     def process_point(i, j, k):
#         # 构建雅可比矩阵
#         jacobian = np.array([
#             [d_Bx_dx[i, j, k], d_Bx_dy[i, j, k], d_Bx_dz[i, j, k]],
#             [d_By_dx[i, j, k], d_By_dy[i, j, k], d_By_dz[i, j, k]],
#             [d_Bz_dx[i, j, k], d_Bz_dy[i, j, k], d_Bz_dz[i, j, k]]
#         ], dtype=np.float64)
#         # 计算特征值和特征向量
#         evals, evecs = np.linalg.eig(jacobian)  # evecs每一列是特征向量
#         # 写入结果（线程安全）
#         eigenvals[i, j, k] = evals
#         eigenvecs[i, j, k] = evecs
#         return  # 仅用于触发进度条更新
    
#     # 生成所有网格点的索引 (i,j,k)，并包装进度条
#     indices = itertools.product(range(nx), range(ny), range(nz))
#     # 用tqdm包装迭代器，显示进度（total指定总任务数）
#     pbar = tqdm(indices, total=total_points, desc="计算雅可比矩阵特征值", unit="点")
    
#     # 多线程处理所有点（结合进度条）
#     max_workers = max_workers or os.cpu_count()
#     print(max_workers)
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:  # 用ProcessPoolExecutor替代ThreadPoolExecutor
#         list(executor.map(lambda x: process_point(x[0], x[1], x[2]), pbar))
    
#     return eigenvals, eigenvecs
# def calc_vector_field_jacobian_eigen(vector_field, hx, hy, hz):
#     """
#     计算三维矢量场的雅可比矩阵（一阶梯度矩阵）的特征值和特征向量。
    
#     参数:
#         vector_field (np.ndarray): 三维矢量场数据，形状为 (nx, ny, nz, 3)，
#                                  最后一个维度对应3个分量（如 [Bx, By, Bz]）。
#         hx (float): x方向空间步长（相邻网格x坐标差），必须为正数。
#         hy (float): y方向空间步长，必须为正数。
#         hz (float): z方向空间步长，必须为正数。
    
#     返回:
#         eigenvals (np.ndarray): 每个网格点的雅可比矩阵特征值，形状为 (nx, ny, nz, 3)，
#                                特征值可能为复数（因雅可比矩阵不一定对称）。
#         eigenvecs (np.ndarray): 每个网格点的雅可比矩阵特征向量，形状为 (nx, ny, nz, 3, 3)，
#                                其中eigenvecs[i,j,k,:,m]对应第m个特征值的特征向量，
#                                特征向量可能为复数。
    
#     数值方法:
#         - 内部网格点：采用**中心差分**（二阶精度）计算一阶偏导数。
#         - 边界网格点：采用**向前/向后差分**（一阶精度）避免越界。
#     """
#     # 输入校验
#     if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
#         raise ValueError(f"矢量场必须是形状为 (nx, ny, nz, 3) 的4维数组，当前形状为 {vector_field.shape}")
#     if hx <= 0 or hy <= 0 or hz <= 0:
#         raise ValueError(f"空间步长必须为正数，当前 hx={hx}, hy={hy}, hz={hz}")
    
#     nx, ny, nz, _ = vector_field.shape
#     # 存储特征值（可能为复数）
#     eigenvals = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
#     # 存储特征向量（最后两个维度：3个特征向量，每个为3维向量）
#     eigenvecs = np.zeros((nx, ny, nz, 3, 3), dtype=np.complex128)
    
#     # 定义一阶偏导数计算函数（处理边界）
#     def first_deriv(arr, axis, h):
#         """
#         计算数组沿指定轴的一阶偏导数。
#         参数:
#             arr: 输入数组（形状为 (nx, ny, nz)）
#             axis: 求导轴（0=x, 1=y, 2=z）
#             h: 该轴的空间步长
#         返回:
#             deriv: 与arr同形状的一阶导数数组
#         """
#         deriv = np.zeros_like(arr, dtype=np.float64)
#         n = arr.shape[axis]  # 该轴的网格点数
        
#         # 内部点：中心差分 (f[i+1] - f[i-1])/(2h)
#         slices = [slice(None)] * 3
#         slices[axis] = slice(1, n-1)  # 内部索引
#         slices_plus = [slice(None)] * 3
#         slices_plus[axis] = slice(2, n)  # i+1
#         slices_minus = [slice(None)] * 3
#         slices_minus[axis] = slice(0, n-2)  # i-1
#         deriv[tuple(slices)] = (arr[tuple(slices_plus)] - arr[tuple(slices_minus)]) / (2 * h)
        
#         # 边界点：首边界用向前差分 (f[1] - f[0])/h
#         slices_first = [slice(None)] * 3
#         slices_first[axis] = 0
#         slices_first_plus = [slice(None)] * 3
#         slices_first_plus[axis] = 1
#         deriv[tuple(slices_first)] = (arr[tuple(slices_first_plus)] - arr[tuple(slices_first)]) / h
        
#         # 边界点：尾边界用向后差分 (f[-1] - f[-2])/h
#         slices_last = [slice(None)] * 3
#         slices_last[axis] = -1
#         slices_last_minus = [slice(None)] * 3
#         slices_last_minus[axis] = -2
#         deriv[tuple(slices_last)] = (arr[tuple(slices_last)] - arr[tuple(slices_last_minus)]) / h
        
#         return deriv
    
#     # 计算雅可比矩阵的9个元素（3个分量×3个方向的偏导数）
#     # Bx的偏导数：∂Bx/∂x, ∂Bx/∂y, ∂Bx/∂z
#     d_Bx_dx = first_deriv(vector_field[..., 0], axis=0, h=hx)
#     d_Bx_dy = first_deriv(vector_field[..., 0], axis=1, h=hy)
#     d_Bx_dz = first_deriv(vector_field[..., 0], axis=2, h=hz)
    
#     # By的偏导数：∂By/∂x, ∂By/∂y, ∂By/∂z
#     d_By_dx = first_deriv(vector_field[..., 1], axis=0, h=hx)
#     d_By_dy = first_deriv(vector_field[..., 1], axis=1, h=hy)
#     d_By_dz = first_deriv(vector_field[..., 1], axis=2, h=hz)
    
#     # Bz的偏导数：∂Bz/∂x, ∂Bz/∂y, ∂Bz/∂z
#     d_Bz_dx = first_deriv(vector_field[..., 2], axis=0, h=hx)
#     d_Bz_dy = first_deriv(vector_field[..., 2], axis=1, h=hy)
#     d_Bz_dz = first_deriv(vector_field[..., 2], axis=2, h=hz)
    
#     # 遍历每个网格点，构建雅可比矩阵并计算特征值和特征向量
#     for i in range(nx):
#         print(i)
#         for j in range(ny):
#             for k in range(nz):
                
#                 # 构建当前点的3×3雅可比矩阵
#                 jacobian = np.array([
#                     [d_Bx_dx[i, j, k], d_Bx_dy[i, j, k], d_Bx_dz[i, j, k]],
#                     [d_By_dx[i, j, k], d_By_dy[i, j, k], d_By_dz[i, j, k]],
#                     [d_Bz_dx[i, j, k], d_Bz_dy[i, j, k], d_Bz_dz[i, j, k]]
#                 ], dtype=np.float64)
#                 symmetric_jacobian = 0.5 * (jacobian + jacobian.T)  # 对称化雅可比矩阵，以防奇异
                
#                 # 计算特征值和特征向量（雅可比矩阵可能非对称，用eig求复特征值/向量）
#                 evals, evecs = np.linalg.eig(jacobian)  # evecs每一列是一个特征向量
#                 eigenvals[i, j, k] = evals
#                 # 存储特征向量（保持与特征值的对应关系）
#                 eigenvecs[i, j, k] = evecs  # 形状为(3,3)，每列对应一个特征值的特征向量
    
#     return eigenvals, eigenvecs
if __name__ == '__main__':
    epoch = 15
    x = np.linspace(-32, 32, 256)
    y = np.linspace(-32, 32, 256)
    z = np.linspace(-32, 32, 256)
    for epoch in range(1):
        t_idx = 0
        Bx, By, Bz = Q_3d['bx'][:,:,:,t_idx],Q_3d['by'][:,:,:,t_idx],Q_3d['bz'][:,:,:,t_idx]
       
        uix, uiy, uiz = Q_3d['uix'][:,:,:,t_idx],Q_3d['uiy'][:,:,:,t_idx],Q_3d['uiz'][:,:,:,t_idx]
        ni = Q_3d['ni'][:,:,:,t_idx]


        Jx, Jy, Jz = calculate_curl(Bx, By, Bz, x, y, z)
        uex, uey, uez = uix-Jx/ni, uiy-Jy/ni, uiz-Jz/ni
        #t_arr = np.linspace(t_idx_start, t_idx_end, (t_idx_end-t_idx_start)//t_idx_step)
        interp_uex_single = RegularGridInterpolator((x, y, z), uex)
        interp_uey_single = RegularGridInterpolator((x, y, z), uey)
        interp_uez_single = RegularGridInterpolator((x, y, z), uez)
    
        B_vec = np.stack([Bx,By,Bz],axis=-1)
        u_vec = np.stack([uix,uiy,uiz],axis=-1)
        print(B_vec.shape)
  
        
        interp_Bx = RegularGridInterpolator((x, y, z), Bx)
        interp_By = RegularGridInterpolator((x, y, z), By)
        interp_Bz = RegularGridInterpolator((x, y, z), Bz)
        def trace_fieldline_3D_single(start, ds=0.01, max_steps=1000, direction=1):
            path = [start]
            pos = np.array(start)
            for _ in range(max_steps):
                Bx_val = interp_Bx([pos[0], pos[1], pos[2]])
                By_val = interp_By([pos[0], pos[1], pos[2]])
                Bz_val = interp_Bz([pos[0], pos[1], pos[2]])
                B_vec_tmp = np.array([Bx_val, By_val, Bz_val])
                B_norm = np.linalg.norm(B_vec_tmp)
                if B_norm < 1e-10:
                    break
                pos = pos.squeeze() + direction * ds * B_vec_tmp.squeeze() / B_norm
                # print(pos)
                if not (x[0] <= pos[0] <= x[-1] and y[0] <= pos[1] <= y[-1] and z[0] <= pos[2] <= z[-1]):
                    break 
                path.append(pos.copy())
            return np.array(path)
            #得到完整的path之后再去采样
        
        # import numpy as np

        # import numpy as np

        def trace_fieldline_3D_optimized(
            start, 
            x_grid, y_grid, z_grid,  # 磁场网格范围（需为网格的实际坐标数组，如x=np.linspace(xmin,xmax,nx)）
            interp_Bx, interp_By, interp_Bz,  # 磁场插值函数（如RegularGridInterpolator）
            ds_init=0.01,  
            max_steps=2000, 
            direction=1,  
            atol=1e-6,  
            rtol=1e-4,  
            min_ds=1e-5,  
            max_ds=0.1,   
            B_min=1e-10   
        ):
            # 提取网格边界（用于快速检查）
            x_min, x_max = x_grid[0], x_grid[-1]
            y_min, y_max = y_grid[0], y_grid[-1]
            z_min, z_max = z_grid[0], z_grid[-1]
            
            # 检查起点是否在网格内
            start = np.asarray(start, dtype=np.float64)
            if not (x_min <= start[0] <= x_max and y_min <= start[1] <= y_max and z_min <= start[2] <= z_max):
                raise ValueError("起点超出网格范围！")
            
            path = [start.copy()]
            pos = start.copy()
            ds = ds_init
            
            # 辅助函数：检查位置是否在网格内（含微小误差容忍，避免浮点精度问题）
            def in_bounds(p):
                return (
                    (x_min - 1e-12 <= p[0] <= x_max + 1e-12) and
                    (y_min - 1e-12 <= p[1] <= y_max + 1e-12) and
                    (z_min - 1e-12 <= p[2] <= z_max + 1e-12)
                )
            
            # 辅助函数：获取磁场（含边界检查）
            def get_B(p):
                if not in_bounds(p):
                    return None  # 位置越界，返回无效
                try:
                    Bx = interp_Bx(p)
                    By = interp_By(p)
                    Bz = interp_Bz(p)
                    return np.array([Bx, By, Bz], dtype=np.float64).squeeze()
                except ValueError:
                    # 极端情况：浮点误差导致的越界，强制返回无效
                    return None
            
            # RK4步长计算（含中间位置检查）
            def rk4_step(pos, ds):
                B0 = get_B(pos)
                if B0 is None:
                    return None, 0.0  # 初始位置越界（理论上不应发生，因主循环会检查）
                norm_B0 = np.linalg.norm(B0)
                if norm_B0 < B_min:
                    return None, norm_B0
                
                # 计算k1
                k1 = ds * direction * B0 / norm_B0
                
                # 计算pos1和k2（检查pos1是否越界）
                pos1 = pos + k1 / 2
                B1 = get_B(pos1)
                if B1 is None:
                    return None, 0.0  # 中间位置越界
                norm_B1 = np.linalg.norm(B1)
                if norm_B1 < B_min:
                    return None, norm_B1
                k2 = ds * direction * B1 / norm_B1
                
                # 计算pos2和k3（检查pos2是否越界）
                pos2 = pos + k2 / 2
                B2 = get_B(pos2)
                if B2 is None:
                    return None, 0.0
                norm_B2 = np.linalg.norm(B2)
                if norm_B2 < B_min:
                    return None, norm_B2
                k3 = ds * direction * B2 / norm_B2
                
                # 计算pos3和k4（检查pos3是否越界）
                pos3 = pos + k3
                B3 = get_B(pos3)
                if B3 is None:
                    return None, 0.0
                norm_B3 = np.linalg.norm(B3)
                if norm_B3 < B_min:
                    return None, norm_B3
                k4 = ds * direction * B3 / norm_B3
                
                # 计算新位置和误差
                pos_new = pos + (k1 + 2*k2 + 2*k3 + k4) / 6
                pos_rk3 = pos + (k1 + 4*k2 + k3) / 6  # 低阶估计用于误差
                error = np.linalg.norm(pos_new - pos_rk3)
                return pos_new, error
            
            # 主追踪循环
            for _ in range(max_steps):
                pos_new, error = rk4_step(pos, ds)
                
                # 处理中间步骤越界或磁场过弱的情况
                if pos_new is None:
                    # 尝试用更小的步长（1/2）重试一次
                    ds_half = ds / 2
                    if ds_half < min_ds:
                        break  # 步长已达最小值，无法再小
                    pos_new, error = rk4_step(pos, ds_half)
                    if pos_new is None:
                        break  # 小步长仍失败，停止
                    else:
                        ds = ds_half  # 更新步长为小步长
                
                # 检查最终位置是否在网格内（双重保险）
                if not in_bounds(pos_new):
                    break
                
                # 自适应调整步长
                if error < 1e-12:
                    ds_new = min(2 * ds, max_ds)
                else:
                    ds_new = ds * ( (atol + rtol * np.linalg.norm(pos_new)) / error ) **0.25
                    ds_new = np.clip(ds_new, min_ds, max_ds)
                
                # 更新轨迹和位置
                pos = pos_new 
                path.append(pos.copy())
                ds = ds_new
            
            return np.array(path)
        def get_position_start(center, ds=0.002, max_steps=3000, step:int = 200):
            line_0 = np.array(trace_fieldline_3D_single(center, 
                                                        ds=ds, max_steps=max_steps))
            line_0_minus = np.array(trace_fieldline_3D_single(center,ds=ds, max_steps=max_steps, direction=-1))
            position_start=line_0_minus[::-step,:].tolist()+line_0[::step,:].tolist()
            return position_start
        # line_test = trace_fieldline_3D_optimized([0.,0.,0.], x, y, z, interp_Bx, interp_By, interp_Bz,direction=1, max_steps=10000)
        # fig = plt.figure(figsize=(10, 7))
        # plt.streamplot(x,y,Bx[:,:,nz//2].T, By[:,:,nz//2].T, density=1.5, linewidth=1, color='black', broken_streamlines=False)
        # plt.plot(line_test[:,0],line_test[:,1],color='red')
        # plt.savefig("./figures/test.png")
        # print("------TEST FINISHED----------")
        #算的时候得用梯度矩阵计算
        if not os.path.exists("./data/lambda_mat_B.npy"):
            lambda_mat_B, eigen_vec_B = calc_vector_field_jacobian_eigen_vectorized(B_vec,hx=0.25,hy=0.25,hz=0.25)
            lambda_mat_u, eigen_vec_u = calc_vector_field_jacobian_eigen_vectorized(u_vec,hx=0.25,hy=0.25,hz=0.25)
            os.mkdir("./data/")
            np.save("./data/lambda_mat_B.npy",lambda_mat_B)
            np.save("./data/eigen_vec_B.npy",eigen_vec_B)
            np.save("./data/lambda_mat_u.npy",lambda_mat_u)
            np.save("./data/eigen_vec_u.npy",eigen_vec_u)
        else:
            lambda_mat_B = np.load("./data/lambda_mat_B.npy")
            eigen_vec_B = np.load("./data/eigen_vec_B.npy")
            lambda_mat_u = np.load("./data/lambda_mat_u.npy")
            eigen_vec_u = np.load("./data/eigen_vec_u.npy")
        # print("test:",trace_fieldline_3D_single(np.array([1.241,-2.014,-12.856]),ds=0.002, max_steps=5000,direction=1))
        line_test = trace_fieldline_3D_single(np.array([-8.128,8.772,22.624]),ds=0.001, max_steps=2000,direction=1)
        print(line_test[np.abs(line_test[:,2]-23.118).argmin(),:])
        x_arr = np.linspace(-15,15,31)
        y_arr = np.linspace(-15,15,31)
        ve_grad_perp_max_mat = np.zeros((len(x_arr),len(y_arr)))
        ve_grad_para_max_mat = np.zeros((len(x_arr),len(y_arr)))
        ve_grad_max_mat = np.zeros((len(x_arr),len(y_arr)))
        # for i in range(24,27):
        #     print(i)
        #     for j in range(7,10):
        #         start_line_test = get_position_start(np.array([x_arr[i],y_arr[j],0]),ds=0.02, max_steps=1500,step=50)
        #         vex_lst, vey_lst, vez_lst = [], [], []
        #         bx_lst, by_lst, bz_lst = [], [], []

                
        #         for pos_tmp in start_line_test:
        #             vex_tmp = interp_uex_single(np.array(pos_tmp))
        #             vey_tmp = interp_uey_single(np.array(pos_tmp))
        #             vez_tmp = interp_uez_single(np.array(pos_tmp))
        #             bx_tmp, by_tmp, bz_tmp = interp_Bx(np.array(pos_tmp))[0], interp_By(np.array(pos_tmp))[0], interp_Bz(np.array(pos_tmp))[0]
        #             b_vec_norm = np.array([bx_tmp,by_tmp,bz_tmp])/np.linalg.norm(np.array([bx_tmp,by_tmp,bz_tmp]))
        #             vex_lst.append(vex_tmp)
        #             vey_lst.append(vey_tmp)
        #             vez_lst.append(vez_tmp)
        #             bx_lst.append(b_vec_norm[0])
        #             by_lst.append(b_vec_norm[1])
        #             bz_lst.append(b_vec_norm[2])


                
        #         vex_grad = np.gradient(np.array(vex_lst).squeeze())
        #         vey_grad = np.gradient(np.array(vey_lst).squeeze())
        #         vez_grad = np.gradient(np.array(vez_lst).squeeze())
        #         bx_arr, by_arr, bz_arr = np.array(bx_lst).squeeze(), np.array(by_lst).squeeze(), np.array(bz_lst).squeeze()

        #         ve_grad_para = (vex_grad*bx_arr+vey_grad*by_arr+vez_grad*bz_arr).reshape(-1,1)*np.column_stack((bx_arr,by_arr,bz_arr))
        #         # print(ve_grad_para.shape)

        #         ve_grad_perp = np.column_stack((vex_grad,vey_grad,vez_grad))-ve_grad_para
        #         ve_grad_perp_norm = np.linalg.norm(ve_grad_perp,axis=1)
        #         ve_grad_para_norm = np.linalg.norm(ve_grad_para,axis=1)
        #         ve_grad_perp_max_mat[i,j] = ve_grad_perp_norm[1:-1].max()
        #         ve_grad_para_max_mat[i,j] = ve_grad_para_norm[1:-1].max()
        #         ve_grad_max_mat[i,j] = np.sqrt(vex_grad**2+vey_grad**2+vez_grad**2)[1:-1].max()
        #         #print(i,j,ve_grad_perp_norm.max(),ve_grad_perp_norm.argmax())
        #         print(i,j,ve_grad_perp_norm[1:-1].max(),ve_grad_perp_norm[1:-1].argmax(),'ve_vec: ',vex_grad[ve_grad_perp_norm.argmax()],vey_grad[ve_grad_perp_norm.argmax()],vez_grad[ve_grad_perp_norm.argmax()],'B_vec: ',
        #               bx_arr[ve_grad_perp_norm.argmax()],by_arr[ve_grad_perp_norm.argmax()],bz_arr[ve_grad_perp_norm.argmax()],
        #               ve_grad_para_norm[ve_grad_perp_norm.argmax()])





        # plt.figure(figsize=(25, 7))
        # fig, axes=plt.subplots(1,3,figsize=(25, 7))

        # # print()
        # ax = axes[0]
        # pclr=ax.pcolormesh(x_arr,y_arr,ve_grad_max_mat.T,cmap='jet')
        # cbar=plt.colorbar(pclr,ax=ax)
        # ax.set_xlabel("x[di]", fontsize=15)
        # ax.set_ylabel("y[di]", fontsize=15)
        # cbar.set_label(r'$(\Delta v_{e})_{max}$', fontsize=15)
        # ax.set_title(fr"max of $(\Delta v_{{e}})_{{max}}$={ve_grad_max_mat.max():.3f}")

        # ax = axes[1]
        # pclr=ax.pcolormesh(x_arr,y_arr,ve_grad_perp_max_mat.T,cmap='jet')
        # cbar=plt.colorbar(pclr,ax=ax)
        # ax.set_xlabel("x[di]", fontsize=15)
        # ax.set_ylabel("y[di]", fontsize=15)
        # cbar.set_label(r'$(\Delta v_{e,\perp})_{max}$', fontsize=15)
        # ax.set_title(fr"max of $(\Delta v_{{e,\perp}})_{{max}}$={ve_grad_perp_max_mat.max():.3f}")

        # ax = axes[2]
        # pclr=ax.pcolormesh(x_arr,y_arr,ve_grad_para_max_mat.T,cmap='jet')
        # cbar=plt.colorbar(pclr,ax=ax)
        # ax.set_xlabel("x[di]", fontsize=15)
        # ax.set_ylabel("y[di]", fontsize=15)
        # cbar.set_label(r'$(\Delta v_{e,\parallel})_{max}$', fontsize=15)
        # ax.set_title(fr"max of $(\Delta v_{{e,\parallel}})_{{max}}$={ve_grad_para_max_mat.max():.3f}")
        # plt.suptitle(fr"t={t_idx_start*0.2:.2f}$\omega_{{ci}}^{{-1}}$", fontsize=18)
        # plt.savefig(f"./figures/ve_grad_{t_idx_start}.png")
        # plt.close(fig)






        condition = np.where(np.abs(line_test[:,2]+12.035)<0.005)
        condition_2 = np.where(np.abs(line_test[:,2]+8)<0.005)
        print(line_test[condition,:])
        print(line_test[condition_2,:])
        C = 5
        print(lambda_mat_B.shape)
        print(lambda_mat_B[0,0,0,:])
        sort_indices = np.argsort(np.abs(np.real(lambda_mat_B)),axis=3)
        sort_indices_u = np.argsort(np.abs(np.real(lambda_mat_u)),axis=3)
        print(sort_indices[0,0,0,:])
        lambda_mat_B_sorted = np.take_along_axis(lambda_mat_B,sort_indices,axis=3)
        lambda_mat_u_sorted = np.take_along_axis(lambda_mat_u,sort_indices_u,axis=3)
        eig_vec_B_sorted = np.take_along_axis(eigen_vec_B,sort_indices[...,np.newaxis],axis=3)
        eig_vec_u_sorted = np.take_along_axis(eigen_vec_u,sort_indices_u[...,np.newaxis],axis=3)
        n_vec_B = np.cross(eig_vec_B_sorted[:,:,:,1,:],eig_vec_B_sorted[:,:,:,2,:],axis=3)
        n_vec_u= np.cross(eig_vec_u_sorted[:,:,:,1,:],eig_vec_u_sorted[:,:,:,2,:],axis=3)
        print(eig_vec_B_sorted.shape)
        print(eig_vec_B_sorted[0,1,0,:,:],eigen_vec_B[0,1,0,:,:])
        condition_B = (np.abs(np.imag(lambda_mat_B[:,:,:,0]))<1e-10) & (np.abs(np.imag(lambda_mat_B[:,:,:,1]))<1e-10) & (np.abs(np.imag(lambda_mat_B[:,:,:,2]))<1e-10)  & (np.abs(lambda_mat_B_sorted[:,:,:,1])>C*np.abs(lambda_mat_B_sorted[:,:,:,0])) & (np.real(lambda_mat_B_sorted[:,:,:,1]*lambda_mat_B_sorted[:,:,:,2])<0)
        condition_u = (np.abs(np.imag(lambda_mat_u[:,:,:,0]))<1e-10) & (np.abs(np.imag(lambda_mat_u[:,:,:,1]))<1e-10) & (np.abs(np.imag(lambda_mat_u[:,:,:,2]))<1e-10)  & (np.abs(lambda_mat_u_sorted[:,:,:,1])>C*np.abs(lambda_mat_u_sorted[:,:,:,0])) & (np.real(lambda_mat_u_sorted[:,:,:,1]*lambda_mat_u_sorted[:,:,:,2])<0)
        condition_plane = np.abs(np.sum(n_vec_B*n_vec_u,axis=3)/np.linalg.norm(n_vec_B,axis=3)/np.linalg.norm(n_vec_u,axis=3))>0.9
        condition = np.where(condition_B & condition_u & condition_plane)
        i_lst, j_lst, k_lst = list(zip(condition))
        # lambda_mat_B_sorted[i_lst,j_lst,k_lst].shape
        # print(lambda_mat_B_sorted[condition[0]][0])
        # print(eigen_vec_B[condition[0]])
        # print(condition[0].shape)
        i_right = 0
        x_point_lst = []
        lambda_1_lst, lambda_2_lst = [], []
        eig_vec_b1_lst = []
        eig_vec_b2_lst = []
        delta_l = 3
        cos_lambda_1z_lst = []
        cos_lambda_2z_lst = []
        cos_lambda_max_lst = []
        cos_lambda_min_lst = []
        Bz_total = Bz
        
       
        # interp_Ti = RegularGridInterpolator((x, y, z), Ti)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = np.linspace(-5, 5, 100)  # a0方向坐标
        v = np.linspace(-5, 5, 100)  # b0方向坐标
        U, V = np.meshgrid(u, v)
        
        for i in range(len(condition[0])):
            # print(i)
            cos_lambda_1z = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:],[0,0,1])/np.linalg.norm(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:])
            cos_lambda_2z = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:],[0,0,1])/np.linalg.norm(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])
            cos_lambda_1z_lst.append(cos_lambda_1z)
            cos_lambda_2z_lst.append(cos_lambda_2z)
            cos_lambda_max_lst.append(max(np.abs(cos_lambda_1z),np.abs(cos_lambda_2z)))
            cos_lambda_min_lst.append(min(np.abs(cos_lambda_1z),np.abs(cos_lambda_2z)))

            # vec_1, vec_2 = np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:]), np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])
            # a0 = vec_1 / np.linalg.norm(vec_1)
            # b_perp = vec_2 - np.dot(vec_2, a0)*a0  # 施密特正交化
            # b0 = b_perp / np.linalg.norm(b_perp)
            # n = np.cross(vec_1, vec_2)
            # center = np.array([-32,-32,-32])+0.25*np.array([i_lst[0][i],j_lst[0][i],k_lst[0][i]])
            # R = (center + U[..., None]*a0 + V[..., None]*b0+32)%64-32 
            # Bx_proj = interp_Bx(R)
            # By_proj = interp_By(R)
            # Bz_proj = interp_Bz(R)
            # B = np.stack([Bx_proj, By_proj, Bz_proj], axis=-1)
            # B_plane = B - np.dot(B, n)[..., None]*n / np.linalg.norm(n)**2
            # B_u = np.dot(B_plane, a0)
            # B_v = np.dot(B_plane, b0)
            # u_near = np.linspace(-2, 2, 50)
            # v_naer = np.linspace(-2, 2, 50)
            # # print(interp_Bx(((center + u_near[0]*a0)+32)%64-32).shape)
            # for j in range(len(u_near)):
            #     bx_temp_1, by_temp_1, bz_temp_1 = interp_Bx(((center + u_near[0]*a0)+32)%64-32), interp_By(((center + u_near[0]*a0)+32)%64-32),interp_Bz(((center + u_near[0]*a0)+32)%64-32)
            #     b1_temp = np.dot([bx_temp_1, by_temp_1, bz_temp_1],vec_2)
            #     bx_temp_2, by_temp_2, bz_temp_2 = interp_Bx(((center + u_near[0]*a0)+32)%64-32), interp_By(((center + u_near[0]*a0)+32)%64-32),interp_Bz(((center + u_near[0]*a0)+32)%64-32)
            #     b2_temp = np.dot([bx_temp_2, by_temp_2, bz_temp_2],vec_1)
            #     if j==0:
            #         b1_temp_0 = b1_temp
            #         b2_temp_0 = b2_temp
            #     if np.sign(b1_temp)*np.sign(b1_temp_0)<0 and np.sign(b2_temp)*np.sign(b2_temp_0)<0:
            #         i_right += 1
            #         eig_vec_b1_lst.append(vec_1)
            #         eig_vec_b2_lst.append(vec_2)


            
            a_1,b_1,c_1 = find_nearest_lattice((int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])),list(np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])),delta_l)
            a_2,b_2,c_2 = find_nearest_lattice((int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])),list(-np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])),delta_l)
            a_3,b_3,c_3 = find_nearest_lattice((int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])),list(np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:])),delta_l)
            a_4,b_4,c_4 = find_nearest_lattice((int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])),list(-np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:])),delta_l)
            a_1,b_1,c_1 = a_1%256,b_1%256,c_1%256
            a_2,b_2,c_2 = a_2%256,b_2%256,c_2%256
            a_3,b_3,c_3 = a_3%256,b_3%256,c_3%256
            a_4,b_4,c_4 = a_4%256,b_4%256,c_4%256
            B1 = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:],[Bx[a_1,b_1,c_1],By[a_1,b_1,c_1],Bz_total[a_1,b_1,c_1]])
            B2 = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:],[Bx[a_2,b_2,c_2],By[a_2,b_2,c_2],Bz_total[a_2,b_2,c_2]])
            B3 = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:],[Bx[a_3,b_3,c_3],By[a_3,b_3,c_3],Bz_total[a_3,b_3,c_3]])
            B4 = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:],[Bx[a_4,b_4,c_4],By[a_4,b_4,c_4],Bz_total[a_4,b_4,c_4]])
            # print(B1,B2,B3,B4)
            if (B1*B2<0)&(B3*B4<0):
                i_right += 1
                eig_vec_b1_lst.append(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:])
                eig_vec_b2_lst.append(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])
                x_point_lst.append([i_lst[0][i],j_lst[0][i],k_lst[0][i]])
                lambda_1_lst.append(np.real(lambda_mat_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2]))
                lambda_2_lst.append(np.real(lambda_mat_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1]))
                print(i_right)
            # if (i%100==0):
                print("origin",(int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])))
                print("new",(a_1,b_1,c_1),r"$\vec{B}\cdot\hat{l}$=",np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],0,:],[Bx[a_1,b_1,c_1],By[a_1,b_1,c_1],Bz[a_1,b_1,c_1]]))
    
    position_end = []
    
    i_plot = 263-1
    # a = np.array([1, 0, 0])  # 示例向量a
    # b = np.array([0, 1, 0])  # 示例向量b（此处为xy平面，可替换为任意不平行向量）
    a = np.real(eig_vec_b1_lst[i_plot])
    b = np.real(eig_vec_b2_lst[i_plot])
    a0 = a / np.linalg.norm(a)
    b_perp = b - np.dot(b, a0)*a0  # 施密特正交化
    b0 = b_perp / np.linalg.norm(b_perp)
    # a_prime = eigenvecs_sorted_norm[0,2,:]
    # b_prime = eigenvecs_sorted_norm[0,1,:]
    # print(a,b)
    # print(-32+0.25*np.array(x_point_lst[:][2]))
    # print(np.where((-32+0.25*np.array(x_point_lst)[:,2]>-5)&(-32+0.25*np.array(x_point_lst)[:,2]<5)))
    center = np.array([-32,-32,-32])+0.25*np.array(x_point_lst[i_plot])
    max_steps=2000
    ds = 0.002
    # line_0 = np.array(trace_fieldline_3D_single(center, 
    #                                                ds=ds, max_steps=max_steps))
    # line_0_minus = np.array(trace_fieldline_3D_single(center,ds=ds, max_steps=max_steps, direction=-1))
    # # print(len(line_0))
    # position_start=line_0_minus[::-200,:].tolist()+line_0[::200,:].tolist()
    
    # position_start = get_position_start(np.array([2.5,-3,center[2]]))
    position_start_2 = get_position_start(center-2*a0-2.5*b0)
    position_start_single = []
    x_point = np.linspace(1.5,3,16)
    y_point = np.linspace(-3,-2.5,3)
    start = True
    if start:
        # position_start = get_position_start(np.array([-10.9,10,0]), step=1,ds=0.02, max_steps=1500)
        position_start = trace_fieldline_3D_single(np.array([ -6.35096253,  11.62916545, -25.97957101]), ds=0.02, max_steps=3000,direction=1).tolist()
        vex_lst, vey_lst = [], []
                
        for pos_tmp in position_start:
            vex_tmp = interp_uex_single(np.array(pos_tmp))
            vey_tmp = interp_uey_single(np.array(pos_tmp))
            vex_lst.append(vex_tmp)
            vey_lst.append(vey_tmp)
        
        vex_grad = np.gradient(np.array(vex_lst).squeeze())
        vey_grad = np.gradient(np.array(vey_lst).squeeze())
        print("old line:",np.sqrt(vex_grad**2+vey_grad**2).max())
        np.save("./data/position_start.npy",np.array(position_start))
    else:
        position_start = np.load("./data/position_end_arr_1.npy").tolist()


    position_start_single.append(np.array(position_start[0]).squeeze())
    position_start_single.append(np.array([10,-13,center[2]]))

    # position_start_single.append(np.array(position_start[0]).squeeze())

    # position_start_single.append(np.array([2.5,-3,center[2]]))
    position_start_single.append(np.array(position_start[-1]).squeeze())
    # for x_point_tmp in x_point:
    #     for y_point_tmp in y_point:
    #         position_start = get_position_start(np.array([x_point_tmp,y_point_tmp,center[2]]))
    #         position_start_single.append(np.array(position_start[0]).squeeze())
    #         position_start_single.append(np.array([x_point_tmp,y_point_tmp,center[2]]))

    #         # position_start_single.append(np.array(position_start[0]).squeeze())

    #         # position_start_single.append(np.array([2.5,-3,center[2]]))
    #         position_start_single.append(np.array(position_start[-1]).squeeze())
    # position_start_single.append(trace_fieldline_3D_single(position_start_single[0],ds=0.002,max_steps=500,direction=1)[-1,:].squeeze())
    
    print(trace_fieldline_3D_single(position_start_single[0],ds=0.002,max_steps=500,direction=1)[-1,:].squeeze())
    # position_start_single.append(np.array([1.7,-3,center[2]]))
    # print(get_position_start(center=np.array(position_start_single[0])))
    # line_1_test = trace_fieldline_3D_single(np.array([0,0,0]),ds=0.002,max_steps=200,direction=1)
    # line_2_test = trace_fieldline_3D_single(np.array(line_1_test[-1,:]),ds=0.002,max_steps=200,direction=-1)
    # print(line_1_test[-1,:], line_2_test[-1,:])
    # print(position_start)
    # print(position_start_2)
    i_pos = 0
    use_new_tracing_code = False
    """
    OLD TRACING CODE START
    """
    x_grid = np.linspace(center[0]-10,center[0]+10,201)
    y_grid = np.linspace(center[1]-10,center[1]+10,200)
    z_plot= position_start_single[0][2]
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='xy')
    n_points = X_grid.size  # 100*100=10000
    # 展平x、y网格，z固定为z_plot
    points = np.column_stack([
        X_grid.ravel(),    # x坐标（展平为10000个元素）
        Y_grid.ravel(),    # y坐标（展平为10000个元素）
        np.full(n_points, z_plot)  # z坐标（全为z_plot）
    ])  # shape: (10000, 3)

    # 3. 批量插值（一次性计算所有点，避免循环）
    
    # for _x in x_grid:
    #     for _y in y_grid:
    #         Bx_plot = interp_Bx(np.array([_x,_y,z_plot]))
    #         By_plot = interp_By(np.array([_x,_y,z_plot]))
    Bx_total, By_total, Bz_total = Q_3d['bx'],Q_3d['by'],Q_3d['bz']
    ni_total = Q_3d['ni']
    uix_total, uiy_total, uiz_total = Q_3d['uix'],Q_3d['uiy'],Q_3d['uiz']
    Jx_total, Jy_total, Jz_total = calculate_curl(Bx_total, By_total, Bz_total, x, y, z)
    uex, uey, uez = uix_total-Jx_total/ni_total, uiy_total-Jy_total/ni_total, uiz_total-Jz_total/ni_total
    t_arr = np.linspace(t_idx_start, t_idx_end, (t_idx_end-t_idx_start)//t_idx_step)
    interp_uex = RegularGridInterpolator((x, y, z, t_arr), uex)
    interp_uey = RegularGridInterpolator((x, y, z, t_arr), uey)
    interp_uez = RegularGridInterpolator((x, y, z, t_arr), uez)
    x_arr = np.linspace(-11.5,-10.5,11)
    y_arr = np.linspace(9.5,10.5,11)
    

    if not use_new_tracing_code:
        max_dis_mat = np.zeros((len(x_arr), len(y_arr)), dtype=float)

        for i in range(6,7):
            for j in range(5,6):
                pos_start_lst_tmp = get_position_start(np.array([x_arr[i],y_arr[j],0]),ds=0.02, max_steps=1300,step=25)
                position_end = []
                position_e_mat = np.zeros((len(position_start),len(t_arr),3))

                for pos_start_tmp in position_start:
                    position_e_lst = []
                    # print(i_pos)
                    
                    for idx in range(t_idx_start, t_idx_end, t_idx_step):
                        t_idx = (idx-t_idx_start)//t_idx_step
                        #print(t_idx)
                        
                        # print(uex.max(),uey.max(),uez.max(),np.sqrt(uex**2+uey**2+uez**2).max())

                        
                        # from matplotlib.patches import Rectangle
                        # import matplotlib.gridspec as gridspec
                        # from matplotlib.image import imread
                    
                        
                        #print(uix[x_point_lst[i_plot][0],x_point_lst[i_plot][1],x_point_lst[i_plot][2]],uiy[x_point_lst[i_plot][0],x_point_lst[i_plot][1],x_point_lst[i_plot][2]],uiz[x_point_lst[i_plot][0],x_point_lst[i_plot][1], x_point_lst[i_plot][2]])

                        # print(center)
                        # print(np.dot([Bx[i_lst[0][i],j_lst[0][i],k_lst[0][i]],By[i_lst[0][i],j_lst[0][i],k_lst[0][i]],Bz_total[i_lst[0][i],j_lst[0][i],k_lst[0][i]]],a))
                        # a0 = a / np.linalg.norm(a)
                        # b_perp = b - np.dot(b, a0)*a0  # 施密特正交化
                        # b0 = b_perp / np.linalg.norm(b_perp)
                        # n = np.cross(a, b)
                        # # from scipy.interpolate import RegularGridInterpolator

                        # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                        # # 3. 构建目标平面（xy平面）的二维网格
                        # u = np.linspace(-6, 6, 101)  # a0方向坐标

                        # v = np.linspace(-5, 5, 100)  # b0方向坐标
                        # u_near = np.linspace(-1.5, 1.5, 20)  # a0方向坐标
                        # v_near = np.linspace(-1.5, 1.5, 20)  # b0方向坐标
                        # U, V = np.meshgrid(u, v)
                        # U_near, v_near = np.meshgrid(u_near, v_near)
                        # # 将二维网格点映射到三维空间（xy平面，z=0）
                        # R = (center + U[..., None]*a0 + V[..., None]*b0+32)%64-32  # R.shape = (100,100,3)
                        # R_near = (center + U_near[..., None]*a0 + v_near[..., None]*b0+32)%64-32  # R.shape = (100,100,3)
                        # interp_Bx = RegularGridInterpolator((x, y, z), Bx)
                        # interp_By = RegularGridInterpolator((x, y, z), By)
                        # interp_Bz = RegularGridInterpolator((x, y, z), Bz_total)
                        # interp_uix = RegularGridInterpolator((x, y, z), uix)
                        # interp_uiy = RegularGridInterpolator((x, y, z), uiy)
                        # interp_uiz = RegularGridInterpolator((x, y, z), uiz)
                #         # print(center,R)
                #         # 4. 插值获取平面上的三维磁场并投影
                #         # 构建三维插值器
                        
                #         # print(len(position_start))
                #         # print(position_start)
                #         # print("test:",interp_uex(np.array([0,0,0])))
                        
                            
                        if idx==t_idx_start:
                            position_e_lst.append(np.array(pos_start_tmp))

                            
                        
                            # target_point_lst.append(center)
                        else:
                            
                            # print(position_e_lst[-1])
                            uex_tmp = float(interp_uex([position_e_lst[-1][0],position_e_lst[-1][1],position_e_lst[-1][2], idx]))
                            uey_tmp = float(interp_uey([position_e_lst[-1][0],position_e_lst[-1][1],position_e_lst[-1][2], idx]))
                            uez_tmp = float(interp_uez([position_e_lst[-1][0],position_e_lst[-1][1],position_e_lst[-1][2], idx]))
                            # print(uex_tmp, uey_tmp)
                            # print(interp_uix([position_e_lst[-1][0],position_e_lst[-1][1],position_e_lst[-1][2]]),interp_uiy([position_e_lst[-1][0],position_e_lst[-1][1],position_e_lst[-1][2]]))
                            x_e = float(position_e_lst[-1][0]+uex_tmp*dt*t_idx_step)
                            y_e = float(position_e_lst[-1][1]+uey_tmp*dt*t_idx_step)
                            z_e = float(position_e_lst[-1][2]+uez_tmp*dt*t_idx_step)
                            position_e_lst.append(np.array([x_e, y_e, z_e]))
                            # if i_pos == 106 or i_pos==107 or i_pos==108:
                            #     print(f"epoch={idx}, position:({x_e:.3f},{y_e:.3f},{z_e:.3f}), velocity:({uex_tmp:.3f},{uey_tmp:.3f},{uez_tmp:.3f})")
                            if idx==t_idx_end-t_idx_step:
                                position_end.append((x_e,y_e,z_e))  
                        position_e_mat[i_pos,t_idx,:]=np.array(position_e_lst[-1])
                    i_pos+=1
                        # Bx_plot_flat = interp_Bx(points)  # 结果 shape: (10000,)
                        # By_plot_flat = interp_By(points)  # 结果 shape: (10000,)

                        # # 4. 重塑为网格形状（与X、Y对应，方便后续绘图）
                        # Bx_plot = Bx_plot_flat.reshape(X_grid.shape)  # (100, 100)
                        # By_plot = By_plot_flat.reshape(X_grid.shape)
                        # uex_plot_flat = interp_uex(points)  # 结果 shape: (10000,)
                        # uey_plot_flat = interp_uey(points)  # 结果 shape: (10000,)

                        # # 4. 重塑为网格形状（与X、Y对应，方便后续绘图）
                        # uex_plot = uex_plot_flat.reshape(X_grid.shape)  # (100, 100)
                        # uey_plot = uey_plot_flat.reshape(X_grid.shape)
                        # """
                        # PLOT XY PLANE FIELD LINE AND VELOCITY
                        # """
                        # fig, axes = plt.subplots(1,2,figsize=(20, 10))
                        # ax = axes[0]
                        # idx_x, idx_y, idx_z = x_point_lst[i_plot][0], x_point_lst[i_plot][1], x_point_lst[i_plot][2]
                        
                        # # Bx_plot = interp_Bx(np.array([x_grid,y_grid,z_plot]))
                        # # print(X_grid.shape, Bx_plot.shape)
                        # ax.streamplot(X_grid,Y_grid,Bx_plot, By_plot, color='black', density=1, broken_streamlines=False)
                        # pclr=ax.pcolormesh(X_grid, Y_grid, uex_plot, cmap='jet')
                        # cbar= plt.colorbar(pclr,ax=ax)
                        # ax.scatter(position_e_lst[-1][0],position_e_lst[-1][1], c='blue', s=70)
                        # ax.set_xlabel("x[di]", fontsize=20)
                        # ax.set_ylabel("y[di]", fontsize=20)

                        # # ax.streamplot(x[x_plot_range],y[y_plot_range],Bx[x_plot_range,y_plot_range,x_point_lst[i_plot][2]].T, By[:,:,x_point_lst[i_plot][2]].T, color='black', density=1.8)
                        # # ax.set_xlim([center[0]-5,center[0]+5])
                        # # ax.set_ylim([center[1]-5,center[1]+5])
                        # ax = axes[1]
                        # ax.streamplot(X_grid,Y_grid,uex_plot, uey_plot, color='black', density=2.5)
                        # pclr=ax.pcolormesh(X_grid, Y_grid, uey_plot, cmap='jet')
                        # cbar= plt.colorbar(pclr,ax=ax)
                        # ax.scatter(position_e_lst[-1][0],position_e_lst[-1][1], c='blue', s=70)
                        # ax.set_xlabel("x[di]", fontsize=20)
                        # ax.set_ylabel("y[di]", fontsize=20)
                        # plt.suptitle(f"time: {idx}, z0: {position_start_single[0][2]:.2f},(x,y,z): ({position_e_lst[-1][0]:.3f},{position_e_lst[-1][1]:.3f},{position_e_lst[-1][2]:.3f})", fontsize=20)
                        # # ax.set_xlim([center[0]-5,center[0]+5])
                        # # ax.set_ylim([center[1]-5,center[1]+5])
                        # if not os.path.exists(f"./img/img_x_{position_start_single[0][0]:.2f}_y_{position_start_single[0][1]:.2f}_z_{position_start_single[0][2]:.2f}/"):
                        #     os.mkdir(f"./img/img_x_{position_start_single[0][0]:.2f}_y_{position_start_single[0][1]:.2f}_z_{position_start_single[0][2]:.2f}/")
                        # plt.savefig(f"./img/img_x_{position_start_single[0][0]:.2f}_y_{position_start_single[0][1]:.2f}_z_{position_start_single[0][2]:.2f}/fig_"+str(idx)+'.png')
                        # plt.close()
                        # # interp_Ti = RegularGridInterpolator((x, y, z), Ti)
                        # # interp_J_dot_e_prime = RegularGridInterpolator((x, y, z), J_dot_e_prime)
                        # # interp_J_dot_e_prime_parallel = RegularGridInterpolator((x, y, z), J_dot_e_prime_parallel)
                        # # interp_J_dot_e = RegularGridInterpolator((x, y, z), J_dot_e)
                        # # 插值得到平面上的B向量
                        # Bx_proj = interp_Bx(R)
                        # By_proj = interp_By(R)
                        # Bz_proj = interp_Bz(R)
                        # uix_proj = interp_uix(R)
                        # uiy_proj = interp_uiy(R)
                        # uiz_proj = interp_uiz(R)
                        # # Ti_proj = interp_Ti(R)
                        # # J_dot_e_prime_proj = interp_J_dot_e_prime(R)
                        # # J_dot_e_prime_parallel_proj = interp_J_dot_e_prime_parallel(R)
                        # # J_dot_e_proj = interp_J_dot_e(R)
                        # uix_proj_near = interp_uix(R_near)
                        # uiy_proj_near = interp_uiy(R_near)
                        # uiz_proj_near = interp_uiz(R_near)
                        
                            
                        
                        # B = np.stack([Bx_proj, By_proj, Bz_proj], axis=-1)
                        # ui = np.stack([uix_proj, uiy_proj, uiz_proj], axis=-1)
                        # ui_near = np.stack([uix_proj_near, uiy_proj_near, uiz_proj_near], axis=-1)
                        # # 剔除法向分量（n=(0,0,1)，此处即剔除Bz）
                        # B_plane = B - np.dot(B, n)[..., None]*n / np.linalg.norm(n)**2
                        # ui_plane = ui - np.dot(ui, n)[..., None]*n / np.linalg.norm(n)**2
                        # ui_plane_near = ui_near - np.dot(ui_near, n)[..., None]*n / np.linalg.norm(n)**2
                        # # 转化为局部二维分量（B_u = B·a0，B_v = B·b0）
                        # B_u = np.dot(B_plane, a0)
                        # B_v = np.dot(B_plane, b0)
                        # u_u = np.dot(ui_plane, a0)
                        # u_v = np.dot(ui_plane, b0)
                        # u_u_near = np.dot(ui_plane_near, a0)
                        # u_v_near = np.dot(ui_plane_near, b0)
                        # u_u_prime = u_u-u_u_near.mean()
                        # u_v_prime = u_v-u_v_near.mean()

                        # rect = Rectangle(
                        #     (-2, -1.5),  # 左下角坐标
                        #     width=3,  # 沿x轴长度
                        #     height=3,  # 沿y轴长度
                        #     edgecolor='red',  # 边框颜色
                        #     facecolor='none',  # 填充颜色（none为空心）
                        #     linewidth=2,  # 边框线宽
                        #     linestyle='-'  # 边框样式（虚线）
                        # )
                        # # 5. 绘制磁力线投影
                        # fig = plt.figure(figsize=(18, 16))
                        # gs = gridspec.GridSpec(1, 2, figure=fig, height_ratios=[1], hspace=0.45)
                        # # ax = fig.add_subplot(gs[0, :])
                        # # ax.imshow(imread("screenshot_90.png"),aspect='auto')
                        # # ax.set_xticks([])
                        # # ax.set_yticks([])
                        # ax = fig.add_subplot(gs[0])
                        # if idx==t_idx_start:
                        #     # vector = np.array(position_start)-center[np.newaxis,...]
                        #     # vector_2 = vector-(vector@n)@n/np.linalg.norm(n)**2
                        #     for pos_tmp in position_start:
                        #         vector = (np.array(pos_tmp)-center)-np.dot(np.array(pos_tmp)-center,n)*n/np.linalg.norm(n)**2
                        #         _a,_b = decompose_vector(vector,a0,b0)
                        #         print(_a,_b)
                        #         ax.scatter(_a,_b,c='b')
                        # elif idx==t_idx_end-1:
                        #     for pos_tmp in position_end:
                        #         vector = (np.array(pos_tmp)-center)-np.dot(np.array(pos_tmp)-center,n)*n/np.linalg.norm(n)**2
                        #         _a,_b = decompose_vector(vector,a0,b0)
                        #         ax.scatter(_a,_b,c='g')



                        # #     # for _i in range(1,len(line_0)-1,20):
                        # #     #     vector = (line_0[_i, :]-center)-np.dot(line_0[_i,:]-center,n)*n/np.linalg.norm(n)**2
                        # #     #     _a,_b = decompose_vector(vector,a0,b0)
                        # #     #     ax.scatter(_a,_b,c='b')



                        # ax.streamplot(U, V, B_u, B_v, density=1, color='k', linewidth=0.8, broken_streamlines=False)
                        # # ax.streamplot(x,y,Bx)
                        # # ax.scatter((np.array(target_point_lst[-1]).squeeze()-center)[0],)
                        # # print(u_u_prime.shape,u_v_prime.shape)
                        # # pclr = ax.pcolormesh(U,V,J_dot_e_prime_parallel_proj,cmap='bwr',shading='auto', vmin=-0.02, vmax=0.02)
                        # # cbar=plt.colorbar(pclr,ax=ax)
                        # # cbar.set_label(r'$(J\cdot E^\prime)_{\parallel}$', fontsize=20)
                        # ax.arrow(-1.2,0.8,10*(u_u_prime[58,38]), 10*u_v_prime[58,38], head_width=0.2,color='b')
                        # # ax.arrow(0.3,0,10*(u_u_prime[50,53]), 10*u_v_prime[50,53], head_width=0.2,color='b')
                        # ax.arrow(0.2,1,10*(u_u_prime[60,52]), 10*u_v_prime[60,52], head_width=0.2, color='r')
                        # ax.arrow(0.5,-0.8,10*(u_u_prime[42,55]), 10*u_v_prime[42,55], head_width=0.2, color='b')
                        # ax.arrow(-1,-0.8,10*(u_u_prime[42,40]), 10*u_v_prime[40,42], head_width=0.2, color='r')
                        # # ax.arrow(1.5,-0.3,10*(u_u_prime[50,65]), 10*u_v_prime[50,65], head_width=0.2, color='r')
                        # # # ax.streamplot(U, V, u_u-u_u_near.mean(), u_v-u_v_near.mean(), density=2, color='b', linewidth=0.8)
                        # # # plt.streamplot(U, V, u_u-u_u_near.mean(), u_v-u_v_near.mean(), density=2, color='b', linewidth=0.8)
                        # ax.add_patch(rect)
                        # ax.set_xlabel(f'$e_1$')
                        # ax.set_ylabel(f'$e_2$')
                        # ax.set_title(f'magnetic field line projection\n on the principal eigenvector plane', fontsize=25)
                        # ax.axis('equal')
                        # ax = fig.add_subplot(gs[1])
                        # rect = Rectangle(
                        #     (-2, -1.5),  # 左下角坐标
                        #     width=3,  # 沿x轴长度
                        #     height=3,  # 沿y轴长度
                        #     edgecolor='red',  # 边框颜色
                        #     facecolor='none',  # 填充颜色（none为空心）
                        #     linewidth=2,  # 边框线宽
                        #     linestyle='-'  # 边框样式（虚线）
                        # )
                        # # ax.streamplot(U, V, B_u, B_v, density=2, color='k', linewidth=0.8)
                        # ax.streamplot(U, V, u_u-u_u_near.mean(), u_v-u_v_near.mean(), density=3, color='k', linewidth=0.8)
                        
                        # # pclr = ax.pcolormesh(U,V,J_dot_e_prime_parallel_proj,cmap='bwr',shading='auto', vmin=-0.02, vmax=0.02)
                        # # cbar=plt.colorbar(pclr,ax=ax)
                        # # cbar.set_label(r'$(J\cdot E^\prime)_{\parallel}$', fontsize=20)
                        # ax.arrow(-1.2,0.8,10*(u_u_prime[58,38]), 10*u_v_prime[58,38], head_width=0.2,color='b')
                        # # ax.arrow(0.3,0,10*(u_u_prime[50,53]), 10*u_v_prime[50,53], head_width=0.2,color='b')
                        # ax.arrow(0.2,1,10*(u_u_prime[60,52]), 10*u_v_prime[60,52], head_width=0.2, color='r')
                        # ax.arrow(0.5,-0.8,10*(u_u_prime[42,55]), 10*u_v_prime[42,55], head_width=0.2, color='b')
                        # ax.arrow(-1,-0.8,10*(u_u_prime[42,40]), 10*u_v_prime[40,42], head_width=0.2, color='r')
                        # ax.add_patch(rect)
                        # ax.set_xlabel(f'$e_1$')
                        # ax.set_ylabel(f'$e_2$')
                        # ax.set_title(f'velocity field line projection\n on the principal eigenvector plane', fontsize=25)
                        # ax.axis('equal')
                        # plt.savefig(f'./figures/fig_{i_plot}_epoch_{idx}.png')
                        # plt.close(fig=fig)
                    # plt.figure(figsize=(10,7))
                    # plt.plot(np.array(position_e_lst)[:,0],np.array(position_e_lst)[:,1])
                    # plt.savefig("./figures/fig_e_traj.png")
                position_end_arr = np.array(position_end)
                position_end_arr = (position_end_arr+32)%64-32
                delta_dis_arr = np.sqrt((position_end_arr[1:,0]-position_end_arr[:-1,0])**2 + (position_end_arr[1:,1]-position_end_arr[:-1,1])**2 + (position_end_arr[1:,2]-position_end_arr[:-1,2])**2)
                print(i,j,'max distance= ',delta_dis_arr.max(),', index= ',delta_dis_arr.argmax())
                max_dis_mat[i,j] = delta_dis_arr.max()
        # plt.figure(figsize=(10,7))
        # pclr=plt.pcolormesh(x_arr,y_arr,max_dis_mat.T,cmap='coolwarm')
        # cbar = plt.colorbar(pclr)
        # plt.savefig("./figures/max_distance.png")
        # plt.close()
        np.save("./data/position_end_arr_1.npy", position_end_arr)
        np.save("./data/position_e_mat.npy", position_e_mat)






    # position_start_arr = np.array(position_start)


    #np.save("./data/position_end_arr_1.npy", position_end_arr)


"""
-------OLD TRACING CODE END----------
"""

"""
-------NEW TRACING CODE START----------
"""
if use_new_tracing_code:
    from scipy.interpolate import RegularGridInterpolator
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    def precompute_interpolators(t_idx_start, t_idx_end, Q_3d, x, y, z):
        """预先计算所有时间步的插值器，供所有任务共享"""
        interpolators = []
        for idx in range(t_idx_start, t_idx_end):
            #print(idx)
            t_idx = idx - t_idx_start  # 相对时间索引（Q_3d的第4维）
            # 计算磁场旋度J
            Bx = Q_3d['bx'][:, :, :, t_idx]
            By = Q_3d['by'][:, :, :, t_idx]
            Bz = Q_3d['bz'][:, :, :, t_idx]
            Jx, Jy, Jz = calculate_curl(Bx, By, Bz, x, y, z)  # 假设该函数已定义
            # 计算uex/uey/uez
            ni = Q_3d['ni'][:, :, :, t_idx]
            uix = Q_3d['uix'][:, :, :, t_idx]
            uiy = Q_3d['uiy'][:, :, :, t_idx]
            uiz = Q_3d['uiz'][:, :, :, t_idx]
            uex = uix - Jx / ni
            uey = uiy - Jy / ni
            uez = uiz - Jz / ni
            # 创建插值器并存储
            interp_uex = RegularGridInterpolator((x, y, z), uex)
            interp_uey = RegularGridInterpolator((x, y, z), uey)
            interp_uez = RegularGridInterpolator((x, y, z), uez)
            interpolators.append((interp_uex, interp_uey, interp_uez))
        return interpolators
    
    def process_single_position(pos_start_tmp, interpolators, t_idx_start, t_idx_end, dt):
        position_e_lst = [np.array(pos_start_tmp)]  # 初始位置（形状(3,)）
        for step in range(1, t_idx_end - t_idx_start):
            prev_pos = position_e_lst[-1]  # 形状为(3,)的一维数组
            
            # 关键修改：将一维坐标转为二维数组 (1, 3)
            prev_pos_2d = prev_pos.reshape(1, -1)  # 或 [prev_pos]（列表转数组后也是(1,3)）
            
            # 使用二维数组调用插值器
            interp_uex, interp_uey, interp_uez = interpolators[step]
            uex_tmp = interp_uex(prev_pos_2d)  # 输入形状(1,3)，匹配三维插值器
            uey_tmp = interp_uey(prev_pos_2d)
            uez_tmp = interp_uez(prev_pos_2d)
            
            # 后续计算不变（注意uex_tmp等是数组，需取标量值）
            new_pos = np.array([
                prev_pos[0] + uex_tmp[0] * dt,  # 取第一个元素（因为只有1个点）
                prev_pos[1] + uey_tmp[0] * dt,
                prev_pos[2] + uez_tmp[0] * dt
            ]).astype(float)
            position_e_lst.append(new_pos)
        return position_e_lst[-1]
    
    def parallel_main(position_start, t_idx_start, t_idx_end, Q_3d, x, y, z, dt):
        # 1. 预处理插值器（所有任务共享）
        interpolators = precompute_interpolators(t_idx_start, t_idx_end, Q_3d, x, y, z)
        
        # 2. 并行执行所有初始位置的计算
        position_end = []
        total_tasks = len(position_start)  # 总任务数（用于进度条）
        
        # 初始化进度条（desc为描述文本，total为总任务数）
        pbar = tqdm(total=total_tasks, desc="计算轨迹进度")
        
        with ProcessPoolExecutor(max_workers=None) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    process_single_position,
                    pos, interpolators, t_idx_start, t_idx_end, dt
                ): i for i, pos in enumerate(position_start)
            }
            # 按原顺序收集结果，并更新进度条
            results = [None] * total_tasks
            for future in as_completed(futures):
                idx = futures[future]  # 原始索引（保证结果顺序）
                results[idx] = future.result()  # 存储结果
                pbar.update(1)  # 每完成一个任务，进度条+1
        
        pbar.close()  # 关闭进度条
        position_end = results
        return position_end
    position_end = parallel_main(position_start_single, t_idx_start, t_idx_end, Q_3d, x, y, z, dt)
    # i_count = 0
    for m in position_end:
        if m%3==0:
            idx_pos_mid_end = np.argmin(trace_fieldline_3D_single(position_end[m],ds=0.002,num_steps=5000,direction=1)[:,2].squeeze()-float(position_end[m+1][2]))
            idx_pos_top_end = np.argmin(trace_fieldline_3D_single(position_end[m],ds=0.002,num_steps=5000,direction=1)[:,2].squeeze()-float(position_end[m+2][2]))
            pos_mid_end = trace_fieldline_3D_single(position_end[m],ds=0.002,num_steps=5000,direction=1)[idx_pos_mid_end,:]
            pos_top_end = trace_fieldline_3D_single(position_end[m],ds=0.002,num_steps=5000,direction=1)[idx_pos_top_end,:]
            print(pos_mid_end, position_end[m+1])
            print(pos_top_end, position_end[m+2])


            
    # position_end_2 = parallel_main(position_start_2, t_idx_start, t_idx_end, Q_3d, x, y, z, dt)
    """
    NEW TRACING CODE END
    """

                    # print(f"idx={idx-t_idx_start}",interp_uex(np.array(target_point_lst[-1])))
                    # print(f"idx={idx-t_idx_start}",np.array(target_point_lst[-1]).shape,np.array([uex_tmp,uey_tmp,uez_tmp]).squeeze().shape)
                    # target_point_lst.append(np.array(target_point_lst[-1])+np.array([uex_tmp,uey_tmp,uez_tmp]).squeeze()*dt)


        # for i, item in enumerate(target_point_lst):
        #     print(f"第{i}个元素: {item}, 长度: {len(item)}")
        # plt.figure(figsize=(10, 7))
        # plt.plot(np.array(position_e_lst)[:,0],label='x')
        # plt.plot(np.array(position_e_lst)[:,1],label='y')
        # plt.plot(np.array(position_e_lst)[:,2],label='z')
        # plt.savefig("./figures/trajectory.png")
    # vector = np.array(position_start)-center[np.newaxis,...]
                # vector_2 = vector-(vector@n)@n/np.linalg.norm(n)**2
    '''
    PLOT FIELDLINE TRACKING RESULTS IN 2D
    '''
    # plt.figure(figsize=(10, 7))
    # plt.streamplot(U, V, B_u, B_v, density=1, color='k', linewidth=0.8, broken_streamlines=False)
    # for pos_tmp in position_start:
    #     vector = (np.array(pos_tmp)-center)-np.dot(np.array(pos_tmp)-center,n)*n/np.linalg.norm(n)**2
    #     _a,_b = decompose_vector(vector,a0,b0)
    #     plt.scatter(_a,_b,c='b')
    # for pos_tmp in position_end:
    #     vector = (np.array(pos_tmp)-center)-np.dot(np.array(pos_tmp)-center,n)*n/np.linalg.norm(n)**2
    #     _a,_b = decompose_vector(vector,a0,b0)
    #     plt.scatter(_a,_b,c='g')
    # plt.savefig("./figures/trajectory_2d.png")
    '''
    PLOT FIELDLINE TRACKING RESULTS IN 3D
    '''
    # import numpy as np
    import pyvista as pv
    pv.OFF_SCREEN = True


    # ----------------------沿用你的变量（确保以下变量已定义）----------------------
    # x, y, z: 一维数组，通过 np.linspace(-32, 32, 256) 生成
    # Bx, By, Bz_total: 三维数组，形状为 (256, 256, 256)
    # a, b: 主特征向量
    # center: 平面中心点

    # 单位化主特征向量
    a0 = a / np.linalg.norm(a)
    b_perp = b - np.dot(b, a0) * a0
    b0 = b_perp / np.linalg.norm(b_perp)
    n = np.cross(a0, b0)  # 平面法向量


    # ----------------------关键修改：正确创建三维网格----------------------
    # 1. 用 meshgrid 生成三维坐标网格（与你的磁场数据维度匹配）
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # 形状均为 (256, 256, 256)

    # 2. 创建结构化网格：指定点坐标和网格维度
    grid = pv.StructuredGrid()
    # 将三维坐标展平为 (N, 3) 形状的点坐标（N=256×256×256）
    grid.points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    # 指定网格维度（必须与 x, y, z 的长度一致）
    grid.dimensions = (len(x), len(y), len(z))  # (256, 256, 256)


    # ----------------------正确添加磁场向量场----------------------
    # 展平磁场分量并合并为向量场（形状为 (N, 3)，N=256³）
    magnetic_field = np.stack([Bx.ravel(), By.ravel(), Bz_total.ravel()], axis=1)
    # magnetic_field = np.stack([uix.ravel(), uiy.ravel(), uiz.ravel()], axis=1)
    grid['magnetic_field'] = magnetic_field  # 此时点数量与向量场长度匹配


    # ----------------------生成三维磁力线----------------------
    # 种子点：在中心点附近生成（确保在网格范围内）
    seed_points = pv.PointSet(
        center + np.random.rand(50, 3) * 10 - 5  # 中心点周围±5范围
    )
    seed_points_center = pv.PointSet(
        [center-2*a0+2*b0]
    )

    # 生成磁力线
    streamlines = grid.streamlines_from_source(
                    seed_points,  # 使用包含卫星种子点的集合
                    vectors="magnetic_field",
                    max_time=700,  # 延长流线长度（确保穿过卫星）
                    integration_direction="both",  # 双向积分（从卫星向前后延伸）
                    max_steps=1000  # 增加步数，避免流线过早终止
                )
    streamline_center = grid.streamlines_from_source(
                    seed_points_center,  # 使用包含卫星种子点的集合
                    vectors="magnetic_field",
                    max_time=700,  # 延长流线长度（确保穿过卫星）
                    integration_direction="both",  # 双向积分（从卫星向前后延伸）
                    max_steps=1000  # 增加步数，避免流线过早终止
                )

    # ----------------------绘制主特征向量箭头----------------------
    arrow_length = 8.0
    points_start = np.array(position_start)
    point_cloud = pv.PolyData(points_start)
    # point_cloud["x_value"] = points_start[:, 0]  # 用X坐标着色

# 创建Glyph：每个点替换为立方体，大小由factor控制
    glyphs_start = point_cloud.glyph(
        geom=pv.Sphere(),  # 几何体（可选：Sphere(), Arrow()等）
        scale=False,     # 不按标量缩放（固定大小）
        factor=0.5       # 几何体大小（值越大，点越大）
    )

    points_end = np.array(position_end)
    point_cloud = pv.PolyData(points_end)
    # point_cloud["x_value"] = points_end[:, 0]  # 用X坐标着色

# 创建Glyph：每个点替换为立方体，大小由factor控制
    glyphs_end = point_cloud.glyph(
        geom=pv.Sphere(),  # 几何体（可选：Sphere(), Arrow()等）
        scale=False,     # 不按标量缩放（固定大小）
        factor=0.5       # 几何体大小（值越大，点越大）
    )
    points_start_2 = np.array(position_start_2)
    point_cloud = pv.PolyData(points_start_2)
    # point_cloud["x_value"] = points_start[:, 0]  # 用X坐标着色

# 创建Glyph：每个点替换为立方体，大小由factor控制
    glyphs_start_2 = point_cloud.glyph(
        geom=pv.Sphere(),  # 几何体（可选：Sphere(), Arrow()等）
        scale=False,     # 不按标量缩放（固定大小）
        factor=0.5       # 几何体大小（值越大，点越大）
    )

    points_end_2 = np.array(position_end_2)
    point_cloud = pv.PolyData(points_end_2)
    # point_cloud["x_value"] = points_end[:, 0]  # 用X坐标着色

# 创建Glyph：每个点替换为立方体，大小由factor控制
    glyphs_end_2 = point_cloud.glyph(
        geom=pv.Sphere(),  # 几何体（可选：Sphere(), Arrow()等）
        scale=False,     # 不按标量缩放（固定大小）
        factor=0.5       # 几何体大小（值越大，点越大）
    )
    # 向量a的箭头
    arrow_a = pv.Arrow(
        start=center,
        direction=a* arrow_length,
        # tip_radius=0.01,
        # tip_length=1.0,
        # shaft_radius=0.15,
        scale=10
    )

    # 向量b的箭头
    arrow_b = pv.Arrow(
        start=center,
        direction=b * arrow_length,
        # tip_radius=0.01,
        # tip_length=1.0,
        # shaft_radius=0.15,
        scale=10
    )

    # arrow_a_prime = pv.Arrow(
    #     start=center,
    #     direction=-a_prime* arrow_length,
    #     # tip_radius=0.01,
    #     # tip_length=1.0,
    #     # shaft_radius=0.15,
    #     scale=10
    # )

    # # 向量b的箭头
    # arrow_b_prime = pv.Arrow(
    #     start=center,
    #     direction=-b_prime * arrow_length,
    #     # tip_radius=0.01,
    #     # tip_length=1.0,
    #     # shaft_radius=0.15,
    #     scale=10
    # )
    # arrow_b_prime_2 = pv.Actor(
    #     mapper=pv.DataSetMapper(arrow_b_prime)  # 绑定箭头数据
    # )

    # 3. 通过Actor的Property设置虚线和线条粗细
    # SetLineStyle(2)：2=虚线（1=实线，3=点线，4=点划线）
    # vtk_actor = arrow_b_prime_2.GetVTKActor()  # 获取 VTK 原生 Actor
    # vtk_prop = vtk_actor.GetProperty()  # 获取 VTK 原生属性对象（支持 SetLineStyle）

    # # 4. 设置虚线和线条粗细（VTK 原生方法）
    # vtk_prop.SetLineStyle(2)
    # ----------------------绘制a和b张成的平面----------------------
    plane_size = 15.0
    plane = pv.Plane(
        center=center,
        direction= n,
        # direction_x=a0,
        # direction_y=b0,
        i_size=plane_size,
        j_size=plane_size,
        i_resolution=50,
        j_resolution=50
    )
    # plane['opacity'] = 0.3


    # ----------------------可视化----------------------
    p = pv.Plotter(window_size=[1200, 800])
    # p.add_mesh(streamlines, line_width=2, color='blue', label='Magnetic Field Lines')
    # p.add_mesh(streamline_center, line_width=4, color='red')
    p.add_mesh(plane, color='lightgreen', opacity=0.3, label='Principal eigenplane')
    p.add_mesh(arrow_a, color='red', label=r'Principal eigenvector e_1')
    p.add_mesh(arrow_b, color='green', label=r'Principal eigenvector e_2')
    p.add_mesh(glyphs_start, color='black')
    p.add_mesh(glyphs_end, color='red')
    p.add_mesh(glyphs_start_2, color='blue')
    p.add_mesh(glyphs_end_2, color='yellow')
    # p.add_mesh(arrow_b_prime, color=(1.0, 0.7, 0.7), label=r'Principal eigenvector(muti-satellites) e_2')
    # p.add_mesh(arrow_a_prime, color=(0.7, 1.0, 0.7), label=r'Principal eigenvector(muti-satellites) e_1')
    print(p.actors.keys())
    # arrow_actor=p.actors['PolyData(Addr=000001CD5D858CD0)']

    # 4. 通过VTK底层方法设置虚线（兼容旧版本的核心步骤）
    # 获取VTK原生的属性对象（无论PyVista版本多旧，VTK底层一定支持）
    # vtk_prop = arrow_actor.GetProperty()
    # vtk_prop.SetLineStyle(2)
    # p.add_points(center, color='yellow', point_size=30, label='Center')
    p.add_axes()
    p.add_title('3D Magnetic Field with Principal Vectors and Plane')
    legend = p.add_legend(loc='lower right',size=(0.3, 0.3))
    if hasattr(legend, 'GetLabelTextProperty'):
        label_prop = legend.GetLabelTextProperty()  # 全局条目文本属性
        label_prop.SetFontSize(26)
    p.camera_position = [center + [30, 30, 30], center, n]  # 调整视角
    p.show_grid()
    # p.show()
    # image_file = '3D_MagneticField.png'
    # image_3d = p.screenshot(image_file, transparent_background=False)
    p.export_html('pv.html')