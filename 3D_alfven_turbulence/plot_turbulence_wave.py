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
# qs = ["bx","by","bz","uix","uiy","uiz","ni"]
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
t_idx_start = 180
t_idx_end = 230
t_idx_step = 1

calculate_w_k = 1
use_cwt_2DT_CPU = 0
use_cwt_2DT_GPU = 1
re_calc = 0

W_thermal_arr = np.array([])

xv = np.linspace(0,Lx,nx)-Lx/2.0
yv = np.linspace(0,Ly,ny)-Ly/2.0
zv = np.linspace(0,Lz,nz)-Lz/2.0
slicenums = []



# variable = Q_3d['bx'][:,:,:,0]
# vari_rms = np.sqrt(np.mean(variable**2) - np.mean(variable)**2)
#print("vari_rms: ", vari_rms)

# import math
# def find_nearest_lattice(r1, direction, l):
#     """
#     在三维格点中，找到沿指定方向、距离起始格点r1为l的最近格点。
    
#     参数:
#         r1 (tuple/list): 起始格点坐标，需为3个整数，如(1, 2, 3)
#         direction (tuple/list): 方向向量，需为3个数字，如(1.0, 1.0, 0.0)
#         l (float/int): 距离，需为正数
    
#     返回:
#         tuple: 最近的格点坐标（3个整数）
    
#     异常:
#         ValueError: 输入参数不符合要求时抛出
#     """
#     # 验证起始格点r1的有效性（3个整数）
#     if not (isinstance(r1, (tuple, list)) and len(r1) == 3 and all(isinstance(x, int) for x in r1)):
#         raise ValueError("r1必须是包含3个整数的元组或列表")
    
#     # 验证方向向量的有效性（3个数字）
#     if not (isinstance(direction, (tuple, list)) and len(direction) == 3):
#         raise ValueError("direction必须是包含3个数字的元组或列表")
    
#     # 验证距离l的有效性（正数）
#     if not (isinstance(l, (int, float)) and l > 0):
#         raise ValueError("l必须是正数")
    
#     # 解析方向向量分量
#     vx, vy, vz = direction
    
#     # 计算方向向量的模长（避免零向量）
#     mod = math.sqrt(vx**2 + vy**2 + vz**2)
#     if mod < 1e-10:  # 考虑浮点数精度误差
#         raise ValueError("方向向量不能是零向量")
    
#     # 归一化方向向量（得到单位向量）
#     nx = vx / mod
#     ny = vy / mod
#     nz = vz / mod
    
#     # 计算沿方向移动距离l后的目标点坐标（非格点）
#     px = r1[0] + l * nx
#     py = r1[1] + l * ny
#     pz = r1[2] + l * nz
    
#     # 四舍五入得到最近的格点（各分量取最近整数）
#     nearest_lattice = (round(px), round(py), round(pz))
    
#     return nearest_lattice

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


if __name__ == '__main__':
    #epoch = 15
    x = np.linspace(-32, 32, 256)
    y = np.linspace(-32, 32, 256)
    z = np.linspace(-32, 32, 256)
    len_t_arr = t_idx_end-t_idx_start
    Bx_z_t = np.zeros((80,len_t_arr))
    e_para_z_t = np.zeros((80,len_t_arr))
    F_total_z_t = np.zeros((80,len_t_arr))
    H_reconnection_z_t = np.zeros((80,len_t_arr))
    Jz_z_t = np.zeros((80,len_t_arr))
    print(f"start epoch={t_idx_start}, end epoch={t_idx_end}")
    for epoch in range(len_t_arr):
        print(epoch)
        Q = {}
        Q_3d = {}
        for q in qs:
            slices = []
            for slice_ in range(epoch+t_idx_start, epoch+t_idx_start+1, t_idx_step):
                    #print(slice_)
                    tmp = loadSlice(dir,q,slice_,nx,ny,nz)
                    slices.append(tmp[:,:,:])
            Q_3d[q] = np.stack(slices, axis=-1)

        Bx, By, Bz = Q_3d['bx'][:,:,:,0],Q_3d['by'][:,:,:,0],Q_3d['bz'][:,:,:,0]
        uix, uiy, uiz = Q_3d['uix'][:,:,:,0],Q_3d['uiy'][:,:,:,0],Q_3d['uiz'][:,:,:,0]
        Jx, Jy, Jz = calculate_curl(Bx, By, Bz, x, y, z)
        ex, ey, ez  = Q_3d['ex'][:,:,:,0], Q_3d['ey'][:,:,:,0], Q_3d['ez'][:,:,:,0]
        ni = Q_3d['ni'][:,:,:,0]
        pi_xx, pi_yy, pi_zz = Q_3d['pi-xx'][:,:,:,0], Q_3d['pi-yy'][:,:,:,0], Q_3d['pi-zz'][:,:,:,0]
        uex, uey, uez = uix-Jx/ni, uiy-Jy/ni, uiz-Jz/ni
        Ti = (pi_xx+pi_yy+pi_zz)/3/ni
        e_para = (ex*Bx+ey*By+ez*Bz)/np.sqrt(Bx**2+By**2+Bz**2)
        ex_prime, ey_prime, ez_prime = ex+uiy*Bz-uiz*By, ey+uiz*Bx-uix*Bz, ez+uix*By-uiy*Bx
        e_prime_norm = np.sqrt(ex_prime**2+ey_prime**2+ez_prime**2)
        # print(3*e_prime_norm.std())
        H_reconnection = Jx*ex_prime+Jy*ey_prime+Jz*ez_prime
        H_total = Jx*ex+Jy*ey+Jz*ez
        J_para = (Jx*Bx+Jy*By+Jz*Bz)/np.sqrt(Bx**2+By**2+Bz**2)
        e_para = (ex*Bx+ey*By+ez*Bz)/np.sqrt(Bx**2+By**2+Bz**2)
        J_total = np.sqrt(Jx**2+Jy**2+Jz**2)
        ex_prime_e, ey_prime_e, ez_prime_e = ex+uey*Bz-uez*By, ey+uez*Bx-uex*Bz, ez+uex*By-uey*Bx
        Rx, Ry, Rz = calculate_curl(ex_prime, ey_prime, ez_prime, x, y, z)
        Fx, Fy, Fz = (By*Rx-Bz*Ry)/np.sqrt(Bx**2+By**2+Bz**2), (Bz*Rx-Bx*Rz)/np.sqrt(Bx**2+By**2+Bz**2), (Bx*Ry-By*Rx)/np.sqrt(Bx**2+By**2+Bz**2)
        F_total = np.sqrt(Fx**2+Fy**2+Fz**2)
        Bx_z_t[:,epoch] = np.mean(Bx[70-3:70+4,193-3:193+4,105:105+80],axis=(0,1))
        e_para_z_t[:,epoch] = np.mean(e_para[70-3:70+4,193-3:193+4,105:105+80],axis=(0,1))
        F_total_z_t[:,epoch] = np.mean(F_total[70-3:70+4,193-3:193+4,105:105+80],axis=(0,1))
        Jz_z_t[:,epoch] = np.mean(Jz[70-3:70+4,193-3:193+4,105:105+80],axis=(0,1))
        H_reconnection_z_t[:,epoch] = np.mean(H_reconnection[70-3:70+4,193-3:193+4,105:105+80],axis=(0,1))
    np.save(f"./data_tracing/Bx_z_t_{t_idx_start}_{t_idx_end}.npy",Bx_z_t)
    np.save(f"./data_tracing/e_para_z_t_{t_idx_start}_{t_idx_end}.npy",e_para_z_t)
    np.save(f"./data_tracing/F_total_z_t_{t_idx_start}_{t_idx_end}.npy",F_total_z_t)
    np.save(f"./data_tracing/H_reconnection_z_t_{t_idx_start}_{t_idx_end}.npy",H_reconnection_z_t)
    np.save(f"./data_tracing/J_total_z_t_{t_idx_start}_{t_idx_end}.npy",Jz_z_t)
    # t_arr = np.linspace(0,49,50)
    # z_arr = np.linspace(105,184,80)/4
    # fig, axes = plt.subplots(3,1, figsize=(10,12))
    # ax=axes[0]
    # pclr=ax.pcolormesh(t_arr,z_arr,Bx_z_t,cmap='jet')
    # ax.set_xlabel("t")
    # ax.set_ylabel("z")
    # plt.colorbar(pclr,ax=ax)
    # plt.title(r"$B_{x}$",fontsize=15)
    # ax=axes[1]
    # pclr=ax.pcolormesh(t_arr,z_arr,e_para_z_t,cmap='jet')
    # ax.set_xlabel("t")
    # ax.set_ylabel("z")
    # plt.title(r"$E_{//}$", fontsize=15)
    # plt.colorbar(pclr,ax=ax)
    # ax=axes[2]
    # pclr=ax.pcolormesh(t_arr,z_arr,F_total_z_t,cmap='jet')
    # ax.set_xlabel("t")
    # ax.set_ylabel("z")
    # plt.title(r"$|(\nabla\times E')_{\perp}|$", fontsize=15)
    # plt.colorbar(pclr,ax=ax)
    # plt.savefig("./figures/turbulence_waves.png")
    
