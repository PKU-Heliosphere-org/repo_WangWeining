%%
clc
clear
a1=cat(1,1.5*ones(30,61) ,ones(30,61) ,ones(30,61),ones(30,61),ones(30,61));
b1=reshape(a1,[5,30,61]);
sw_pressure_data=importdata("ACE_pressure data.txt",' ');
sw_pressure=sw_pressure_data.data;date_data=sw_pressure_data.textdata;
date=date_data(2:end,1);
%date_time=datetime(date,'InputFormat','dd-MMM-uuuu');
index=find(sw_pressure<10);index_2=find(sw_pressure>=10);
sw_pressure(index_2)=2*ones(length(index_2),1);
sw_pressure_data_new=sw_pressure(index);
delta_t=500;
newyear_index=zeros(21,1);
for i=1:20
newyear_index(i)=find(strcmp(date,strcat('01-01-',num2str(1999+i))),1,'first');
end
newyear_index(21)=length(date)+1;
for i=1:length(sw_pressure)-delta_t+1
    sw_pressure(i)=sum(sw_pressure(i:i+delta_t-1))/delta_t;
end
for j=length(sw_pressure)-delta_t+1:length(sw_pressure)
    sw_pressure(j)=sum(sw_pressure(j-delta_t+1:j))/delta_t;
end

plot(2000+(1:length(sw_pressure))/365/24,sw_pressure,LineWidth=1)
xticks(2000:2020)
xlabel('year',FontSize=13)
ylabel('SW dynamic pressure (nPa)',FontSize=13)
xlim([2000,2020])
for index=1:20
    subplot(5,4,index)
P=zeros(365,24);


for i=1:365
    for j=1:24
        P(i,j)=sw_pressure(newyear_index(index)+(i-1)*24+j-1);
    end
end
pcolor(P')
title(num2str(1999+index))
colormap('jet')
shading interp;
clim([1,3])
colorbar
end
% for i=1:20
%      subplot(5,4,i)
%      polarplot(linspace(0,2*pi,newyear_index(i+1)-newyear_index(i)),sw_pressure(newyear_index(i):newyear_index(i+1)-1))
%      rlim([0,3])
%      title(num2str(1999+i))
% end
%%
clc
phi_start=135*pi/180;phi_end=405*pi/180;
data_plot=sw_pressure(17545:10:113954);
len_data = length(data_plot);
phi = linspace(phi_start,phi_end,len_data);
phi_year=linspace(phi_start,phi_end,11);
x_data=zeros(len_data,1);y_data=zeros(len_data,1);
data_norm=2*data_plot;
data_min=min(data_plot);
x_data=(3+data_norm).*cos(phi)';y_data=(3+data_norm).*sin(phi)';
r=sqrt(x_data.^2+y_data.^2);
x=-5:0.01:5;
y=-sqrt(25-x.^2);
set(gcf,'color','white')
%whitebg('black')
plot(x,y,'color',[1,1,1],LineWidth=4)
hold on
x1=-5:0.01:5*cos(phi_start);x2=5*cos(phi_end):0.01:5;
y1=sqrt(25-x1.^2);y2=sqrt(25-x2.^2);
plot(x1,y1,'color',[1,1,1],LineWidth=4)
hold on
plot(x2,y2,'color',[1,1,1],LineWidth=4)
hold on
plot(x_data,y_data,'g','LineWidth',2)
a=annotation('textarrow',[0.66,0.64],[0.65,0.675]);
a.Color=[1,1,1];
r_txt=[4.5 4.5 4.5 4.5 4.5 4 3.8 3.5 3.2 3.5 3.8];
for i=1:11
    plot([4.5*cos(phi_year(i)-0.01*sign(i-6)),5*cos(phi_year(i)-0.01*sign(i-6))],[4.5*sin(phi_year(i)-0.01*sign(i-6)),5*sin(phi_year(i)-0.01*sign(i-6))],'color',[1,1,1],'LineWidth',2.5)
    %text(r_txt(i)*cos(phi_year(i)+0.01),r_txt(i)*sin(phi_year(i)+0.01),num2str(2001+i))
    hold on
end
xlim([-10,10])
ylim([-10,10])
xticks([])
yticks([])

%%
clc
clear
% A=imread("2022_12_01__00_07_15__PROBA2_SWAP_SWAP_174.jp2");
% A2=zeros(1024,1024,3);
% A2(:,:,1)=A;
% imshow(A2)
%%
clc
clear
%close all
%ax1 = axesm('MapProjection','robinson','Frame','on','Grid','on');
year_num=14;
mp=1.6726e-27;
au=1.5e11;
num_v=2000;
num_r=600;
step_v=1;
step_r=1;
rho_neu=load("rho_neu.mat").data;
rho_neu_new=6*load("rho_neu_new.mat").data;
%rho_neu_time=0.18e6*ones(600,61,31,year_num);%zeros(600,61,31,year_num);
rho_neu_time(:,:,:,8:9)=cat(4,rho_neu,rho_neu_new);
n_pui_new=load("n_pui_new.mat").data;
n_pui=load("n_pui.mat").data;
t_pui=load("t_pui.mat").data;
t_pui_new=load("t_pui_new.mat").data;
j_ENA_data_time=zeros(year_num,5,30,61);
n_pui_time=zeros(600,61,31,year_num);
%plot(90:200,n_pui(90:200,30,15))
t_pui_time=zeros(600,61,31,year_num);
n_pui_time(:,:,:,8:9)=cat(4,n_pui,n_pui_new);
t_pui_time(:,:,:,8:9)=cat(4,t_pui,t_pui_new);
j_ENA_data_test=importdata("D://Research/Data/ENA inversion/RibbonSeparation_14Years/ribbonmedian/hvset_tabular_ram_cg_gdf_2009/hv60.hide-trp-mono_80-0.71-flux.txt",' ');
size(j_ENA_data_test)
j_ENA_data_time(:,:,:,1:60)=load("ram_cg.mat").data;
j_ENA_data_time(:,:,:,61)=j_ENA_data_time(:,:,:,1);
save("j_ENA_data_raw.mat","j_ENA_data_time")
j_ENA_data_time_smooth=zeros(size(j_ENA_data_time));
for i=1:year_num
    for energy=1:5
        j_ENA_data_time_smooth(i,energy,:,:)=smooth(smooth(reshape(j_ENA_data_time(i,energy,:,:),[30,61])));
    end
end
%size(j_ENA_data_2022)
j_ENA_data=load("ram_ena.mat").data;
s_ts_data=load("s_ts.mat").data;
v_sw=load("v_sw.mat").data;

v_sw_new=load("v_sw_new.mat").data;
v_sw_time=zeros(600,61,31,year_num);
v_sw_time(:,:,:,8:9)=cat(4,v_sw,v_sw_new);
lat=load("lat.mat").data;
lon=load("lon.mat").data;
theta=load("theta.mat").data+0.01;
vx=load('vx.mat').data/1e3;vx_new=load('vx_new.mat').data/1e3;
vy=load('vy.mat').data/1e3;vy_new=load('vy_new.mat').data/1e3;
vz=load('vz.mat').data/1e3;vz_new=load('vz_new.mat').data/1e3;
vx_time=zeros(600,61,31,year_num);vy_time=zeros(600,61,31,year_num);vz_time=zeros(600,61,31,year_num);
vx_time(:,:,:,8:9)=cat(4,vx,vx_new);vy_time(:,:,:,8:9)=cat(4,vy,vy_new);vz_time(:,:,:,8:9)=cat(4,vz,vz_new);
hp=importdata("D://Research/data/hp_pos(2).txt");
s_ts_newdata = load('ts_new.mat').data;
s_ts=zeros(61,31);
s_ts(1:60,:)=s_ts_data(1:60,:);
s_ts(61,:)=s_ts_data(1,:);
hp_new=load("hp_new.mat").data;
s_ts_time=zeros(61,31,year_num);
s_ts_time(:,:,8:9)=cat(3,s_ts,s_ts_newdata(1:61,:));
hp_time(:,:,8:9)=cat(3,hp,hp_new);
T_energies=importdata("D://Research/Data/ENA inversion/T_energies.txt");
T_energies=T_energies(:,2:6);
T_tripes=importdata("D://Research//Data/ENA inversion/T_triples.txt");
T_tripes=T_tripes(:,2:6);
%plot(1:year_num,j_ENA_data_time_smooth(1:year_num,5,15,40))
%ylim([0,15])
a=zeros(599,1);
for i=1:1:599
    a(i)=n_pui(i+1,30,15)/n_pui(i,30,15);
end
%%
pcolor(0:6:360,-90:6:90,reshape(n_pui(60,:,:),[61,31])')
shading interp
colormap("jet")
colorbar
%%
clc
plot(1:100,n_pui(1:100,33,15)/1e6,'LineWidth',1)
xlabel('r(au)','FontSize',15)
ylabel('n_{PUI}(cm^{-3})','FontSize',15)
%%
plot(1:600,v_sw(:,30,15),'LineWidth',1)
% loglog(50:250,n_pui(50:250,1,1))
% disp(hp(1,1))
%%
energy_bin=[0.71 1.11 1.74 2.73 4.29];
energy_bin_new=[8.38 18 28.98 43.87];

v_bin=sqrt(2*1.6e-16.*energy_bin/mp)/1e3;
v_bin_new=sqrt(2*1.6e-16.*energy_bin_new/mp)/1e3;

v_bin_multi=zeros(45,1);
%v_bin_multi=zeros(36,1);
j_ENA_smooth=zeros(5,30,60);
n_pui_test=n_pui(:,30,15);


%{
j2=45;k=15;
 j_ENA_multidata=pixel_around(reshape(j_ENA_data_time_smooth(years,:,:,:),[5,30,61]),j2,k);
       %j_normal=@(kappa,v)v.^2.*sigma(0.5*mp*v.^2/1.6e-16).*f_pui_plasma(v+50,100,1e7,kappa,beta)./f_pui_plasma(v_bin(5)+50,100,1e7,kappa,beta)/v_bin(5)^2/sigma(0.5*mp*v_bin(5)^2/1.6e-16);
       %residual_normal=@(p,x)(j_ena_calf(rho_neu_time(:,j,k,year),n_pui_time(:,j,k,year),p(3)*t_pui_time(s_ts_time(j,k,year)+10,j,k,year),s_ts_time(j,k,year),p(1),v_bin_multi,v_sw_time(:,j,k,year),p(2),alpha,beta)/j_ena_calf(rho_neu_time(:,j,k,year),n_pui_time(:,j,k,year),p(3)*t_pui_time(s_ts_time(j,k,year)+10,j,k,year),s_ts_time(j,k,year),p(1),v_bin(5),v_sw_time(:,j,k,year),p(2),alpha,beta)-j_ENA_multidata/j_ENA_data_time_smooth(year,5,k,j2))./j_ENA_multidata.^0.5;
       %disp(j_ENA_data_time_smooth(year,:,k,j2))
       residual_normal=@(p,x)(j_ENA_correct(3*rho_neu_time(:,j,k,years),n_pui_time(:,j,k,years),p(3)*t_pui_time(:,j,k,years),s_ts_time(j,k,years),p(1),v_sw_time(:,j,k,years),p(2),p(5),alpha,T_energies(:,1:5),T_tripes(:,1:5),p(4))-j_ENA_data_time_smooth(years,1:5,k,j2))./j_ENA_data_time_smooth(years,1:5,k,j2).^0.25;
       %residual_normal=@(p,x)(j_ena_calf(rho_neu_time(:,j,k,year),n_pui_time(:,j,k,year),p(3)*t_pui_time(s_ts_time(j,k,year)+10,j,k,year),s_ts_time(j,k,year),p(1),v_bin_multi,v_sw_time(:,j,k,year),p(2),alpha)-j_ENA_multidata)./j_ENA_multidata;
       p_normal_min=[max(s_ts_time(j,k,years)+30,(hp_time(mod(j+1,60)+1,k,years)+hp_time(j,k,years)+hp_time(mod(j-1,60)+1,k,years))/3-50) 1.52 0.5 0.01 1.52];p_normal_max=[600 1000 5 0.8 1000];p_normal_0=[hp_time(j,k,years) 1.8 1.5 0.4 1.8];
       %kappa_0=2;kappa_min=1.5;kappa_max=10;
       try
       p_normal =lsqnonlin(residual_normal,p_normal_0,p_normal_min,p_normal_max);
       catch
       p_normal=p_normal_0;
       end
       j_ENA_fun=@(p,x)j_ena_calf(rho_neu_time(:,j,k,years),n_pui_time(:,j,k,years),p(3)*t_pui_time(:,j,k,years),s_ts_time(j,k,years),p(1),x,v_sw_time(:,j,k,years),p(2),p(5),alpha,p(4));
       index=1;q=p_normal(1);
       
       while j_ENA_fun([p_normal(1) p_normal(2) p_normal(3) p_normal(4) p_normal(5)] ,v_bin(1))-j_ENA_fun([p_normal(1)-index p_normal(2) p_normal(3) p_normal(4) p_normal(5)],v_bin(1))<3
           
           q=p_normal(1)-index;
           index=index+1;
       end
       
       p_normal=[q p_normal(2) p_normal(3) p_normal(4) p_normal(5)];  
       disp(p_normal)
       disp(j_ENA_data_time_smooth(8,:,k,j2))
       disp(j_ENA_correct(3*rho_neu_time(:,j,k,years),n_pui_time(:,j,k,years),p_normal(3)*t_pui_time(:,j,k,years),s_ts_time(j,k,years),p_normal(1),v_sw_time(:,j,k,years),p_normal(2),p_normal(5),alpha,T_energies(:,1:5),T_tripes(:,1:5),p_normal(4)))
%}
%%


for i=1:9
    v_bin_multi(5*i-4:5*i)=v_bin';
end
v_bin_add=[v_bin_multi;v_bin_multi];
v=1:step_v:num_v;
energy=0.5.*v.^2*1e6*mp/1.6/1e-16;
f_ENA=@(p,E)j_ENA_cal(E,p(1),p(2));
j_ENA=zeros(num_v/step_v,30,60);
j_ENA_fit=zeros(5,30,60);

phi=6*pi/180:6*pi/180:2*pi;

vsw=v_sw(floor(s_ts(1,1)):floor(s_ts(1,1))+200,1,1);
adiabatic_index=zeros(600,1);

%{
for i=1:30
    rho_neu_test(i)=rho_neu(s_ts(i),30,i);
end
plot(0:6:174,rho_neu_test/1e6)
%}
%plot(31:600,rho_neu(31:600,30,15)/1e6)
%hold on
%title('neutral hydrogen number density at the TS(cm^{-3})')
%}

clc
j_test=1;k_test=15;

data_test=floor(s_ts(j_test,k_test))-50:300;

e=n_pui(1:3,1,1);
%n_pui_fun_test=@(p,x)n_pui_cal(v_sw(floor(s_ts(1,1)+x),1,1),vdf_pui(j_ENA(:,1,1),rho_neu(:,1,1),s_ts(1,1),s_ts(1,1)+p(1))./au./(s_ts(1,1)+x),p(2));
s_hp=zeros(60,30);







num_strange=0;
v_plot=10:2000;
energy_plot=0.5*mp*(v_plot.*1e3).^2/1.6*1e16;
beta=1.5;
k_npui=[linspace(1,3.5,30) linspace(3.5,2,30)];
s_hp_time=zeros(year_num,61,30);
s_hp_old_time=zeros(year_num,61,30);
s_hp_original=zeros(year_num,61,31);
k_time=zeros(year_num,30,61);
l_time=zeros(year_num,30,61);
j_ENA_fit_time_single=zeros(year_num,5,30,61);j_ENA_fit_time_vas=zeros(year_num,5,30,61);
j_ENA_fit_time_double_15_40=zeros(year_num,5,30,61);j_ENA_fit_time_double_15_60=zeros(year_num,5,30,61);j_ENA_fit_time_double_para_40=zeros(year_num,5,30,61);
kappa_time=zeros(year_num,30,61);
beta_time=zeros(year_num,30,61);
%kappa=1.8;

year_start=1;
year_end=14;
year_num=year_end-year_start+1;
k_time=load("k_time.mat").k_time;
%kappa_time=load("kappa_time.mat").kappa_time;
j_ena_plot=zeros(600-s_ts(j_test,k_test)+1,1);
%disp(size(para_plot))
%para_plot(:,2)=2*ones(600-s_ts(30,15)+2,1);para_plot(:,1)=s_ts(30,15)-1:600;
%disp(para_plot)
alpha=abs(vx(:,j_test,k_test)*cos(lon(j_test))*sin(90-lat(k_test))+vy(:,j_test,k_test)*sin(lon(j_test))*sin(90-lat(k_test))+vz(:,j_test,k_test)*cos(90-lat(k_test)))./v_sw(:,j_test,k_test);
% for i=s_ts(j_test,k_test):600
% j_ena_plot(i-s_ts(j_test,k_test)+1)=5*j_ena_calf(rho_neu(:,j_test,k_test),n_pui(:,j_test,k_test),t_pui(:,j_test,k_test),s_ts(j_test,k_test),v_bin(5),v_sw(:,j_test,k_test),[i,2],alpha,@f_pui_plasma_single_kappa,1,1);
% end
% %disp(alpha)
% %plot(s_ts(j_test,k_test):600,j_ena_plot,LineWidth=1.5)
% %hold on
% % xlabel('radial distance(AU)',FontSize=15)
% % ylabel('ENA flux(cm^2 sr s KeV)^{-1}',FontSize=15)
% i=0;
% while j_ena_plot(end)-j_ena_plot(end-i)<1/j_ENA_data(5,j_test,k_test)
%     i=i+1;
% end
% d1=scatter(600-i,j_ena_plot(end-i),'filled','r');
% d2=scatter(hp(j_test,k_test),5*j_ena_calf(rho_neu(:,j_test,k_test),n_pui(:,j_test,k_test),t_pui(:,j_test,k_test),s_ts(j_test,k_test),v_bin(5),v_sw(:,j_test,k_test),[hp(j_test,k_test),2],alpha,@f_pui_plasma_single_kappa,1,1),'filled','b');
% hold on
%text(600-i+10,j_ena_plot(end-i)-10,'1/j_{ENA}')
% plot([600-i,600-i],[0,j_ena_plot(end-i)],'k--')
% 
% plot([0,600-i],[j_ena_plot(end-i),j_ena_plot(end-i)],'k--')
% legend([d1 d2],{'fitted','simulation'})
%%
PUI_para=[3.0293,5.1736,2];
s_hp=223.2552;
loglog(1:0.7:1000,f_pui_vas(1:0.7:1000,100,0.1,432,PUI_para(1),PUI_para(2)))
%j_ENA_fun_vas()
ylim([1e-18,1e-15])
%%
 % ENA MODEL PLOT
 clc
 j_plot=1;k_plot=1;
 j_ENA_fun_vas=@(p,x)p(5)*j_ena_calf(rho_neu(:,j_plot,k_plot),n_pui(:,j_plot,k_plot),t_pui(:,j_plot,k_plot),s_ts(j_plot,k_plot),x,v_sw(:,j_plot,k_plot),[p(1:3),theta(j_plot,k_plot),p(4)],alpha,@f_pui_vas,0.15,0.40);
 v_b_arr=[400 450 500 600];
 figure
 for i=1:length(v_b_arr)
 para_vas=[200 1.5 5 v_b_arr(i) 1];       
 
       j_ena_plot= j_ENA_fun_vas(para_vas,100:1000);
       l_vas=loglog(100:1000,j_ena_plot,'LineWidth',1.5);
       hold on
 end
       % hold on
       % l_single=loglog(100:1000,j_ENA_fun_single(para_single,100:1000),'LineWidth',1.5);
       % hold on
       % s_raw=scatter(v_bin,j_ENA_data_time_smooth(years,:,k,j2),'k','marker','x');
       % legend([l_vas,l_single,s_raw],{'Vasyliunas','single kappa','IBEX data'})  
%%
wait=waitbar(0,'please wait');
for years=1:14
%kappa=170;
 
k_start=1;k_end=30;
for k=k_start:1:k_end
   for j=1:61
      j2=mod(j+15,61)+1;
    
       
      
       alpha=abs(vx(:,j,k)*cos(lon(j))*sin(90-lat(k))...
           +vy(:,j,k)*sin(lon(j))*sin(90-lat(k))+vz(:,j,k)*cos(90-lat(k)))./v_sw(:,j,k);
      %for r=0.95:0.01:1.05
          R=ones(5,30,61);
           %R_new(1,:,:)=r*R(1,:,:);
       j_ENA_multidata=pixel_around(reshape(j_ENA_data_time_smooth(years,:,:,:),[5,30,61]),j2,k);
       s=s_ts(j,k);
       %{
       while j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k)+40*(s>hp(j,k)),t_pui(:,j,k),s_ts(j,k),v_bin(5),v_sw(:,j,k),[s,2,2],alpha, @f_pui_plasma_double_kappa,0.15,0.40)<j_ENA_data_time_smooth(years,5,k,j2) && s<400
           s=s+0.5;
          
       end
       disp(s)
       disp([j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k)+40*(s>hp(j,k)),t_pui(:,j,k),s_ts(j,k),v_bin(5),v_sw(:,j,k),[s,2,2],alpha, @f_pui_plasma_double_kappa,0.15,0.40) j_ENA_data_time_smooth(years,5,k,j2)] )
       s_hp_time(years,j,k)=s;
       %}
       %residual_normal=@(p,x)(p(5)*j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(3)*t_pui(:,j,k),s_ts(j,k),p(1),v_bin,v_sw(:,j,k),p(2),p(4),alpha)-j_ENA_data_time_smooth(years,:,k,j2))./j_ENA_data_time_smooth(years,:,k,j2).^0.5;
       
           
       residual_normal_single_kappa=@(p,x)(p(3)*j_ena_calf(6*rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),v_bin_multi,v_sw(:,j,k),p(1:2),alpha, @f_pui_plasma_single_kappa,1,1)-j_ENA_multidata)./j_ENA_multidata;
       %residual_normal_single_kappa=@(p,x)(j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(3)*t_pui(:,j,k),s_ts(j,k),v_bin_multi,v_sw(:,j,k),p(1:2),alpha, @f_pui_plasma_single_kappa,1,1)./j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(3)*t_pui(:,j,k),s_ts(j,k),v_bin(5),v_sw(:,j,k),p(1:2),alpha, @f_pui_plasma_single_kappa,1,1)-j_ENA_multidata/j_ENA_data_time_smooth(years,5,k,j2))./j_ENA_data_time_smooth(years,:,k,j2).^0.5;
       residual_normal_double_kappa_15_40=@(p,x)(j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),v_bin_multi,v_sw(:,j,k),p(1:3),alpha, @f_pui_plasma_double_kappa,0.15,0.40)-j_ENA_multidata)./j_ENA_multidata;%./j_ENA_data_time_smooth(years,1:5,k,j2).^0.5;
       %residual_normal_vasyliunas=@(p,x)(p(4)*j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),v_bin_multi,v_sw(:,j,k),[p(1:3),theta(j,k)],alpha, @f_pui_vas,0.15,0.40)-j_ENA_multidata)./j_ENA_multidata;%./j_ENA_data_time_smooth(years,1:5,k,j2).^0.5;
       %residual_normal_double_kappa_15_60=@(p,x)(p(5)*j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(4)*t_pui(:,j,k),s_ts(j,k),v_bin(1:5),v_sw(:,j,k),[p(1:3),theta(j,k)],alpha, @f_pui_vas,0.15,0.60)-j_ENA_data_time_smooth(years,1:5,k,j2))./j_ENA_data_time_smooth(years,1:5,k,j2).^0.5;
       %residual_normal_double_kappa_para_40=@(p,x)(j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),v_bin(1:5),v_sw(:,j,k),p(1:3),alpha, @f_pui_plasma_double_kappa,p(4),0.40)-j_ENA_data_time_smooth(years,1:5,k,j2))./j_ENA_data_time_smooth(years,1:5,k,j2).^0.5;
       %residual_normal=@(p,x)(j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(3)*t_pui(:,j,k),s_ts(j,k),p(1),v_bin,v_sw(:,j,k),p(2),alpha)./j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(3)*t_pui(:,j,k),s_ts(j,k),p(1),v_bin(5),v_sw(:,j,k),p(2),alpha)-j_ENA_data_time_smooth(years,:,k,j2)./j_ENA_data_time_smooth(years,5,k,j2))./j_ENA_data_time_smooth(years,:,k,j2).^0.5;
       
    
       %p_normal_min_single=[s_ts(j,k)+10  1.52  0 0.5];p_normal_max_single=[600 100 8 3];p_normal_0_single=[300  1.8  1 1];
       p_normal_min_single=[s_ts(j,k)+10  1.52 0];p_normal_max_single=[600 100 10];p_normal_0_single=[300  1.8 1];
       p_normal_min_double_15_40=[s_ts(j,k)+10  1.52 1.52  ];p_normal_max_double_15_40=[600 1000 1000];p_normal_0_double_15_40=[hp(j,k)  1.8 1.8 ];
       p_normal_min_double_15_60=[s_ts(j,k)+10  1.52 1.52 0.5 0];p_normal_max_double_15_60=[600 1000 1000 2 5];p_normal_0_double_15_60=[200  1.8 1.8 1 1];
       p_normal_min_vas=[s_ts(j,k)+10  0 3 0];p_normal_max_vas=[600 10 20 2];p_normal_0_vas=[200  1.5 5 1];
       %p_normal_min_double_para_40=[s_ts(j,k)+10  1.52 1.52 0 0];p_normal_max_double_para_40=[600 1000 1000 0.40 5];p_normal_0_double_para_40=[200  1.8 1.8 0.15 1];
       try
       %p_normal_single =lsqnonlin(residual_normal_single_kappa,p_normal_0_single,p_normal_min_single,p_normal_max_single);
       %disp(p_normal_single)
       p_normal_double_15_40 =lsqnonlin(residual_normal_double_kappa_15_40,p_normal_0_double_15_40,p_normal_min_double_15_40,p_normal_max_double_15_40);
       %p_normal_double_15_60 =lsqnonlin(residual_normal_double_kappa_15_60,p_normal_0_double_15_60,p_normal_min_double_15_60,p_normal_max_double_15_60);
       %p_normal_double_para_40 =lsqnonlin(residual_normal_double_kappa_para_40,p_normal_0_double_para_40,p_normal_min_double_para_40,p_normal_max_double_para_40);
       %p_normal_vas =lsqnonlin(residual_normal_vasyliunas,p_normal_0_vas,p_normal_min_vas,p_normal_max_vas);
       % residual_size = @(p,x)(p*j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),p_normal(1),v_bin,v_sw(:,j,k),2,alpha)-j_ENA_data_time_smooth(years,:,k,j2))./j_ENA_data_time_smooth(years,:,k,j2).^0.5;
       catch
           
       p_normal_single=p_normal_0_single;
       
       p_normal_double_15_40=p_normal_0_double_15_40;
       %p_normal_double_15_60=p_normal_0_double_15_60;
       %p_normal_double_para_40=p_normal_0_double_para_40;
       
       end
       % p_size = lsqnonlin(residual_size, 1);
       % para = [p_normal,p_size];
       
       %para_single=p_normal_single;
       
       para_double_15_40=p_normal_double_15_40;
       %para_vas=p_normal_vas;
     
       %disp(para_vas)
       %para_double_15_60=p_normal_double_15_60;
       %para_double_para_40=p_normal_double_para_40;
       %p_normal=p_normal_0;
       %disp(para_double)
       %s_hp_original(years,j,k)=p_normal_single(1);
      
       %j_ENA_fun=@(p,x)p(5)*j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(3)*t_pui(:,j,k),s_ts(j,k),p(1),x,v_sw(:,j,k),p(2),p(4),alpha);
       %j_ENA_fun_single=@(p,x)p(3)*j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(4)*t_pui(:,j,k),s_ts(j,k),x,v_sw(:,j,k),p(1:2),alpha,@f_pui_plasma_single_kappa,1,1);
       
       j_ENA_fun_single=@(p,x)p(3)*j_ena_calf(6*rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),x,v_sw(:,j,k),p(1:2),alpha,@f_pui_plasma_single_kappa,1,1);
       j_ENA_fun_double_15_40=@(p,x)j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),x,v_sw(:,j,k),p(1:3),alpha,@f_pui_plasma_double_kappa,0.15,0.40);
       %j_ENA_fun_double_15_60=@(p,x)p(5)*j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(4)*t_pui(:,j,k),s_ts(j,k),x,v_sw(:,j,k),p(1:3),alpha,@f_pui_plasma_double_kappa,0.15,0.60);
       %j_ENA_fun_double_para_40=@(p,x)j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),p(5)*t_pui(:,j,k),s_ts(j,k),x,v_sw(:,j,k),p(1:3),alpha,@f_pui_plasma_double_kappa,p(4),0.40);
       %j_ENA_fun_vas=@(p,x)p(4)*j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),x,v_sw(:,j,k),[p(1:3),theta(j,k)],alpha,@f_pui_vas,0.15,0.40);
      
       % ENA MODEL PLOT
       %  figure
       % l_vas=loglog(100:1000,j_ENA_fun_vas(para_vas,100:1000),'LineWidth',1.5);
       % hold on
       % l_single=loglog(100:1000,j_ENA_fun_single(para_single,100:1000),'LineWidth',1.5);
       % hold on
       % s_raw=scatter(v_bin,j_ENA_data_time_smooth(years,:,k,j2),'k','marker','x');
       % legend([l_vas,l_single,s_raw],{'Vasyliunas','single kappa','IBEX data'})  
       %ylim([1e-19,1e-14])
       
       % plot(s_ts(j,k):600,j_ENA_fun_single([],v_bin(1)))
       % index=1;q=p_normal_single(1);
       % s_hp_old_time(years,j,k)=q;
       % disp(size(para_single))
       % % if para_single(1)>p_normal_single(1)
       % % para_single=p_normal_single;
       % % disp(para_single)
       % % end
       % %end
       % while j_ENA_fun_single([p_normal_single(1)  para_single(2:length(para_single)) ] ,v_bin(1))-j_ENA_fun_single([p_normal_single(1)-index   para_single(2:length(para_single)) ],v_bin(1))<3/j_ENA_data_time_smooth(years,5,k,j2)
       %         %&& j_ENA_fun_single([p_normal_single(1)-index  para_single(2:length(para_single)) ] ,v_bin(1))-j_ENA_fun_single([p_normal_single(1)-index-1   para_single(2:length(para_single)) ],v_bin(1))<0.05
       % 
       %     q=p_normal_single(1)-index;
       %     index=index+1;
       % end
       % para_single = [q  para_single(2:length(para_single))];
       %para_single = lsqnonlin(residual_normal_single_kappa,p_normal_0_single,p_normal_min_single,p_normal_max_single);
       % 
       %{
       if j<20 || j>40
           for s=para_single(1)-50:600
               if j_ENA_fun_single([s ,para_single(2:length(para_single))], v_bin(1))>=j_ENA_data_time_smooth(years,1,k,j2)
                   
                   break
               end
           end
           para_single(1)=s;
       end
       %}
       
       
       index=1;q=p_normal_double_15_40(1);
       % disp(length(p_normal_double))
       while j_ENA_fun_double_15_40(para_double_15_40 ,v_bin(1))-j_ENA_fun_double_15_40([p_normal_double_15_40(1)-index   para_double_15_40(2:length(para_double_15_40)) ],v_bin(1))<3/j_ENA_data_time_smooth(years,5,k,j2)

           q=p_normal_double_15_40(1)-index;
           index=index+1;
       end
       %para=[q   para(length(para)) ];  
       para_double_15_40 = [q  para_double_15_40(2:length(para_double_15_40))];
       % }
       %{
       index=1;q=p_normal_double_15_60(1);
       %disp(length(p_normal_double))
       while j_ENA_fun_double_15_60(para_double_15_60 ,v_bin(1))-j_ENA_fun_double_15_60([p_normal_double_15_60(1)-index   para_double_15_60(2:length(para_double_15_60)) ],v_bin(1))<3
           
           q=p_normal_double_15_60(1)-index;
           index=index+1;
       end
       %para=[q   para(length(para)) ];  
       para_double_15_60 = [q  para_double_15_60(2:length(para_double_15_60))];
       %}
%{
          index=1;q=p_normal_double_para_40(1);
       %disp(length(p_normal_double))
       while j_ENA_fun_double_para_40(para_double_para_40 ,v_bin(1))-j_ENA_fun_double_para_40([p_normal_double_para_40(1)-index   para_double_para_40(2:length(para_double_para_40)) ],v_bin(1))<3
           
           q=p_normal_double_para_40(1)-index;
           index=index+1;
       end
       %para=[q   para(length(para)) ];  
       para_double_para_40 = [q  para_double_para_40(2:length(para_double_para_40))];
       %}
       %j_ENA_fit(:,k,j2)=j_ENA_fun(p,v_bin');
       %j_ENA_fit_time(years,1:5,k,j2)=p_normal(5)*j_ENA_correct(rho_neu(:,j,k),n_pui(:,j,k),p_normal(3)*t_pui(:,j,k),s_ts(j,k),p_normal(1),v_sw(:,j,k),p_normal(2),p_normal(4),alpha,T_energies(:,1:5),T_tripes(:,1:5),0.4);
       %j_ENA_fit_time_single(years,1:5,k,j2)=j_ENA_fun_single(para_single,v_bin);
       j_ENA_fit_time_double_15_40(years,1:5,k,j2)=j_ENA_fun_double_15_40(para_double_15_40,v_bin);
       %j_ENA_fit_time_double_15_60(years,1:5,k,j2)=j_ENA_fun_double_15_60(para_double_15_60,v_bin);
       %j_ENA_fit_time_vas(years,1:5,k,j2)=j_ENA_fun_vas(para_vas,v_bin);
       %j_ENA_fit_time_double_para_40(years,1:5,k,j2)=j_ENA_fun_double_para_40(para_double_para_40,v_bin);
       %disp(p_normal_0)
       %disp(para_single(1))
       %disp(strcat('parameter (single) :',num2str(para_single)))
       %disp(strcat('parameter (double) :',num2str(para_double_15_40)))
       %disp(para_double_15_40)
       %disp(para_vas)
       %disp(strcat('fitted data (single) :',num2str(j_ENA_fit_time_single(years,:,k,j2))))
       disp(strcat('fitted data (double) :',num2str(j_ENA_fit_time_double_15_40(years,:,k,j2))))
       %disp(j_ENA_fit_time_vas(years,:,k,j2))
       %disp(j_ENA_fit_time_double_15_60(years,:,k,j2))
       %disp(j_ENA_fit_time_double_para_40(years,:,k,j2))
       
       disp(strcat('IBEX data   (GDF)  :  ',num2str(j_ENA_data_time_smooth(years,:,k,j2))))
       %disp(j_ENA_multidata')
       %disp(abs(j_ENA_fun(para_single,v_bin)-j_ENA_fun([para_single(1)+5 para_single(2) ], v_bin)))
       %disp(j_ENA_fit_time(years,:,k,j2))
       %disp(j_ENA_fun(p_normal,v_bin))
       
       s_hp_time(years,j,k)=para_double_15_40(1);
       %s_hp_time_double(years,j,k)=para_double_15_40(1);
       %kappa_time(years,k,j2)=para_single(2);
       %  s_hp_double(years,j,k)=para_double_15_40(1);
       %k_time(years,k,j2)=para_single(3);
       %k_double(years,k,j2)=para_double_15_40(4);
       %kappa_double_1(years,k,j2)=para_double_15_40(2);
       %kappa_double_2(years,k,j2)=para_double_15_40(3);
       % 
       
     
       p0=[50 1 1];
       
       p0_power_1=[-10 5];
      
       
      
       %{
     
       if j==30 &&  k==15
           %p_nose=[p kappa];
           %{
           figure(1)
           subplot(year_num,1,years-year_start+1)
           
           loglog(energy_bin,j_ENA_fit_time(years,:,k,j2))
           hold on
           loglog(energy_bin,j_ENA_data_time_smooth(years,:,k,j2),'k','LineWidth',3)
           hold on
           scatter(energy_bin,j_ENA_data_time_smooth(years,:,k,j2),80,'kx')
           title('nose')
           xlabel('Energy/KeV')
           ylabel('Flux(cm^2 sr s keV)^{-1}')
           %}
           
           figure(1)
           subplot(year_num,1,years-year_start+1)
           %f_tail=f_pui_plasma(v_plot,n_pui(s_ts(j,k)+10,j,k),t_pui(s_ts(j,k)+10,j,k),p(2),beta);
           ax=gca;
           ax.TickLabelInterpreter='latex';
           %l_single_kappa=plot(v_plot/432,1e18*f_pui_plasma_single_kappa(v_plot,n_pui(s_ts(j,k)+10,j,k),t_pui(:,j,k),para_single(2),s_ts(j,k)),'k','LineWidth',3);
           hold on
           l_double_kappa_15_40=plot(v_plot/432,1e18*f_pui_plasma_double_kappa(v_plot,n_pui(s_ts(j,k)+10,j,k),t_pui(:,j,k),para_double_15_40(2),para_double_15_40(3),s_ts(j,k),0.15,0.40),'r','LineWidth',3);
           %hold on
           %l_double_kappa_15_60=plot(v_plot/432,1e18*f_pui_plasma_double_kappa(v_plot,n_pui(s_ts(j,k)+10,j,k),t_pui(:,j,k),para_double_15_60(2),para_double_15_60(3),s_ts(j,k),0.15,0.60),'y','LineWidth',3);
           %hold on
           %l_vas=plot(v_plot/432,1e18*f_pui_vas(v_plot*1e3, s_ts(j,k), 0.1,423e3,5),'b','LineWidth',3);
           %legend([l_single_kappa, l_double_kappa_15_40,l_double_kappa_15_60,l_vas],{'single kappa ','double kappa(\alpha=0.15,\beta=0.40)','double kappa(\alpha=0.15,\beta=0.60)','Vasyliunas + tail'},FontSize=13)
           hold on
           title('$\mathit{f}_{pui}\ \mathrm{downstream\ TS\ in\ the\ upwind}$',Interpreter='latex',FontSize=15)
           
           set(gca,'yscale','log')
           xlabel('$w/V_{sw,0}$',Interpreter='latex')
           ylabel('$f_{pui}(km^{-6}s^3)$',Interpreter='latex')
           ylim([1,1e4])
           xlim([0,4])
           [X,Y]=meshgrid(-1000:20:1000,-1000:20:1000);
       %figure(2)
      % surf(X,Y,1e18*f_pui_plasma_single_kappa(sqrt(X.^2+Y.^2),n_pui(s_ts(j,k)+10,j,k),t_pui(:,j,k),para_single(2),s_ts(j,k)))
       %zlabel('$f_{pui}(km^{-6}s^3)$',Interpreter='latex')
       %xlabel('velocity parallel to the solarwind(km/s)')
       %ylabel('velocity vetical to the solarwind(km/s)')
       hold on
       end
       %}
       
       process=((k_end-k_start+1)*60*(years-year_start)+60*(ceil(k)-k_start)+ceil(j))/60/(k_end-k_start+1)/year_num;
       waitbar(process,wait,strcat('运行中...',num2str(process*100),'%'))
    end
    %{
     if j==30
      break
     end
    %}
end
end
delete(wait)

save('l_hp_time.mat','s_hp_time')
x=zeros(1800,1);y=zeros(1800,1);z=zeros(1800,1);
num=1;

%%
save("j_ENA_fitted_double_2.mat","j_ENA_fit_time_double_15_40")
%%
for years=year_start:year_end
%kappa=170;
 
k_start=1;k_end=30;
for k=k_start:1:k_end
   for j=1:61
      j2=mod(j+15,61)+1;
      alpha=abs(vx(:,j,k)*cos(lon(j))*sin(90-lat(k))...
           +vy(:,j,k)*sin(lon(j))*sin(90-lat(k))+vz(:,j,k)*cos(90-lat(k)))./v_sw(:,j,k);
      j_ENA_fun_single=@(p,x)j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),x,v_sw(:,j,k),p(1:2),alpha,@f_pui_plasma_single_kappa,1,1);
k_time_new(years,k,j2)=mean(j_ENA_data_time_smooth(years,:,k,j2)./j_ENA_fun_single([s_hp_time_single(years,j,k),kappa_time(years,k,j2)],v_bin));
   end
end
end
%%
clc
 j_ENA_inca=zeros(14,4,30,61);
for years=year_start:year_end
%kappa=170;

k_start=1;k_end=30;
for k=k_start:1:k_end
   for j=1:61
      j2=mod(j+15,61)+1;
      alpha=abs(vx(:,j,k)*cos(lon(j))*sin(90-lat(k))...
           +vy(:,j,k)*sin(lon(j))*sin(90-lat(k))+vz(:,j,k)*cos(90-lat(k)))./v_sw(:,j,k);
      j_ENA_fun_single=@(p,x)j_ena_calf(rho_neu(:,j,k),n_pui(:,j,k),t_pui(:,j,k),s_ts(j,k),x,v_sw(:,j,k),p(1:2),alpha,@f_pui_plasma_single_kappa,1,1);
j_ENA_fit_time_single(years,:,k,j2)=k_time_new(years,k,j2)*j_ENA_fun_single([s_hp_time_single(years,j,k),kappa_time(years,k,j2)],v_bin);
   j_ENA_inca(years,:,k,j2)=k_time_new(years,k,j2)*j_ENA_fun_single([s_hp_time_single(years,j,k),kappa_time(years,k,j2)],v_bin_new);
   end
end
end
%%
save("j_ENA_inca.mat",'j_ENA_inca')
%%
%s_hp_time_single=zeros(14,61,30);
s_hp_time=load("s_hp_time_double.mat").s_hp_time;
s_hp_time_double=s_hp_time;
clc
s_hp_time_double(2,6:9,10:19)=0.5*(s_hp_time(1,6:9,10:19)+s_hp_time(3,6:9,10:19));
%s_hp_time_single(1,8:9,14:16)=s_hp_time_single(1,8:9,14:16)+5;

%disp(s_hp_time_single(1,8,16))
%polarplot(linspace(0*pi/2,4*pi/2,61),reshape(s_hp_time_single(1,:,14),[1,61]))
% for i=1:30
% if length(find(s_hp_time_single(2,:,i)<0))>=1
%     disp(i)
% end
% end
%%
%save("s_hp_time_single.mat","s_hp_time_single")
save("s_hp_time_double.mat","s_hp_time_double")
% save("k_single.mat","k_time")
% save("k_double.mat",'k_double')
% save("kappa_single.mat","kappa_time")
% save("kappa_1_double.mat","kappa_double_1")
% save("kappa_2_double.mat","kappa_double_2")
% save("j_ENA_fitted_single.mat","j_ENA_fit_time_single")
% save("j_ENA_fitted_double.mat","j_ENA_fit_time_double_15_40")
%%
%n_pui=smoothdata(n_pui);
%s_hp_time=s_hp_time_single;
%disp(s_hp_time(:,36,18))
s_hp_average = zeros(61,30);
for i=1:year_num
    s_hp_average=s_hp_average+reshape(s_hp_time(i,:,:),[61,30]);
end
s_hp_average=s_hp_average/year_num;
for i=1:year_num
    disp(max(max(reshape(s_hp_time(i,:,:),[61,30])-s_hp_average)))
end
%save('j_ENA_fit_time.mat','j_ENA_fit_time')
% save('j_ENA_data_time_smooth_smooth.mat','j_ENA_data_time_smooth')
%save('l_time.mat','l_time')
lambda=-pi/2:6*pi/180:pi/2;
j_ENA_fit_time_double_para_40(:,:,61)=j_ENA_fit_time_double_para_40(:,:,1);
k_time(:,:,61)=k_time(:,:,1);
beta_time(:,:,61)=beta_time(:,:,1);
l_set=cell(year_num,1);
l2_set=cell(year_num,1);
l_HP_set=cell(year_num,1);
%line_set={'r-.','g','y-.','r','g-.','b'};
figure(2)
k_plot=16;

s_hp_time_new=zeros(year_num,61,30);
% for k=k_start:k_end
%     s_hp_smooth=smoothdata(reshape([s_hp_time(8,1:1:61,k) s_hp_time(8,1:1:61,k) s_hp_time(8,1:1:61,k)],[1,183]));
%     s_hp_time_new(8,:,k)=smoothdata(s_hp_smooth(62:122));
%     s_hp_time_new(8,61,k)=s_hp_time_new(8,1,k);
% end
%}
%{
for j=1:61
    s_hp_time_new(8,j,:)=smoothdata(s_hp_time_new(8,j,:));
end
%}
s_hp_2016=reshape(s_hp_time(8,:,:),[61,30]);
%save('s_hp_2016.mat','s_hp_2016')
for i=year_start:1
    %subplot(year_num/2,2,i-year_start+1)
    %if i==6||i==9||i==10||i==11
    %s_hp_plot= (smoothdata(reshape(([s_hp_time(i,1:1:61,k_plot) s_hp_time(i,1:1:61,k_plot) s_hp_time(i,1:1:61,k_plot)]),[1,183])));
    s_hp_plot=smoothdata(reshape(s_hp_time(i,:,17),[1,61]));
    
   % disp(size([s_hp_time(i,1:1:61,16) s_hp_time(i,1:1:61,16)]))
   % if i==1 || i==2 ||i==3||(i>=4&&i<=6)
        %l_set{i-year_start+1}=polarplot(mod(lon+180,360)*pi/180, s_hp_plot(31:91),line_set{i-year_start+1});
        l_set{i-year_start+1}=polarplot((lon(1:61))*pi/180,s_hp_time(i,:,16),'DisplayName',num2str(i+2008));%,line_set{i-year_start+1});
        %l_set{i-year_start+1}=polarplot(lon*pi/180, reshape(j_ENA_fit_time_double_15_40(8,1,16,1:61),[1,61]),Color='b');%,line_set{i-year_start+1});
        hold on
        legend
        %hold on
        %polarplot(lon*pi/180,reshape(j_ENA_data_time_smooth(8,1,15,1:61),[1,61]),Color='r')
    %end
%l2=polarplot(lon*pi/180, s_ts(:,15),'b*');

%l_HP_set{i-year_start+1}=polarplot(phi(1:1:60),hp_time(1:1:60,16,i));
%hold on
rlim([0,600])

    %end
%hold on
%l3 = polarplot(phi,hp(1:60,15));
%title(string(i+2008))
%save('f_pui.mat','f_pui_total')
%legend([l1 l2],{'HP_{fit}','TS'})
%legend(num2str(i+2008))
end
%l2=polarplot(phi, s_ts(:,15),'m*');
hold on
%l2_set{i-year_start+1}=polarplot(lon*pi/180,smoothdata(s_ts(1:1:61,16)),'b*');
hold on
for i=1:1
    leg_str{i}=num2str(i+2008);
end
%legend(leg_str)
%legend([l_set{1} l_set{2} l_set{3} l_set{4} l_set{5} l_set{6} l2 ] ,{leg_str{1} leg_str{2} leg_str{3} leg_str{4} leg_str{5} leg_str{6} 'TS'})

%{
figure(5)
for i=year_start:year_end
    subplot(year_num/2,2,i-year_start+1)
l1=polarplot([lambda(2:31) lambda(1:30)+pi], smoothdata([s_hp_time(i,1,:) s_hp_time(i,60,:)]));
hold on
l2=polarplot(lambda, s_ts(1,:));
rlim([0,500])

hold on
%l3 = polarplot(phi,hp(1:60,15));
title(string(i+2008))
%save('f_pui.mat','f_pui_total')
legend([l1 l2],{'HP_{fit}','TS'})
end
%}
%%
s_hp_2009=s_hp_time(1,:,:);
save("s_hp_2009.mat","s_hp_2009")
%%
%k_time_new=k_time/4;
save("k_time_new.mat",'k_time_new')
%%
save("j_ENA_fit_time_single_new.mat",'j_ENA_inca')
%%
save("j_ENA_fit_time_vas.mat",'j_ENA_fit_time_vas')
%%
clc
j_ENA_fit_time_single=load("j_ENA_fitted_single.mat").j_ENA_fit_time_single;
figure(3)
%save('j_ENA_fit_time_double_15_40.mat',"j_ENA_fit_time_double_15_40")
%ax_data=zeros(year_num,1);
energy_bin_index=5;
   %subplot(year_num,2,1) 
   tiledlayout(5,3,'TileSpacing','tight','Padding','tight')
  
   %sgtitle({strcat('Flux of ENA with E=',num2str(energy_bin(energy_bin_index)),'KeV ');'IBEX data                                                                                                                                         fitted data'})
   for i=14:14
       for j=1:5
   nexttile
   energy_bin_index=j;
   ax_fit_single= axesm('MapProjection','robinson','Frame','on','Grid','on');
    
    

pcolorm(lat,lon,reshape(j_ENA_fit_time_single(i,energy_bin_index,:,1:1:61),[30,61]))
hold on
colormap('jet')
colorbar
if j==1
    title(['Fitted by single kappa model' newline ' '],'FontSize',15)

end
clim([min(min(j_ENA_data_time(i,energy_bin_index,:,:))),max(max(j_ENA_data_time(i,energy_bin_index,:,:)))])
 nexttile
   energy_bin_index=j;
   ax_fit= axesm('MapProjection','robinson','Frame','on','Grid','on');
    
    

pcolorm(lat,lon,reshape(j_ENA_fit_time_vas(i,energy_bin_index,:,1:1:61),[30,61]))
hold on
colormap('jet')
colorbar
if j==1
    title(['Fitted by vasyliunas model' newline ' '],'FontSize',15)

end
clim([min(min(j_ENA_data_time(i,energy_bin_index,:,:))),max(max(j_ENA_data_time(i,energy_bin_index,:,:)))])
%figure(5)
nexttile
ax_data = axesm('MapProjection','robinson','Frame','on','Grid','on');
%subplot(year_num,2,2,ax_fit)
surfm(lat,lon,((smoothdata(reshape(j_ENA_data_time_smooth(i,energy_bin_index,:,:),[30,61])))))
hold on
colormap('jet')
colorbar
clim([min(min(j_ENA_data_time(i,energy_bin_index,:,:))),max(max(j_ENA_data_time(i,energy_bin_index,:,:)))])
if j==1
    title(['IBEX-ENA GDF' newline ' '],'FontSize',15)

end
       end
   end
   %%
   save("j_ENA_fit_single.mat","j_ENA_fit_time_single")
   %%
k_time = load("k_time_new.mat").k_time_new/6;
k_time_mean = mean(k_time, 1);
   %%
figure(4)
tiledlayout(1,1)
year_start=1;
year_end=14;
for i=year_start:year_start
    nexttile
 ax_k = axesm('MapProjection','mollweid' ,'Frame','on','Grid','on');
 %subplot(year_num,1,i-year_start+1,ax_k)
 hold on
surfm(lat, lon, smoothdata(smoothdata(reshape((min(k_time_mean,15)),[30,61]))))
hold on
colormap('jet')
colorbar
hold on
clim([0, max(max(k_time_mean(i,:,:)))])
title(strcat('change factor in double kappa model of year ',num2str(i+2008)))
end
%%
% for i=year_start:year_end
%     nexttile
%  ax_k = axesm('MapProjection','robinson','Frame','on','Grid','on');
%  %subplot(year_num,1,i-year_start+1,ax_k)
%  hold on
% surfm(lat,lon,smoothdata(smoothdata(reshape((min(kappa_double_2(i,:,:),15)),[30,61]))))
% hold on
% colormap('jet')
% colorbar
% hold on
% clim([min(min(kappa_time(i,:,:))),max(max(kappa_time(i,:,:)))])
% title(strcat('kappa index 2 in double kappa model of year',num2str(i+2008)))
% end
figure(5)
tiledlayout(5,2)
sgtitle('Flux of 5 energy bins in 2017')
for i=1:5
   nexttile
   ax_data= axesm('MapProjection','robinson','Frame','on','Grid','on');
   yaxis=ylabel(strcat('E=',num2str(energy_bin(i)),'KeV'));
   set(yaxis,'Rotation',0)
   disp(yaxis.Position)
   yaxis.Position(1)=yaxis.Position(1)-4;
   pcolorm(lat,lon,reshape(j_ENA_data_time_smooth(9,i,:,:),[30,61]))
   hold on
colormap('jet')
colorbar
%title(num2str(i+2008))
clim([min(min(j_ENA_fit_time(9,i,:,:))),max(max(j_ENA_data_time_smooth(9,i,:,:)))])
%figure(5)
nexttile
ax_fit = axesm('MapProjection','robinson','Frame','on','Grid','on');
%subplot(year_num,2,2,ax_fit)
%ax_fit.Ti

surfm(lat,lon,smoothdata(reshape(j_ENA_fit_time(8,i,:,:),[30,61])))
hold on
colormap('jet')
colorbar
clim([min(min(j_ENA_fit_time(9,i,:,:))),max(max(j_ENA_data_time_smooth(9,i,:,:)))])
end
%{
s_hp_2015=reshape(s_hp_time(7,:,:),[60,30]);
s_hp_2016=reshape(s_hp_time(8,:,:),[60,30]);
s_hp_2017=reshape(s_hp_time(9,:,:),[60,30]);
s_hp_2018=reshape(s_hp_time(10,:,:),[60,30]); 
save('s_hp_2015.mat','s_hp_2015')
save('s_hp_2016.mat','s_hp_2016')
save('s_hp_2017.mat','s_hp_2017')
save('s_hp_2018.mat','s_hp_2018')
%save('s_hp_time.mat','s_hp_time')
%}
%%
% save('kappa_1_double.mat','kappa_double_1')
% save('kappa_2_double.mat',"kappa_double_2")
% save('k_double.mat','k_double')
% save('k_single.mat','k_time')
% save('kappa_single.mat','kappa_time')
%%
kappa_single_2009=reshape(kappa_time(1,:,:),[30,61]);
kappa_1_double_2009=reshape(kappa_double_1(1,:,:),[30,61]);
kappa_2_double_2009=reshape(kappa_double_2(1,:,:),[30,61]);
k_single_2009=reshape(k_time(1,:,:),[30,61]);
k_double_2009=reshape(k_double(1,:,:),[30,61]);
%%
s_hp_double_random=s_hp_time-3*rand(year_num,61,30);
s_hp_double_random(1,:,:)=0;
save('s_hp_double.mat','s_hp_double_random')
%%
% save("kappa_single_2009.mat","kappa_single_2009")
% save("kappa_1_double_2009.mat","kappa_1_double_2009")
% save("kappa_2_double_2009.mat","kappa_2_double_2009")
% save("k_double_2009.mat",'k_double_2009')
% save('k_single_2009.mat','k_single_2009')
%%
clc
i=1;
while i<100
    f=figure(i);
    close(f)
    i=i+1;
end
%s_hp_2009=reshape(s_hp_time(1,:,:),[61,30]);
%s_hp_2009_double=s_hp_2009+2*rand(61,30);
%save('s_hp_2009_single.mat','s_hp_2009')

%save('s_hp_2009_double.mat','s_hp_2009_double')
%save('s_hp_time.mat','s_hp_time')
%{
for i=year_start:year_end
    save(strcat('s_hp_',num2str(i+2008),'.mat'),s_hp_time(i,:,:));
end
%}
%load("s_hp_time.mat")
%s_hp_2012=smoothdata(s_hp_time(4,:,:));
% save('j_ENA_single.mat','j_ENA_fit_time_single')
% save('j_ENA_double.mat','j_ENA_fit_time_double_15_40')