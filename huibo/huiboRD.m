 %% 正侧视点目标成像--RDA（原始信号-距离压缩-方位向fft-RCMC-方位压缩）
 clc;
 clear;
 close all;
 %% 参数设置
 c=3e8;                        %光速
 fc=3e9;                       %载波频率（Hz)
 lambda=c/fc;                  %波长0.1（m）
 Ralpha=1.2;  Aalpha=1.3;      %距离向、方位向过采样率
 %----------观察场景相关参数----------%
 H=8000;                       %平台运行高度
 Yc=6000;                      %场景中心线长度
 R0=sqrt(H^2+Yc^2);            
 Xmin=-200;                    %[Xmin Xmax]
 Xmax=200;                     %在合成孔径长度Ls基础上，额外方位向范围
 Yw=300;                      
 %----------天线孔径设置----------%
 La=1.5;
 theta=0.886*lambda/La;
 Ls=R0*theta;
 %----------慢时间参数设置----------%
 v=150;                        %平台运行速度
 Ts=Ls/v;
 Xwid=Ls+Xmax-Xmin;            %慢时间域时间窗长度
 Twid=Xwid/v;
 Ka=2*v^2/lambda/R0;           %方位向调频率
 Ba=abs(Ka*Ts);                %方位向带宽
 PRF=Aalpha*Ba;
 PRT=1/PRF;
 dx=PRT;                       %方位向采样间隔
 N=ceil(Twid/dx);
 N=2^nextpow2(N);              %提高fft效率
 x=linspace((Xmin-Ls/2)/v,(Xmax+Ls/2)/v,N);
                               %慢时间域时间序列
 X=v*x;                        %慢时间域时间序列对应方位向距离
 PRT=Twid/N;                   %更新
 PRF=1/PRT;
 dx=PRT;
 %----------快时间参数设置----------%
 Tr=5e-6;                      %脉冲宽度
 Br=100e6;                     %带宽100MHz
 Kr=Br/Tr;                     %调频率
 Fs=Ralpha*Br;                 %距离向采样率
 dt=1/Fs;
 Rmin=sqrt((Yc-Yw)^2+H^2);
 Rmax=sqrt((Yc+Yw)^2+H^2);
 Rm=Rmax-Rmin+c*Tr/2;          %斜距测绘带宽
 M=ceil(2*Rm/c/dt);            %采样点数 
 M=2^nextpow2(M);       
 t=linspace(2*Rmin/c-Tr/2,2*Rmax/c+Tr/2,M);
                               %快时间域时间序列
 r=c*t/2;                      %快时间域时间序列对应斜距
 dt=(2*Rmax/c+Tr-2*Rmin/c)/M;  %更新
 Fs=1/dt;
 %----------目标参数设置----------%
 Ntarget=3;
 Ptarget=[0,R0,1;
          50,R0+100,0.8;
          100,R0+100,0.8];
% load 1000x1000.mat
% [S_i,S_j] = size(S);
% DEM_resoluton = 12.5; % DEM图像分辨率
% S(i,j) 坐标：X = Yc + (i-1) * DEM_resoluton
%              Y = (j-1) * DEM_resoluton

 %----------后向散射系数矩阵计算----------%

% r1 = sqrt((xr(i)-T(k,1)).^2 + (T(k,2)+yr).^2);
% X = (T(k,1)+250)*2+1;
% Y = (T(k,2)-1)*2+1;
% n = [Nx(X,Y),Ny(X,Y),Nz(X,Y)];  % 该点的法线
% d = [xr(i)-T(k,1),yr-T(k,2)*0.5,zr-T(k,3)];
% mag_n = sqrt(sum(n .* n));
% mag_d = sqrt(sum(d .* d));
% xita = acos(sum(n .* d)/(mag_n * mag_d)); %计算入射角
% ss = 10*log10((0.0133*cos(xita))/(sin(xita)+0.1*cos(xita))^3);
% 
% for i = 1: S_i
%     for j = 1:S_j
%         R = sqrt
%     end
% end
            
 

 %% 生成SAR回波
%  h=waitbar(0,'SAR回波生成');
%  s0=zeros(N,M);
%  for k=1:Ntarget
%     R=sqrt(Ptarget(k,2)^2+(X-Ptarget(k,1)).^2); 
%     delay=2*R/c;               %距离徙动带来的距离时间延时
%     Delay=ones(N,1)*t-delay'*ones(1,M);
%     Phase=1j*pi*Kr*Delay.^2-1j*4*pi*fc*(R'*ones(1,M))/c;
%     s0=s0+Ptarget(k,3)*rectpuls(Delay/Tr).*...
%       rectpuls((X-Ptarget(k,1))'*ones(1,M)/Ls).*exp(Phase);
%    %s0=s0+Ptarget(k,3)*exp(Phase).*(abs(Delay)<=Tr/2).*(abs((X-Ptarget(k,1))'*ones(1,M))<=Lsar/2);
%     waitbar(k/Ntarget);
%  end
 
h=waitbar(0,'SAR回波生成');
s0=zeros(N,M);
for k=1:Ntarget
    for i=1:N 
       R = sqrt(Ptarget(k, 2)^2 + (X(i) - Ptarget(k, 1))^2); 
       delay = R * 2 / c;
       for j=1:M 
            Delay = t(j) - delay;
            Phase = 1j * (pi * Kr * Delay * Delay - (4 * pi * fc * R) / c);
            
                if i == 100 && j == 100 
                    fprintf("%f, %f, %f, %f\n", X(i), Ptarget(k, 1), Ls, abs((X(i) - Ptarget(k, 1)) / Ls));
                end
            
            if abs(Delay / Tr) < 0.5 && abs((X(i) - Ptarget(k, 1)) / Ls) < 0.5
                s0(i, j) = s0(i, j) + Ptarget(k, 3) * exp(Phase);
            end
       end
    end
    waitbar(k/Ntarget);
end

 waitbar(k/Ntarget);
 close(h);
 
 %% 导入数据
 echo_real = csvread("echo_real.csv");
 echo_imag = csvread("echo_imag.csv");
 s0 = echo_real + 1j*echo_imag;
 
 %% 距离压缩
 %mf=zeros(1,M); 
 %t1=linspace(-Tr/2,Tr/2,ceil(Tr/dt));
 %mf1=exp(-1j*pi*Kr*t1.^2);    %生成距离向匹配滤波器，但其长度不等于M
 %N0=ceil((M-length(mf1))/2);   
 %mf(N0:N0+length(mf1)-1)=mf1; %生成距离向匹配滤波器，长度等于M
 %mf_f=fty(mf);                 
 %s0_f=fty(s0);
 %src_f=s0_f.*(ones(N,1)*(mf_f));%在频域进行脉冲压缩
 %src=ifty(src_f); 
 %距离频域匹配滤波
 fr=(-M/2:M/2-1)*Fs/M;
 mf_f=exp(1j*pi*fr.^2/Kr);
 src_f=fty(s0).*(ones(N,1)*(mf_f));
 src=ifty(src_f);
 %% 方位向fft
 Src=ftx(src);
 %% 距离徙动校正RCMC
 fa=(-N/2:N/2-1)*PRF/N;        %多普勒频率，即方位向频率
 RCM=lambda^2*R0*fa.^2/8/v^2;  %距离徙动与多普勒频率的关系式
 RCM_point=RCM*2*Fs/c;         %距离徙动对应的距离向点数
 N_interp=8;                   %进行8点插值
 N_add=N_interp*ceil(max(RCM_point));
                               %必须大于插值后、距离徙动所占的采样点数
 Src_RCMC=zeros(N,M);
 h=waitbar(0,'sinc插值，请稍后...');
 for k1=1:N
     n_rcm=round(((1:M)+RCM_point(k1)-1)*N_interp+1);     %round表示四舍五入取整
     Src_interp=zeros(1,M*N_interp+N_add);                
     Src_interp(1:M*N_interp)=interp(Src(k1,:),N_interp); %该数组存放每一行信号插值后的数！
     Src_RCMC(k1,:)=Src_interp(n_rcm);
     waitbar(k1/N)
 end
 close(h);
 %% 方位向压缩、方位向ifft
 %mfa_f=exp(-1j*pi*fa.^2/Ka);                              %方位向匹配滤波器
 %Src_mfa=Src_RCMC.*(mfa_f.'*ones(1,M));                   %在方位向频域进行脉压
 %sac=iftx(Src_mfa);                                       %方位向ifft
 Src_mfa=zeros(N,M);
 h=waitbar(0,'方位向压缩');
 for k=1:M
     mfa_f=exp(1j*pi/(-2*v^2/lambda/r(k))*fa.^2);
     Src_mfa(:,k)=Src_RCMC(:,k).*mfa_f.';
     waitbar(k/M);                    
 end
 close(h);
 sac=iftx(Src_mfa);
 %% 图形绘制
 %----------绘制原始信号----------%
 %imagesc(r*1e-3,X,abs(s0));
 Z1=20*log10(abs(s0)+1e-6);
 Zm1=max(max(Z1));
 Zn1=Zm1-40;                   %显示动态范围40dB
 Z1=(Zm1-Zn1)*(Z1-Zn1).*(Z1>Zn1);
 figure(1);
 imagesc(r*1e-3,X,-Z1);
 colormap(gray);               %绘制灰度图
 axis tight;
 xlabel('距离向（km）\rightarrow');
 ylabel('\leftarrow方位向（m）');
 title('原始信号');
 %----------绘制距离压缩后的信号----------%
 figure(2);
 imagesc(r*1e-3,X,abs(src));
 axis tight;
 xlabel('距离向（km）\rightarrow');
 ylabel('\leftarrow方位向（m）');
 title('距离压缩后的信号');
 %----------压缩后的距离多普勒域信号----------%
 figure(3);
 imagesc(abs(Src));
 axis tight;
 xlabel('距离向（采样点数）\rightarrow');
 ylabel('\leftarrow方位向（采样点数）');
 title('方位fft信号-距离多普勒域');      
                               %-实际情况下此处可能会产生多普勒模糊-%
 %----------RCMC后的距离多普勒域信号----------%
 figure(4);
 imagesc(abs(Src_RCMC));
 axis tight;
 xlabel('距离向（采样点数）\rightarrow');
 ylabel('\leftarrow方位向（采样点数）');
 title('RCMC-距离多普勒域');
 %----------RDA----------%
 %figure(5);
 %imagesc(r*1e-3,X,abs(sac));
 Z2=20*log10(abs(sac)+1e-6);
 Zm2=max(max(Z2));
 Zn2=Zm2-40;                   %显示动态范围40dB
 Z2=(Zm2-Zn2)*(Z2-Zn2).*(Z2>Zn2);
 figure(5);
 imagesc(r*1e-3,X,-Z2);
 colormap(gray);               %绘制灰度图
 axis tight;
 xlabel('距离向（km）\rightarrow');
 ylabel('\leftarrow方位向（m）');
 title('RDA最终信号');
 %----------三维成像图----------%
 figure(6);
 subplot(121);
 mesh(abs(sac(980:1050,480:550)));
 axis tight;
 subplot(122);
 mesh(abs(sac(980:1330,480:700)));
 axis tight;
 %----------距离剖面、方位剖面----------%
 [location_a,location_r]=find(abs(sac)==max(max(abs(sac))));
 Ns=8;                         %8点插值
 sac_interp_r=interp(sac(:,location_r).',Ns);
 sac_interp_r_abs=abs(sac_interp_r);
 sac_interp_r_absmax=sac_interp_r_abs/max(sac_interp_r_abs);
 sac_interp_r_log=20*log10(sac_interp_r_absmax);
 rr=find(sac_interp_r_log==max(sac_interp_r_log));
 
 sac_interp_a=interp(sac(location_a,:),Ns);
 sac_interp_a_abs=abs(sac_interp_a);
 sac_interp_a_absmax=sac_interp_a_abs/max(sac_interp_a_abs);
 sac_interp_a_log=20*log10(sac_interp_a_absmax);
 aa=find(sac_interp_a_log==max(sac_interp_a_log));
 
 pslr_r=pslrfunc(sac_interp_r_abs);  %计算距离剖面图峰值旁瓣比dB
 islr_r=islrfunc(sac_interp_r_abs);  %计算距离剖面图积分旁瓣比dB
 pslr_a=pslrfunc(sac_interp_a_abs);  %计算方位剖面图峰值旁瓣比dB
 islr_a=islrfunc(sac_interp_a_abs);  %计算方位剖面图积分旁瓣比dB
 
 figure(7);
 subplot(221);
 plot(sac_interp_r_log);
 axis([rr-150,rr+150,-30,0]);
 ylabel('幅度dB'); title('(a)距离剖面图幅度');
 
 subplot(222);
 plot(sac_interp_a_log);
 axis([aa-150,aa+150,-30,0]);
 ylabel('幅度dB'); title('(b)方位剖面图幅度');
 
 subplot(223);
 plot(angle(sac_interp_r));
 axis([rr-150,rr+150,-4,4]);
 xlabel('距离向（采样点）'); ylabel('相位 度'); 
 title('(c)距离剖面图相位');
 
 subplot(224);
 plot(angle(sac_interp_a));
 axis([aa-150,aa+150,-4,4]);
 xlabel('方位向（采样点）'); ylabel('相位 度'); 
 title('(d)方位剖面图相位');
 

 
 
 
 