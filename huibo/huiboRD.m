 %% �����ӵ�Ŀ�����--RDA��ԭʼ�ź�-����ѹ��-��λ��fft-RCMC-��λѹ����
 clc;
 clear;
 close all;
 %% ��������
 c=3e8;                        %����
 fc=3e9;                       %�ز�Ƶ�ʣ�Hz)
 lambda=c/fc;                  %����0.1��m��
 Ralpha=1.2;  Aalpha=1.3;      %�����򡢷�λ���������
 %----------�۲쳡����ز���----------%
 H=8000;                       %ƽ̨���и߶�
 Yc=6000;                      %���������߳���
 R0=sqrt(H^2+Yc^2);            
 Xmin=-200;                    %[Xmin Xmax]
 Xmax=200;                     %�ںϳɿ׾�����Ls�����ϣ����ⷽλ��Χ
 Yw=300;                      
 %----------���߿׾�����----------%
 La=1.5;
 theta=0.886*lambda/La;
 Ls=R0*theta;
 %----------��ʱ���������----------%
 v=150;                        %ƽ̨�����ٶ�
 Ts=Ls/v;
 Xwid=Ls+Xmax-Xmin;            %��ʱ����ʱ�䴰����
 Twid=Xwid/v;
 Ka=2*v^2/lambda/R0;           %��λ���Ƶ��
 Ba=abs(Ka*Ts);                %��λ�����
 PRF=Aalpha*Ba;
 PRT=1/PRF;
 dx=PRT;                       %��λ��������
 N=ceil(Twid/dx);
 N=2^nextpow2(N);              %���fftЧ��
 x=linspace((Xmin-Ls/2)/v,(Xmax+Ls/2)/v,N);
                               %��ʱ����ʱ������
 X=v*x;                        %��ʱ����ʱ�����ж�Ӧ��λ�����
 PRT=Twid/N;                   %����
 PRF=1/PRT;
 dx=PRT;
 %----------��ʱ���������----------%
 Tr=5e-6;                      %������
 Br=100e6;                     %����100MHz
 Kr=Br/Tr;                     %��Ƶ��
 Fs=Ralpha*Br;                 %�����������
 dt=1/Fs;
 Rmin=sqrt((Yc-Yw)^2+H^2);
 Rmax=sqrt((Yc+Yw)^2+H^2);
 Rm=Rmax-Rmin+c*Tr/2;          %б�������
 M=ceil(2*Rm/c/dt);            %�������� 
 M=2^nextpow2(M);       
 t=linspace(2*Rmin/c-Tr/2,2*Rmax/c+Tr/2,M);
                               %��ʱ����ʱ������
 r=c*t/2;                      %��ʱ����ʱ�����ж�Ӧб��
 dt=(2*Rmax/c+Tr-2*Rmin/c)/M;  %����
 Fs=1/dt;
 %----------Ŀ���������----------%
 Ntarget=3;
 Ptarget=[0,R0,1;
          50,R0+100,0.8;
          100,R0+100,0.8];
% load 1000x1000.mat
% [S_i,S_j] = size(S);
% DEM_resoluton = 12.5; % DEMͼ��ֱ���
% S(i,j) ���꣺X = Yc + (i-1) * DEM_resoluton
%              Y = (j-1) * DEM_resoluton

 %----------����ɢ��ϵ���������----------%

% r1 = sqrt((xr(i)-T(k,1)).^2 + (T(k,2)+yr).^2);
% X = (T(k,1)+250)*2+1;
% Y = (T(k,2)-1)*2+1;
% n = [Nx(X,Y),Ny(X,Y),Nz(X,Y)];  % �õ�ķ���
% d = [xr(i)-T(k,1),yr-T(k,2)*0.5,zr-T(k,3)];
% mag_n = sqrt(sum(n .* n));
% mag_d = sqrt(sum(d .* d));
% xita = acos(sum(n .* d)/(mag_n * mag_d)); %���������
% ss = 10*log10((0.0133*cos(xita))/(sin(xita)+0.1*cos(xita))^3);
% 
% for i = 1: S_i
%     for j = 1:S_j
%         R = sqrt
%     end
% end
            
 

 %% ����SAR�ز�
%  h=waitbar(0,'SAR�ز�����');
%  s0=zeros(N,M);
%  for k=1:Ntarget
%     R=sqrt(Ptarget(k,2)^2+(X-Ptarget(k,1)).^2); 
%     delay=2*R/c;               %�����㶯�����ľ���ʱ����ʱ
%     Delay=ones(N,1)*t-delay'*ones(1,M);
%     Phase=1j*pi*Kr*Delay.^2-1j*4*pi*fc*(R'*ones(1,M))/c;
%     s0=s0+Ptarget(k,3)*rectpuls(Delay/Tr).*...
%       rectpuls((X-Ptarget(k,1))'*ones(1,M)/Ls).*exp(Phase);
%    %s0=s0+Ptarget(k,3)*exp(Phase).*(abs(Delay)<=Tr/2).*(abs((X-Ptarget(k,1))'*ones(1,M))<=Lsar/2);
%     waitbar(k/Ntarget);
%  end
 
h=waitbar(0,'SAR�ز�����');
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
 
 %% ��������
 echo_real = csvread("echo_real.csv");
 echo_imag = csvread("echo_imag.csv");
 s0 = echo_real + 1j*echo_imag;
 
 %% ����ѹ��
 %mf=zeros(1,M); 
 %t1=linspace(-Tr/2,Tr/2,ceil(Tr/dt));
 %mf1=exp(-1j*pi*Kr*t1.^2);    %���ɾ�����ƥ���˲��������䳤�Ȳ�����M
 %N0=ceil((M-length(mf1))/2);   
 %mf(N0:N0+length(mf1)-1)=mf1; %���ɾ�����ƥ���˲��������ȵ���M
 %mf_f=fty(mf);                 
 %s0_f=fty(s0);
 %src_f=s0_f.*(ones(N,1)*(mf_f));%��Ƶ���������ѹ��
 %src=ifty(src_f); 
 %����Ƶ��ƥ���˲�
 fr=(-M/2:M/2-1)*Fs/M;
 mf_f=exp(1j*pi*fr.^2/Kr);
 src_f=fty(s0).*(ones(N,1)*(mf_f));
 src=ifty(src_f);
 %% ��λ��fft
 Src=ftx(src);
 %% �����㶯У��RCMC
 fa=(-N/2:N/2-1)*PRF/N;        %������Ƶ�ʣ�����λ��Ƶ��
 RCM=lambda^2*R0*fa.^2/8/v^2;  %�����㶯�������Ƶ�ʵĹ�ϵʽ
 RCM_point=RCM*2*Fs/c;         %�����㶯��Ӧ�ľ��������
 N_interp=8;                   %����8���ֵ
 N_add=N_interp*ceil(max(RCM_point));
                               %������ڲ�ֵ�󡢾����㶯��ռ�Ĳ�������
 Src_RCMC=zeros(N,M);
 h=waitbar(0,'sinc��ֵ�����Ժ�...');
 for k1=1:N
     n_rcm=round(((1:M)+RCM_point(k1)-1)*N_interp+1);     %round��ʾ��������ȡ��
     Src_interp=zeros(1,M*N_interp+N_add);                
     Src_interp(1:M*N_interp)=interp(Src(k1,:),N_interp); %��������ÿһ���źŲ�ֵ�������
     Src_RCMC(k1,:)=Src_interp(n_rcm);
     waitbar(k1/N)
 end
 close(h);
 %% ��λ��ѹ������λ��ifft
 %mfa_f=exp(-1j*pi*fa.^2/Ka);                              %��λ��ƥ���˲���
 %Src_mfa=Src_RCMC.*(mfa_f.'*ones(1,M));                   %�ڷ�λ��Ƶ�������ѹ
 %sac=iftx(Src_mfa);                                       %��λ��ifft
 Src_mfa=zeros(N,M);
 h=waitbar(0,'��λ��ѹ��');
 for k=1:M
     mfa_f=exp(1j*pi/(-2*v^2/lambda/r(k))*fa.^2);
     Src_mfa(:,k)=Src_RCMC(:,k).*mfa_f.';
     waitbar(k/M);                    
 end
 close(h);
 sac=iftx(Src_mfa);
 %% ͼ�λ���
 %----------����ԭʼ�ź�----------%
 %imagesc(r*1e-3,X,abs(s0));
 Z1=20*log10(abs(s0)+1e-6);
 Zm1=max(max(Z1));
 Zn1=Zm1-40;                   %��ʾ��̬��Χ40dB
 Z1=(Zm1-Zn1)*(Z1-Zn1).*(Z1>Zn1);
 figure(1);
 imagesc(r*1e-3,X,-Z1);
 colormap(gray);               %���ƻҶ�ͼ
 axis tight;
 xlabel('������km��\rightarrow');
 ylabel('\leftarrow��λ��m��');
 title('ԭʼ�ź�');
 %----------���ƾ���ѹ������ź�----------%
 figure(2);
 imagesc(r*1e-3,X,abs(src));
 axis tight;
 xlabel('������km��\rightarrow');
 ylabel('\leftarrow��λ��m��');
 title('����ѹ������ź�');
 %----------ѹ����ľ�����������ź�----------%
 figure(3);
 imagesc(abs(Src));
 axis tight;
 xlabel('�����򣨲���������\rightarrow');
 ylabel('\leftarrow��λ�򣨲���������');
 title('��λfft�ź�-�����������');      
                               %-ʵ������´˴����ܻ����������ģ��-%
 %----------RCMC��ľ�����������ź�----------%
 figure(4);
 imagesc(abs(Src_RCMC));
 axis tight;
 xlabel('�����򣨲���������\rightarrow');
 ylabel('\leftarrow��λ�򣨲���������');
 title('RCMC-�����������');
 %----------RDA----------%
 %figure(5);
 %imagesc(r*1e-3,X,abs(sac));
 Z2=20*log10(abs(sac)+1e-6);
 Zm2=max(max(Z2));
 Zn2=Zm2-40;                   %��ʾ��̬��Χ40dB
 Z2=(Zm2-Zn2)*(Z2-Zn2).*(Z2>Zn2);
 figure(5);
 imagesc(r*1e-3,X,-Z2);
 colormap(gray);               %���ƻҶ�ͼ
 axis tight;
 xlabel('������km��\rightarrow');
 ylabel('\leftarrow��λ��m��');
 title('RDA�����ź�');
 %----------��ά����ͼ----------%
 figure(6);
 subplot(121);
 mesh(abs(sac(980:1050,480:550)));
 axis tight;
 subplot(122);
 mesh(abs(sac(980:1330,480:700)));
 axis tight;
 %----------�������桢��λ����----------%
 [location_a,location_r]=find(abs(sac)==max(max(abs(sac))));
 Ns=8;                         %8���ֵ
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
 
 pslr_r=pslrfunc(sac_interp_r_abs);  %�����������ͼ��ֵ�԰��dB
 islr_r=islrfunc(sac_interp_r_abs);  %�����������ͼ�����԰��dB
 pslr_a=pslrfunc(sac_interp_a_abs);  %���㷽λ����ͼ��ֵ�԰��dB
 islr_a=islrfunc(sac_interp_a_abs);  %���㷽λ����ͼ�����԰��dB
 
 figure(7);
 subplot(221);
 plot(sac_interp_r_log);
 axis([rr-150,rr+150,-30,0]);
 ylabel('����dB'); title('(a)��������ͼ����');
 
 subplot(222);
 plot(sac_interp_a_log);
 axis([aa-150,aa+150,-30,0]);
 ylabel('����dB'); title('(b)��λ����ͼ����');
 
 subplot(223);
 plot(angle(sac_interp_r));
 axis([rr-150,rr+150,-4,4]);
 xlabel('�����򣨲����㣩'); ylabel('��λ ��'); 
 title('(c)��������ͼ��λ');
 
 subplot(224);
 plot(angle(sac_interp_a));
 axis([aa-150,aa+150,-4,4]);
 xlabel('��λ�򣨲����㣩'); ylabel('��λ ��'); 
 title('(d)��λ����ͼ��λ');
 

 
 
 
 