%% PID
clear
close all
clc

%% Run

f_samp = 20;
w_samp = f_samp*2*pi;

speed = 1;

wn = 1;
%zeta = 1;

tau_filt = 1/wn;



n = 100;
zetas = linspace(1,5,n);
kds = zeros(1,n);

kd_max = 21;

for k = 1:n
    zeta = zetas(k);

    kd_best = -1;
    OS = 1000;
    kd_range = linspace(0.8*kd_max, kd_max, 40);
    for kd = kd_range
        %kd = 2*sqrt(wn);
        kp = (1+kd)*(2*zeta*wn);
        ki = (1+kd)*wn*wn;
        
        s = tf('s');
        C = kp + kd*s/(tau_filt*s+1) + ki/s;
        P = 1/s;
        
        Gcl = (C*P)/(1+C*P);
        
        info = stepinfo(Gcl);
        if info.Overshoot < OS
            OS = info.Overshoot;
            kd_best = kd;
        end
    end
    kds(k) = kd_best;
    kd_max = kd_best;
    zeta
    kd_best
end

plot(log10(zetas), log10(kds), "-k");