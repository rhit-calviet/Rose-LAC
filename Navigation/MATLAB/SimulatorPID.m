%% PID
clear
close all
clc

%% Run

f_samp = 20;
w_samp = f_samp*2*pi;

speed = 0.1;

wn = speed*w_samp/5;
zeta = 2;

tau_filt = 1/wn;

kd = 2*sqrt(wn);
kp = (1+kd)*(2*zeta*wn);
ki = (1+kd)*wn*wn;

s = tf('s');
C = kp + kd*s/(tau_filt*s+1) + ki/s;
P = 1/s;

Gcl = (C*P)/(1+C*P);

% Add feedforward

figure
step(Gcl);
stepinfo(Gcl)