%% PID
clear
close all
clc

%% Run

f_samp = 20;
w_samp = f_samp*2*pi;

speed = 1;

wn = 1;
zeta = 10;

tau_filt = 1/wn;


kd = 25 / zeta;
kp = (1+kd)*(2*zeta*wn);
ki = (1+kd)*wn*wn;

s = tf('s');
C = kp + kd*s/(tau_filt*s+1) + ki/s;
P = 1/s;

Gcl = (C*P)/(1+C*P);

info = stepinfo(Gcl);

step(Gcl);
hold on
plot((1:100)/f_samp, zeros(1,100),"ok");