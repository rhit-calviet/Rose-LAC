%% Control Law
%

%% Setup
clear
close all
clc

%% Parameter
ts = 2;
damping = 2;

%% Control Parameters
tau = ts*(1+damping)/12;
p = 4/ts;

kp = 3*tau*p^2;
kd = 3*tau*p - 1;
ki = tau*p^3;

%% Control Law
s = tf('s');
C = (kp + kd*s + ki/s) / (tau*s + 1);

%% Plant
P = 1/s;

%% Open-Loop
Gol = P*C;

%% Closed-Loop
Gcl = P*C / (1 + P*C);

stepinfo(Gcl)