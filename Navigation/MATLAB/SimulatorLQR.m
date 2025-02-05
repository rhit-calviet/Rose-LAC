%% Simulator LQR

%% Setup
clear
close all
clc

%% State Space
A = 0;

B = 1;

%% LQR Terms
Q = 1;
R = 1;

[K,S,P] = lqr(A,B,Q,R,0);