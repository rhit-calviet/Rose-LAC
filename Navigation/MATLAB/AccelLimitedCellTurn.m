%% Cell Turn Acceleration Limited Profile

%% Setup
clear
close all
clc

%% Variables
syms v a t real
v_bot = 0.48;
th_f = sym(pi/2);
ta = v/a;
tf = v/a + th_f/v;

th1 = a*t^2/2;
th2 = v*ta/2 + v*(t-ta);
th3 = th_f-a*(tf-t)^2/2;


dx1 = int(cos(th1), t, 0, ta);
dy1 = int(sin(th1), t, 0, ta);

dx2 = int(v_bot*cos(th2), t, ta, tf-ta);
dy2 = int(v_bot*sin(th2), t, ta, tf-ta);

dx3 = int(v_bot*cos(th3), t, tf-ta, tf);
dy3 = int(v_bot*sin(th3), t, tf-ta, tf);

dx = dx1+dx2+dx3;
dy = dy1+dy2+dy3;

%% Solve
w = 1;
syms lambda real
F = tf + lambda*(dx - w);
eqns = [diff(F,v), diff(F,a), diff(F,lambda)];

soln = vpasolve(eqns, v, a, lambda);

disp(soln);

