%% Cell Turn Acceleration Limited Profile

%% Setup
clear
close all
clc

%% Variables
syms v a t v_lin real
th_f = sym(pi/2);

ta = v/a;
tf = v/a + th_f/v;

th1 = a*t^2/2;
th2 = v*ta/2 + v*(t-ta);
th3 = th_f-a*(tf-t)^2/2;


dx1 = int(cos(th1), t, 0, ta);

dx2 = int(v_lin*cos(th2), t, ta, tf-ta);

dx3 = int(v_lin*cos(th3), t, tf-ta, tf);

dx = dx1+dx2+dx3;

syms t_curr real
dx1t = int(cos(th1),t,0, t_curr);
dx2t = dx1 + int(v_lin*cos(th2), t, ta, t_curr);
dx3t = dx1 + dx2 + int(v_lin*cos(th3), t, tf-ta, t_curr);


%%
clear



v_max = 3;

tf = @(v, a) pi/(2*v) + v/a;




function [C, Ceq] = nonlocon(x)
    v_lin = 0.3;
    dx = @(v, a) (pi^(1/2)*fresnelc(v/(a^(1/2)*pi^(1/2))))/a^(1/2) + (2^(1/2)*v_lin*cos(v^2/(2*a) + pi/4))/v + (v_lin*pi^(1/2)*fresnels(v/(a^(1/2)*pi^(1/2))))/a^(1/2);
    cell_w = 0.15*5;
    dx_targ = cell_w/2;
    Ceq = dx_targ - dx(x(1), x(2));
    C = x(1)^2-pi*x(2)/2;
end

minfunc = @(x) x(2);

soln = fmincon(minfunc, [0;0], [], [], [], [], [0;0], [v_max;inf], @nonlocon);

disp(soln);

