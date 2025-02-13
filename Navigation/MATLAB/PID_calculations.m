clear
close all
clc


syms kd wn zeta w tau positive

kp = 2*zeta*wn*(1+kd);
ki = wn*wn*(1+kd);

s = w*1j;

G = (kd*s^2 + kp*s/(tau*s+1) + ki) / ((1+kd)*s^2 + kp*s + ki);

G = simplify(G);

phase = atan(imag(G)/real(G));

simplify(phase);

dphase = diff(phase,w);

soln = solve(dphase==0,w);

phase1 = simplify(subs(phase,w,soln(1)));
phase2 = simplify(subs(phase,w,soln(2)));