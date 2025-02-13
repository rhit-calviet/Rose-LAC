%%
clear
close all
clc

syms wx wy wz tau real
syms dt sw sa positive

salp = 0;
sx = 0;
sv = 0;

Qc = sym(diag([salp,salp,salp, sw, sw, sw, sx,sx,sx, sv,sv,sv, sa,sa,sa]))^2;

w = [wx; wy; wz];

W =[0 -w(3) w(2) ; w(3) 0 -w(1) ; -w(2) w(1) 0 ];

I = sym(eye(3));
Z = sym(zeros(3));

F = [-W, I, Z, Z, Z;
      Z, Z, Z, Z, Z;
      Z, Z, Z, I, Z;
      Z, Z, Z, Z, I;
      Z, Z, Z, Z, Z];
%%
eF = expm(F * (dt-tau));
integrand = eF * Qc * eF.';
%%
integrand = simplify(integrand);
%%
Q = int(integrand, tau, [0, dt]);
%%
Q = simplify(Q);
Q = rewrite(Q, "exp");
%%
syms w
for i = 1:3
    Q = subs(Q, (-wx^2-wy^2-wz^2)^(1/2), w*1i);
    Q = subs(Q, ( wx^2+wy^2+wz^2)^(1/2), w);
    Q = subs(Q, exp(dt*w*1i), cos(dt*w) + 1i*sin(dt*w));
    Q = subs(Q, exp(dt*w*2i), cos(2*dt*w) + 1i*sin(2*dt*w));
    Q = expand(Q);
    Q = collect(Q);
    Q = simplify(Q);
end
%%
assume(wx ~= 0);
assume(wy ~= 0);
assume(wz ~= 0);
Q = simplify(Q);
assume(wx,"clear")
assume(wy, "clear");
assume(wz, "clear");
assume(wx, "real");
assume(wy, "real");
assume(wz, "real");
%%
ccode(Q,"File", "ccode_noise_cov.c")
%%
at0 = @(val) limit(limit(limit(val, wz, 0), wy, 0), wx, 0);

Q0 = sym(zeros(size(Q)));
Qx = sym(zeros(size(Q)));
Qy = sym(zeros(size(Q)));
Qz = sym(zeros(size(Q)));

for i = 1:size(Q,1)
    for j = 1:size(Q,2)
        expr = Q(i,j);
        Q0(i,j) = at0(expr);

        Qx(i,j) = at0(diff(expr, wx));
        Qy(i,j) = at0(diff(expr, wy));
        Qz(i,j) = at0(diff(expr, wz));

        fprintf("%.0f %.0f\n", i, j)
    end
end

%%
Qtay = Q0 + Qx*wx + Qy*wy + Qz*wz + Qxx*wx*wx + Qxy*wx*wy + Qxz*wx*wz + Qyy*wy*wy + Qyz*wy*wz + Qzz*wz*wz;
Qtay = simplify(Qtay.^2);