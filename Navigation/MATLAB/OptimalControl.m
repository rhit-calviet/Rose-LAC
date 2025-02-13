%% Optimal Control
%

%% Setup
clear
close all
clc

%% Variables
syms px py theta real % State variables
syms v omega real % Control outputs
syms rx ry rtheta % Reference variables

%% Vectors
x = [px; py; theta];
u = [v; omega];
r = [rx; ry; rtheta];

%% Pseudo-Linearized State Space
syms theta_bar v_bar omega_bar real
ubar = [v_bar; omega_bar];

x_lin = [x; 1];
A = [0, 0, -v_bar*sin(theta_bar), v_bar*cos(theta_bar);
     0, 0,  v_bar*cos(theta_bar), v_bar*sin(theta_bar);
     0, 0,  0,                    omega_bar;
     0, 0,  0,                    0                  ];

B = [cos(theta_bar), 0;
     sin(theta_bar), 0;
     0,              1;
     0,              0];

%% QR Matrix
syms Q1 Q2 Q3 positive
syms R1 R2 positive

Q = diag([Q1, Q2, Q3, 0]);
R = diag([R1, R2]);

%% State Costate Dynamics
M = [A,  B*inv(R)*B.';
    -Q, -A.'         ];

M = simplify(M);

Sigma = expm(M);
