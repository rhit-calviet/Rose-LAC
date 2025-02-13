%% Control
%

%% Setup
clear
close all
clc

%% Parameters
f_samp = 20;
w_samp = f_samp*2*pi;

speed = 0.1;

wn = speed*w_samp/5;
zeta = 1;

tau_filt = 1/wn;

kd = 2*sqrt(wn);
kp = (1+kd)*(2*zeta*wn);
ki = (1+kd)*wn*wn;

I_v = 0;
I_w = 0;

%% Reference
r = @(t) [cos(t*0.3)-1; sin(t*0.3)];

%% Plant
xdot = @(t,x,u) [cos(x(3))*u(1); sin(x(3))*u(1); u(2)];

%% Simulation Parameters
x0 = [0;0;pi/2];
stop_time = 10;
Ts = 0.05;

% Initial conditions
x = x0;
t = 0;

% Initialize sets to store our States, Inputs, and References in during the simulation. 
States                      = [];
Inputs                      = [];
Time                        = [];

u = [0;0];

% Loop through our time horizon, one "measurement sample" at a time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k                       = 1:stop_time/Ts


    %% Compute Control Output
    dx = r(t) - x(1:2) + randn(2,1)*0.01;
    theta = x(3) + randn(1,1)*0.01;

    dist = norm(dx);

    thetad = atan2(dx(2), dx(1));

    etheta = thetad-theta;

    while etheta > pi
        etheta = etheta - 2*pi;
    end
    while etheta < -pi
        etheta = etheta + 2*pi;
    end

    ed = dist * cos(etheta);

    I_v = I_v + ed* Ts;
    I_w = I_w + etheta * Ts;

    u = [ed*kp + I_v*ki + randn()*0.02; etheta*kp + I_w*ki + randn()*0.05];

    if abs(u(1)) > 0.48
        u(1) = 0.48 * sign(u(1));
    end

    if abs(u(2)) > 4.13
        u(2) = 4.13 * sign(u(2));
    end

    % call ode45 to numerically integrate over the next time step
    tstep                   = [t, t + Ts];
    [tvals,xvals]           = ode45(xdot,tstep,x,[],u);
    
    % update the latest state "measurements"
    x                       = xvals(end,:)';
    t                       = tvals(end);
    
    % store all the state values from the time step integration
    States                  = [States;xvals];
    Time                    = [Time; tvals];

    % our input was held "constant" over that entire time step
    Inputs                  = [Inputs,u*ones(size(tvals.'))];

    %% Plots
    des = r(t);
    figure(1);
    plot(des(1), des(2), "og", States(:,1), States(:,2), "-k");
    axis([-2, 2, -2, 2]);
    axis equal
    drawnow
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    


figure(2);
subplot(2,1,1);
plot(Time, Inputs(1,:), "-k");
subplot(2,1,2);
plot(Time, Inputs(2,:), "-k");