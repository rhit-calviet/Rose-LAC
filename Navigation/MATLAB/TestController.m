clear
close all
clc

%% Run

f_samp = 20;
w_samp = f_samp*2*pi;

speed = 1;

wn = 1;
zeta = 1;


kd = 25 / zeta;
kp = (1+kd)*(2*zeta*wn);
ki = (1+kd)*wn*wn;

u_max = 0.4;

%% Integrator
I = 0;
D = 0;
error_prev = 1;

%% Reference
r = @(t) 1;

%% Plant
xdot = @(t,x,u) u;

%% Simulation Parameters
x0 = 0;
stop_time = 10;
Ts = 0.05;

alpha = Ts/(Ts + 1/(wn*0.1));

% Initial conditions
x = x0;
t = 0;

% Initialize sets to store our States, Inputs, and References in during the simulation. 
States                      = [];
Inputs                      = [];
Time                        = [];

u = 0;

% Loop through our time horizon, one "measurement sample" at a time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k                       = 1:stop_time/Ts
    %% Compute Control Output
    error = r(t) - x;

    I = I + error * Ts;
    if abs(I) > u_max/ki
        I = sign(I) * u_max/ki;
    end

    D = (1-alpha)*D + (alpha)*(error-error_prev)/Ts;
    error_prev = error;

    u = error*kp + I*ki + D*kd;

    if abs(u) > u_max
        u = u_max * sign(u);
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
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
figure(1);
plot(Time,States);
figure(2);
plot(Time,Inputs);