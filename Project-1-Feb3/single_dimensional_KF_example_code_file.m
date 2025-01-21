%{
 * Embedded Multisensor Systems (EMS) Laboratory
 * Director: Mohamed Atia
 *
 *  All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains the property of EMS and its Director,
 * The intellectual and technical concepts contained herein are proprietary to EMS and its Director,
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from EMS.
 */
%}

% Kalman Filter 1D motion Illustration Example
set(0,'DefaultFigureWindowStyle','docked');
close all;
clear;
clc;

% Parameters
observation_noise_std       =  25; %Used to add noise to observed y
system_noise_factor         = 0.01; %Used to scale Q matrix
measurement_noise_factor    = 0.001; %Used to scale R matrix

% --> Observed y (noisy) and time data
y_observed = [8.57,12.41,19.56,17.32,22.61,30.46,34.03,35.35,37.90,38.27,44.55,43.48,49.75,51.68,56.39,54.94,59.97,62.63,60.58,65.29,70.87,69.04,69.77,73.36,73.12,76.40,76.35,77.93,80.33,83.68,82.40,86.29,77.90,82.56,84.88,82.71,82.02,87.06,87.27,86.58,86.86,83.62,84.37,85.08,83.25,85.33,89.85,82.06,84.12,79.05,84.03,77.20,78.99,75.67,77.14,74.44,72.76,78.59,67.90,67.52,72.41,60.79,62.91,56.60,59.54,55.35,53.64,48.69,47.27,47.52,35.23,33.82,35.83,29.94,27.53,21.12,19.10,16.74,12.24,8.09,5.30,0.00,1.62,-8.40,-5.67,-15.54,-22.71,-25.22,-26.01,-30.49,-37.60,-42.26,-46.47,-56.87,-59.94,-68.43,-70.44,-72.51,-78.75,-80.58];
t = [0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00,2.10,2.20,2.30,2.40,2.50,2.60,2.70,2.80,2.90,3.00,3.10,3.20,3.30,3.40,3.50,3.60,3.70,3.80,3.90,4.00,4.10,4.20,4.30,4.40,4.50,4.60,4.70,4.80,4.90,5.00,5.10,5.20,5.30,5.40,5.50,5.60,5.70,5.80,5.90,6.00,6.10,6.20,6.30,6.40,6.50,6.60,6.70,6.80,6.90,7.00,7.10,7.20,7.30,7.40,7.50,7.60,7.70,7.80,7.90,8.00,8.10,8.20,8.30,8.40,8.50,8.60,8.70,8.80,8.90,9.00,9.10,9.20,9.30,9.40,9.50,9.60,9.70,9.80,9.90];
%--> True y
y_true = [8.71,12.55,16.29,19.93,23.48,26.92,30.27,33.52,36.68,39.73,42.69,45.55,48.31,50.97,53.54,56.01,58.38,60.65,62.82,64.90,66.88,68.76,70.54,72.22,73.81,75.30,76.69,77.98,79.17,80.27,81.27,82.17,82.97,83.68,84.29,84.79,85.21,85.52,85.73,85.85,85.87,85.79,85.62,85.34,84.97,84.50,83.93,83.27,82.50,81.64,80.68,79.62,78.47,77.21,75.86,74.41,72.87,71.22,69.48,67.64,65.70,63.66,61.53,59.29,56.96,54.53,52.01,49.38,46.66,43.84,40.92,37.91,34.79,31.58,28.27,24.86,21.36,17.75,14.05,10.25,6.36,2.36,-1.73,-5.92,-10.21,-14.60,-19.09,-23.67,-28.35,-33.13,-38.00,-42.98,-48.05,-53.22,-58.49,-63.86,-69.32,-74.88,-80.54,-86.30];

%--> Add more noise to the observed y (left to student)

%--> Initialization
true_initial_state =[8.7105; 38.8748; -9.7923];% initial position, initial velocity, and true acceleration
F = [0 1 0;0 0 1;0 0 0];
H = [1 0 0];
G = diag([0 0 1]);
Q = system_noise_factor*diag([1 1 1]);
R = measurement_noise_factor*1;
P(:,:,1) = diag([5; 5; 5;]);
x(1,1) = true_initial_state(1);
x(2,1) = true_initial_state(2);
x(3,1) = 0.0;
dT = mean(diff(t));
estimated_accel_bias = zeros(1,length(t));
P_p = zeros(1,length(t));
P_v = zeros(1,length(t));
P_a = zeros(1,length(t));
P_p(1) = P(1,1,1);
P_v(1) = P(2,2,1);
P_a(1) = P(3,3,1);

for i = 1:length(t)-1
    %--> KF Prediction
    % Calculate Transition Matrix
    Phi = eye(3,3) + F*dT;
    % Calculate System Noise Matrix
    Qd = dT^2*G*Q*G';
    % Predict States
    x(:,i+1) = Phi*x(:,i);
    % Predict State Error Covariance
    P(:,:,i+1) = Phi*P(:,:,i)*Phi' + Qd;
    %--> KF Update
    % Calculate Kalman Gain
    K = P(:,:,i+1)*H'/(H*P(:,:,i+1)*H'+R);
    % Update error covariance matrix P
    P(:,:,i+1) = P(:,:,i+1) - K*H*P(:,:,i+1);
    % Calculate error state
    error_states = K*(y_observed(i+1) - H*x(:,i+1));
    % Correct states
    x(:,i+1) = x(:,i+1) + error_states;
    %--> keep the error state covariance diagnoal elements for plots
    P_p(i+1) = P(1,1,i+1);
    P_v(i+1) = P(2,2,i+1);
    P_a(i+1) = P(3,3,i+1);
end

figure;
plot(t,x(1,1:end),'r');hold on;grid on;
plot(t,y_observed,'k');hold on;grid on;
plot(t,y_true,'b');
legend('Kalman Filter solution','noisy observations','true Y');
title('position(m)');xlabel ('time(s)');ylabel('position(m)');

figure;
plot(t,x(3,:),'r');hold on;grid on;
plot(t,true_initial_state(3)*ones(1,length(t)),'-.','color','b');
legend('KF estimated acceleration','True acceleration');
title('estimated acceleration(m/s2)');xlabel ('time(s)');ylabel('acceleration(m/s2)');

figure;
plot(t,P_p,'r');hold on;grid on;
plot(t,P_v,'g');hold on;grid on;
plot(t,P_a,'b');hold on;grid on;
legend('position variance','velocity variance','acceleration variance');
title('state error covariance');

RMSE    = sqrt(sum((y_true - x(1,:)).^2)/numel(y_true))
