# Embedded Multisensor Systems (EMS) Laboratory
# Director: Mohamed Atia
#
# All Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of EMS and its Director.
# The intellectual and technical concepts contained herein are proprietary to EMS and its Director.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from EMS.
#
# This code has been modified by Alexander Crain on January 31, 2025. All modifications remain the property of EMS.

#==============================================================================#
# PREAMBLE
#==============================================================================#

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
system_noise_factor = 0.001  # Used to scale Q matrix
measurement_noise_factor = 1  # Used to scale R matrix
initial_covariance = 5  # Initial covariance value

# Define any known constants
theta = 60 * np.pi / 180  # Direction angle in radians

# Define any control parameters
KF_UPDATE = False  # If True, the Kalman filter include the update step (measurement update)

#==============================================================================#
# DATA LOADING AND INITIALIZATION
#==============================================================================#

# Load the data from the comman-delimited text file, while ignoring the first row (headers)
data = np.loadtxt('Project_1_observations_data.txt', delimiter=',', skiprows=1)

# From the imported data, the time vector and the north and east observations are stored in numpy arrays
t = data[:, 0]
E_observed = data[:, 1]
N_observed = data[:, 2]

# Set the initial parameters for the Kalman filter
true_initial_state = np.array([4.3553, 7.5435, 19.4374, 33.6666, 0])  # Represents the true initial states
F = np.array([[0, 0, 1, 0, 0], 
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, np.cos(theta)],
              [0, 0, 0, 0, np.sin(theta)],
              [0, 0, 0, 0, 0]])  # The state transition matrix
H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])  # The observation matrix
G = np.diag([0, 0, 0, 0, 1])  # The noise shaping matrix
Q = system_noise_factor * np.diag([1, 1, 1, 1, 1])  # The system noise covariance matrix
R = measurement_noise_factor * np.diag([1, 1])  # The measurement noise covariance matrix
P = np.zeros((5, 5, len(t)))  # The state covariance matrix, initialized as a 3D array of zeros
P[:, :, 0] = initial_covariance * np.eye(5)  # Setting the initial covariance matrix 
x = np.zeros((5, len(t)))  # The state vector, initialized as a 2D array of zeros
x[:, 0] = true_initial_state  # Setting the initial state vector based on the true initial states

P_E = np.zeros(len(t))  # The variance of the east position
P_N = np.zeros(len(t))  # The variance of the north position
P_VE = np.zeros(len(t))  # The variance of the east velocity
P_VN = np.zeros(len(t))  # The variance of the north velocity 
P_a = np.zeros(len(t))  # The variance of the acceleration

dT = np.mean(np.diff(t))  # The time step
P_E[0] = P[0, 0, 0]  # Setting the initial variance of the east position
P_N[0] = P[1, 1, 0]  # Setting the initial variance of the north position
P_VE[0] = P[2, 2, 0]  # Setting the initial variance of the east velocity
P_VN[0] = P[3, 3, 0]  # Setting the initial variance of the north velocity
P_a[0] = P[4, 4, 0]  # Setting the initial variance of the acceleration

#==============================================================================#
# KALMAN FILTER IMPLEMENTATION
#==============================================================================#

for i in range(len(t) - 1):

    # Check if the user wants to disable the update step
    if KF_UPDATE:  # This block of code will be executed if the user wants to include the update step
        
        # KF Prediction Step
        Phi = np.eye(5) + F * dT  
        Qd = dT**2 * G @ Q @ G.T  
        x[:, i + 1] = Phi @ x[:, i]  
        P[:, :, i + 1] = Phi @ P[:, :, i] @ Phi.T + Qd

        # KF Update Step
        K = P[:, :, i + 1] @ H.T @ np.linalg.inv(H @ P[:, :, i + 1] @ H.T + R)
        P[:, :, i + 1] -= K @ H @ P[:, :, i + 1]
        error_states = K @ (np.array([E_observed[i + 1], N_observed[i + 1]]) - H @ x[:, i + 1])
        x[:, i + 1] += error_states

        # Store the variance values
        P_E[i + 1] = P[0, 0, i + 1]
        P_N[i + 1] = P[1, 1, i + 1]
        P_VE[i + 1] = P[2, 2, i + 1]
        P_VN[i + 1] = P[3, 3, i + 1]
        P_a[i + 1] = P[4, 4, i + 1]

    else:  # This block of code will be executed if the user wants to disable the update step

        # KF Prediction Step
        Phi = np.eye(5) + F * dT
        Qd = dT**2 * G @ Q @ G.T
        x[:, i + 1] = Phi @ x[:, i]
        P[:, :, i + 1] = Phi @ P[:, :, i] @ Phi.T + Qd

        # Store the variance values
        P_E[i + 1] = P[0, 0, i + 1]
        P_N[i + 1] = P[1, 1, i + 1]
        P_VE[i + 1] = P[2, 2, i + 1]
        P_VN[i + 1] = P[3, 3, i + 1]
        P_a[i + 1] = P[4, 4, i + 1]

#==============================================================================#
# TRUE STATE CALCULATION
#==============================================================================#

# The Kalman filter appears to converge to an acceleration of -5.0 m/s^2
# Using the kinematic equation, we can calculate the true position

# Initialize the true state vectors to zero arrays
E_true = np.zeros(len(t))
N_true = np.zeros(len(t))
VE_true = np.zeros(len(t))
VN_true = np.zeros(len(t))
a_true = -5*np.ones(len(t))

# Set the initial true state values
E_true[0] = 4.3553
N_true[0] = 7.5435
VE_true[0] = 19.4374
VN_true[0] = 33.6666

# Calculate the true state values from the kinematics
for i in range(len(t) - 1):
    E_true[i + 1] = E_true[i] + VE_true[i] * dT + 0.5 * a_true[i] * np.cos(theta) * dT**2
    N_true[i + 1] = N_true[i] + VN_true[i] * dT + 0.5 * a_true[i] * np.sin(theta) * dT**2
    VE_true[i + 1] = VE_true[i] + a_true[i] * np.cos(theta) * dT
    VN_true[i + 1] = VN_true[i] + a_true[i] * np.sin(theta) * dT

#==============================================================================#
# PLOTTING
#==============================================================================#

if KF_UPDATE:

    # Plot the estimated and measured east position
    plt.figure()
    plt.plot(t, x[0, :], 'r', label='Kalman Filter Solution')
    plt.plot(t, E_observed, 'k', label='Noisy Observations')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('East Position (m)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('East Position (m)', fontsize=14)
    plt.savefig('east_position_part3.pdf')

    # Plot the estimated and measured north position
    plt.figure()
    plt.plot(t, x[1, :], 'r', label='Kalman Filter Solution')
    plt.plot(t, N_observed, 'k', label='Noisy Observations')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('North Position (m)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('North Position (m)', fontsize=14)
    plt.savefig('north_position_part3.pdf')

    # Plot the variances for the east and north position, east and north velocity, and acceleration
    plt.figure()
    plt.plot(t, P_E, 'r', label='East Position Variance (m$^2$)')
    plt.plot(t, P_N, 'g', label='North Position Variance (m$^2$)')
    plt.plot(t, P_VE, 'b', label='East Velocity Variance (m$^2$/s$^2$)')
    plt.plot(t, P_VN, 'c', label='North Velocity Variance (m$^2$/s$^2$)')
    plt.plot(t, P_a, 'm', label='Acceleration Variance (m$^2$/s$^4$)')
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlabel('Time (s)', fontsize=14)
    plt.savefig('variances_part3.pdf')

    # Plot the estimated east velocity
    plt.figure()
    plt.plot(t, x[2,:], 'r', label='KF Estimated East Velocity')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('Estimated East Velocity (m/s)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Velocity (m/s)', fontsize=14)
    plt.savefig('east_velocity_part3.pdf')

    # Plot the estimated north velocity
    plt.figure()
    plt.plot(t, x[3,:], 'r', label='KF Estimated North Velocity')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('Estimated North Velocity (m/s)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Velocity (m/s)', fontsize=14)
    plt.savefig('north_velocity_part3.pdf')

    # Plot the estimated acceleration
    plt.figure()
    plt.plot(t, x[4,:], 'r', label='KF Estimated Acceleration')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('Estimated Acceleration (m/s$^2$)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Acceleration (m/s$^2$)', fontsize=14)
    plt.savefig('acceleration_part3.pdf')

    # Plot the true, measured, and estimated east positions
    plt.figure()
    plt.plot(t, E_true, 'r', label='True East Position')
    plt.plot(t, x[0, :], 'b', label='KF Estimated East Position')
    plt.plot(t, E_observed, 'k', label='Noisy Observations')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('East Position (m)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Position (m)', fontsize=14)
    plt.savefig('east_position_true_vs_estimated_part3.pdf')

    # Plot the true, measured, and estimated north positions
    plt.figure()
    plt.plot(t, N_true, 'r', label='True North Position')
    plt.plot(t, x[1, :], 'b', label='KF Estimated North Position')
    plt.plot(t, N_observed, 'k', label='Noisy Observations')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('North Position (m)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Position (m)', fontsize=14)
    plt.savefig('north_position_true_vs_estimated_part3.pdf')

else:  # For the case where no update step is included

    # Plot the true, estimated, and measured east position with no update step
    plt.figure()
    plt.plot(t, E_true, 'r', label='True East Position')
    plt.plot(t, x[0, :], 'b', label='KF Estimated East Position')
    plt.plot(t, E_observed, 'k', label='Noisy Observations')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('East Position (m)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Position (m)', fontsize=14)
    plt.savefig('east_position_true_vs_estimated_part3_no_update.pdf')

    # Plot the true, estimated, and measured north position with no update step
    plt.figure()
    plt.plot(t, N_true, 'r', label='True North Position')
    plt.plot(t, x[1, :], 'b', label='KF Estimated North Position')
    plt.plot(t, N_observed, 'k', label='Noisy Observations')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('North Position (m)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Position (m)', fontsize=14)
    plt.savefig('north_position_true_vs_estimated_part3_no_update.pdf')


# Calculate the root-mean-square error (RMSE) for the north position
rmse_est_vs_true = np.sqrt(np.sum((x[1, :]-N_true)**2) / len(N_true)) # estimate vs. true
rmse_meas_vs_true = np.sqrt(np.sum((N_observed - N_true)**2) / len(N_true)) # measurement vs. true
rmse_est_vs_meas = np.sqrt(np.sum((x[1, :] - N_observed)**2) / len(N_observed)) # estimate vs. measurement

print('RMSE Estimate vs. True (N): ', rmse_est_vs_true)
print('RMSE Measurement vs. True (N): ', rmse_meas_vs_true)
print('RMSE Estimate vs. Measurement (N): ', rmse_est_vs_meas)

# Calculate the root-mean-square error (RMSE) for the east position
rmse_est_vs_true = np.sqrt(np.sum((x[0, :] - E_true)**2) / len(E_true)) # estimate vs. true
rmse_meas_vs_true = np.sqrt(np.sum((E_observed - E_true)**2) / len(E_true)) # measurement vs. true
rmse_est_vs_meas = np.sqrt(np.sum((x[0, :] - E_observed)**2) / len(E_observed)) # estimate vs. measurement

print('RMSE Estimate vs. True (E): ', rmse_est_vs_true)
print('RMSE Measurement vs. True (E): ', rmse_meas_vs_true)
print('RMSE Estimate vs. Measurement (E): ', rmse_est_vs_meas)

# Show the plots
plt.show()