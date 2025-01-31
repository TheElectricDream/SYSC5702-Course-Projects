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

# Set the randome seed for consistent results
np.random.seed(36863142)

# Define any control parameters
# ONLY ONE CAN BE TRUE AT A TIME!!!
PART1 = True  # Set to True to run the first part of the code
PART2_TASK1 = False  # Set to True to run the second part of the code for Task 1
PART2_TASK2 = False  # Set to True to run the second part of the code for Task 2
PART2_TASK3 = False  # Set to True to run the second part of the code for Task 3
PART2_TASK4 = False  # Set to True to run the second part of the code for Task 4

if PART1:
    # Define parameters
    observation_noise_std = 25.0  # Used to add noise to observed y
    system_noise_factor = 0.01  # Used to scale Q matrix
    measurement_noise_factor = 0.001  # Used to scale R matrix
    initial_covariance = 5.0  # Initial covariance of the state

    # Define the plot names for saving data
    position_plot_name = 'position_plot.pdf'
    acceleration_plot_name = 'acceleration_plot.pdf'
    state_error_covariance_plot_name = 'state_error_covariance.pdf'

elif PART2_TASK1:
    # Define parameters
    observation_noise_std = np.array([1.0, 25.0, 50.0])  # Used to add noise to observed y
    system_noise_factor = 0.01  # Used to scale Q matrix
    measurement_noise_factor = 0.05  # Used to scale R matrix
    initial_covariance = 5.0  # Initial covariance of the state
    plot_names_array = ['position_plot_noise1.pdf', 'acceleration_plot_noise1.pdf', 'state_error_covariance_noise1.pdf',
                        'position_plot_noise25.pdf', 'acceleration_plot_noise25.pdf', 'state_error_covariance_noise25.pdf',
                        'position_plot_noise50.pdf', 'acceleration_plot_noise50.pdf', 'state_error_covariance_noise50.pdf']

elif PART2_TASK2:
    # Define parameters
    observation_noise_std = 25.0
    system_noise_factor = np.array([0.01, 10.0, 100.0])
    measurement_noise_factor = 0.05
    initial_covariance = 5.0  # Initial covariance of the state
    plot_names_array = ['position_plot_sysnoise001.pdf', 'acceleration_plot_sysnoise001.pdf', 'state_error_covariance_sysnoise001.pdf',
                        'position_plot_sysnoise10.pdf', 'acceleration_plot_sysnoise10.pdf', 'state_error_covariance_sysnoise10.pdf',
                        'position_plot_sysnoise100.pdf', 'acceleration_plot_sysnoise100.pdf', 'state_error_covariance_sysnoise100.pdf']


elif PART2_TASK3:
    # Define parameters
    observation_noise_std = 25.0
    system_noise_factor = 0.01
    measurement_noise_factor = np.array([0.001, 100.0, 10000.0])
    initial_covariance = 5.0  # Initial covariance of the state
    plot_names_array = ['position_plot_mesnoise0001.pdf', 'acceleration_plot_mesnoise0001.pdf', 'state_error_covariance_mesnoise0001.pdf',
                        'position_plot_mesnoise100.pdf', 'acceleration_plot_mesnoise100.pdf', 'state_error_covariance_mesnoise100.pdf',
                        'position_plot_mesnoise100000.pdf', 'acceleration_plot_mesnoise100000.pdf', 'state_error_covariance_mesnoise100000.pdf']


elif PART2_TASK4:
    # Define parameters
    observation_noise_std = 1.0
    system_noise_factor = 0.01
    measurement_noise_factor = 0.05
    initial_covariance = np.array([100, 0])  # Initial covariance of the state
    plot_names_array = ['position_plot_p100.pdf', 'acceleration_plot_p100.pdf', 'state_error_covariance_p100.pdf',
                        'position_plot_p0.pdf', 'acceleration_plot_p0.pdf', 'state_error_covariance_p0.pdf']


#==============================================================================#
# DATA PREPARATION 
#==============================================================================#

# Define the array containing the observed y data 
y_observed = np.array([8.57, 12.41, 19.56, 17.32, 22.61, 30.46, 34.03, 35.35, 37.90, 38.27, 44.55, 43.48, 49.75, 51.68, 56.39,
                       54.94, 59.97, 62.63, 60.58, 65.29, 70.87, 69.04, 69.77, 73.36, 73.12, 76.40, 76.35, 77.93, 80.33, 83.68,
                       82.40, 86.29, 77.90, 82.56, 84.88, 82.71, 82.02, 87.06, 87.27, 86.58, 86.86, 83.62, 84.37, 85.08, 83.25,
                       85.33, 89.85, 82.06, 84.12, 79.05, 84.03, 77.20, 78.99, 75.67, 77.14, 74.44, 72.76, 78.59, 67.90, 67.52,
                       72.41, 60.79, 62.91, 56.60, 59.54, 55.35, 53.64, 48.69, 47.27, 47.52, 35.23, 33.82, 35.83, 29.94, 27.53,
                       21.12, 19.10, 16.74, 12.24, 8.09, 5.30, 0.00, 1.62, -8.40, -5.67, -15.54, -22.71, -25.22, -26.01, -30.49,
                       -37.60, -42.26, -46.47, -56.87, -59.94, -68.43, -70.44, -72.51, -78.75, -80.58])

# Define the time array for the observed data
t = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80,
               1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70,
               3.80, 3.90, 4.00, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90, 5.00, 5.10, 5.20, 5.30, 5.40, 5.50, 5.60,
               5.70, 5.80, 5.90, 6.00, 6.10, 6.20, 6.30, 6.40, 6.50, 6.60, 6.70, 6.80, 6.90, 7.00, 7.10, 7.20, 7.30, 7.40, 7.50,
               7.60, 7.70, 7.80, 7.90, 8.00, 8.10, 8.20, 8.30, 8.40, 8.50, 8.60, 8.70, 8.80, 8.90, 9.00, 9.10, 9.20, 9.30, 9.40,
               9.50, 9.60, 9.70, 9.80, 9.90])

# Define the true y data (without noise)
y_true = np.array([8.71, 12.55, 16.29, 19.93, 23.48, 26.92, 30.27, 33.52, 36.68, 39.73, 42.69, 45.55, 48.31, 50.97, 53.54,
                    56.01, 58.38, 60.65, 62.82, 64.90, 66.88, 68.76, 70.54, 72.22, 73.81, 75.30, 76.69, 77.98, 79.17, 80.27,
                    81.27, 82.17, 82.97, 83.68, 84.29, 84.79, 85.21, 85.52, 85.73, 85.85, 85.87, 85.79, 85.62, 85.34, 84.97,
                    84.50, 83.93, 83.27, 82.50, 81.64, 80.68, 79.62, 78.47, 77.21, 75.86, 74.41, 72.87, 71.22, 69.48, 67.64,
                    65.70, 63.66, 61.53, 59.29, 56.96, 54.53, 52.01, 49.38, 46.66, 43.84, 40.92, 37.91, 34.79, 31.58, 28.27,
                    24.86, 21.36, 17.75, 14.05, 10.25, 6.36, 2.36, -1.73, -5.92, -10.21, -14.60, -19.09, -23.67, -28.35, -33.13,
                    -38.00, -42.98, -48.05, -53.22, -58.49, -63.86, -69.32, -74.88, -80.54, -86.30])

#==============================================================================#
# KALMAN FILTER IMPLEMENTATION
#==============================================================================#

# To simplify the reuse of the Kalman filter code, a function has been defined
def kalman_filter(y_observed, x, P, F, H, G, Q, R, dT):

    for i in range(len(t) - 1):

        # KF Prediction Step
        Phi = np.eye(3) + F * dT
        Qd = dT**2 * G @ Q @ G.T
        x[:, i + 1] = Phi @ x[:, i]
        P[:, :, i + 1] = Phi @ P[:, :, i] @ Phi.T + Qd

        # KF Update Step 
        K = P[:, :, i + 1] @ H.T @ np.linalg.inv(H @ P[:, :, i + 1] @ H.T + R)
        P[:, :, i + 1] -= K @ H @ P[:, :, i + 1]
        error_states = K @ (y_observed[i + 1] - H @ x[:, i + 1])
        x[:, i + 1] += error_states

        # Update the state covariance matrix
        P_p[i + 1] = P[0, 0, i + 1]
        P_v[i + 1] = P[1, 1, i + 1]
        P_a[i + 1] = P[2, 2, i + 1]

    return x, P_p, P_v, P_a

#==============================================================================#
# GENERAL PLOTTING FUNCTION
#==============================================================================#

def generate_figures(x, y_observed, y_true, t, plot_names_array, parameter_to_vary, current_parameter):
    """
    Generates and saves figures for the Kalman Filter results, including position, acceleration, and state error covariance.

    Inputs:

    x (numpy.ndarray): The state estimates from the Kalman Filter, where x[0, :] is the position and x[2, :] is the acceleration.
    y_observed (numpy.ndarray): The noisy observations of the position.
    y_true (numpy.ndarray): The true position values.
    t (numpy.ndarray): The time vector.
    plot_names_array (list): A list of strings containing the names of the plots to be saved.

    Returns: None
    """
    
    # Save figures to PDF
    plt.figure()
    plt.plot(t, x[0, :], 'r', label='Kalman Filter Solution')
    plt.plot(t, y_observed, 'k', label='Noisy Observations')
    plt.plot(t, y_true, 'b', label='True Y')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('Position (m)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Position (m)', fontsize=14)
    plt.savefig(plot_names_array[3 * parameter_to_vary.tolist().index(current_parameter)])

    plt.figure()
    plt.plot(t, x[2, :], 'r', label='KF Estimated Acceleration')
    plt.plot(t, true_initial_state[2] * np.ones(len(t)), 'b-.', label='True Acceleration')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('Estimated Acceleration (m/s$^2$)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Acceleration (m/s$^2$)', fontsize=14)
    plt.savefig(plot_names_array[3 * parameter_to_vary.tolist().index(current_parameter) + 1])

    plt.figure()
    plt.plot(t, P_p, 'r', label='Position Variance (m$^2$)')
    plt.plot(t, P_v, 'g', label='Velocity Variance (m$^2$/s$^2$)')
    plt.plot(t, P_a, 'b', label='Acceleration Variance (m$^2$/s$^4$)')
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlabel('Time (s)', fontsize=14)
    plt.title('State Error Covariance', fontsize=14)
    plt.savefig(plot_names_array[3 * parameter_to_vary.tolist().index(current_parameter) + 2])

#==============================================================================#
# MAIN CODE
#==============================================================================# 

# Call the Kalman filter function for the different cases
if PART1:

    # Set the initial parameters for the Kalman filter
    true_initial_state = np.array([8.7105, 38.8748, -9.7923])  # Initial state of the system
    F = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # The state transition matrix
    H = np.array([[1, 0, 0]])  # The observation matrix
    G = np.diag([0, 0, 1])  # The noise shaping matrix
    Q = system_noise_factor * np.diag([1, 1, 1])  # The system noise covariance matrix
    R = measurement_noise_factor * 1  # The measurement noise covariance matrix (in this case, just a scalar)
    P = np.zeros((3, 3, len(t)))  # The state covariance matrix, initialized as a 3D array of zeros
    P[:, :, 0] = initial_covariance*np.diag([1, 1, 1])  # Setting the initial covariance matrix
    x = np.zeros((3, len(t)))  # The state vector, initialized as a 2D array of zeros
    x[:, 0] = true_initial_state  # Setting the initial state vector based on the true initial states
    x[2, 0] = 0  # Setting the initial acceleration to zero

    P_p = np.zeros(len(t))  # The variance of the position
    P_v = np.zeros(len(t))  # The variance of the velocity
    P_a = np.zeros(len(t))  # The variance of the acceleration
    dT = np.mean(np.diff(t))  # The time step
    P_p[0] = P[0, 0, 0]  # Setting the initial variance of the position
    P_v[0] = P[1, 1, 0]  # Setting the initial variance of the velocity
    P_a[0] = P[2, 2, 0]  # Setting the initial variance of the acceleration

    # Call the Kalman filter function 
    x, P_p, P_v, P_a = kalman_filter(y_observed, x, P, F, H, G, Q, R, dT)

    # Save figures to PDF --> Uncomment to save original figures
    plt.figure()
    plt.plot(t, x[0, :], 'r', label='Kalman Filter Solution')
    plt.plot(t, y_observed, 'k', label='Noisy Observations')
    plt.plot(t, y_true, 'b', label='True Y')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('Position (m)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Position (m)', fontsize=14)
    plt.savefig(position_plot_name)

    plt.figure()
    plt.plot(t, x[2, :], 'r', label='KF Estimated Acceleration')
    plt.plot(t, true_initial_state[2] * np.ones(len(t)), 'b-.', label='True Acceleration')
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('Estimated Acceleration (m/s$^2$)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Acceleration (m/s$^2$)', fontsize=14)
    plt.savefig(acceleration_plot_name)

    plt.figure()
    plt.plot(t, P_p, 'r', label='Position Variance (m$^2$)')
    plt.plot(t, P_v, 'g', label='Velocity Variance (m$^2$/s$^2$)')
    plt.plot(t, P_a, 'b', label='Acceleration Variance (m$^2$/s$^4$)')
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlabel('Time (s)', fontsize=14)
    plt.title('State Error Covariance', fontsize=14)
    plt.savefig(state_error_covariance_plot_name)

    plt.show()

elif PART2_TASK1:
    for noise_std in observation_noise_std:

        # Set the initial parameters for the Kalman filter
        true_initial_state = np.array([8.7105, 38.8748, -9.7923])  # Initial state of the system
        F = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # The state transition matrix
        H = np.array([[1, 0, 0]])  # The observation matrix
        G = np.diag([0, 0, 1])  # The noise shaping matrix
        Q = system_noise_factor * np.diag([1, 1, 1])  # The system noise covariance matrix
        R = measurement_noise_factor * 1  # The measurement noise covariance matrix (in this case, just a scalar)
        P = np.zeros((3, 3, len(t)))  # The state covariance matrix, initialized as a 3D array of zeros
        P[:, :, 0] = initial_covariance*np.diag([1, 1, 1])  # Setting the initial covariance matrix
        x = np.zeros((3, len(t)))  # The state vector, initialized as a 2D array of zeros
        x[:, 0] = true_initial_state  # Setting the initial state vector based on the true initial states
        x[2, 0] = 0  # Setting the initial acceleration to zero

        P_p = np.zeros(len(t))  # The variance of the position
        P_v = np.zeros(len(t))  # The variance of the velocity
        P_a = np.zeros(len(t))  # The variance of the acceleration
        dT = np.mean(np.diff(t))  # The time step
        P_p[0] = P[0, 0, 0]  # Setting the initial variance of the position
        P_v[0] = P[1, 1, 0]  # Setting the initial variance of the velocity
        P_a[0] = P[2, 2, 0]  # Setting the initial variance of the acceleration

        y_observed = y_true + np.random.normal(0, noise_std, len(y_true))

        # Call the Kalman filter function
        x, P_p, P_v, P_a = kalman_filter(y_observed, x, P, F, H, G, Q, R, dT)

        # Generate the figures
        generate_figures(x, y_observed, y_true, t, plot_names_array, observation_noise_std, noise_std)

        # RMSE Calculation
        rmse = np.sqrt(np.sum((x[0, :] - y_true) ** 2) / len(y_true))
        rmse_noisy = np.sqrt(np.sum((y_observed - y_true) ** 2) / len(y_true))

        print(f'RMSE PART2_TASK1: {rmse:.2f}')
        print(f'RMSE PART2_TASK1 Noisy: {rmse_noisy:.2f}')

    plt.show()

elif PART2_TASK2:
    for sys_noise in system_noise_factor:

        # Set the initial parameters for the Kalman filter
        true_initial_state = np.array([8.7105, 38.8748, -9.7923])  # Initial state of the system
        F = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # The state transition matrix
        H = np.array([[1, 0, 0]])  # The observation matrix
        G = np.diag([0, 0, 1])  # The noise shaping matrix
        Q = sys_noise * np.diag([1, 1, 1])  # The system noise covariance matrix
        R = measurement_noise_factor * 1  # The measurement noise covariance matrix (in this case, just a scalar)
        P = np.zeros((3, 3, len(t)))  # The state covariance matrix, initialized as a 3D array of zeros
        P[:, :, 0] = initial_covariance*np.diag([1, 1, 1])  # Setting the initial covariance matrix
        x = np.zeros((3, len(t)))  # The state vector, initialized as a 2D array of zeros
        x[:, 0] = true_initial_state  # Setting the initial state vector based on the true initial states
        x[2, 0] = 0  # Setting the initial acceleration to zero

        P_p = np.zeros(len(t))  # The variance of the position
        P_v = np.zeros(len(t))  # The variance of the velocity
        P_a = np.zeros(len(t))  # The variance of the acceleration
        dT = np.mean(np.diff(t))  # The time step
        P_p[0] = P[0, 0, 0]  # Setting the initial variance of the position
        P_v[0] = P[1, 1, 0]  # Setting the initial variance of the velocity
        P_a[0] = P[2, 2, 0]  # Setting the initial variance of the acceleration

        y_observed = y_true + np.random.normal(0, observation_noise_std, len(y_true))

        # Call the Kalman filter function
        x, P_p, P_v, P_a = kalman_filter(y_observed, x, P, F, H, G, Q, R, dT)

        # Generate the figures
        generate_figures(x, y_observed, y_true, t, plot_names_array, system_noise_factor, sys_noise)

        # RMSE Calculation
        rmse = np.sqrt(np.sum((x[0, :] - y_true) ** 2) / len(y_true))
        rmse_noisy = np.sqrt(np.sum((y_observed - y_true) ** 2) / len(y_true))

        print(f'RMSE PART2_TASK2: {rmse:.2f}')
        print(f'RMSE PART2_TASK2 Noisy: {rmse_noisy:.2f}')

    plt.show()

elif PART2_TASK3:
    for mes_noise in measurement_noise_factor:

        # Set the initial parameters for the Kalman filter
        true_initial_state = np.array([8.7105, 38.8748, -9.7923])  # Initial state of the system
        F = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # The state transition matrix
        H = np.array([[1, 0, 0]])  # The observation matrix
        G = np.diag([0, 0, 1])  # The noise shaping matrix
        Q = system_noise_factor * np.diag([1, 1, 1])  # The system noise covariance matrix
        R = mes_noise * 1  # The measurement noise covariance matrix (in this case, just a scalar)
        P = np.zeros((3, 3, len(t)))  # The state covariance matrix, initialized as a 3D array of zeros
        P[:, :, 0] = initial_covariance*np.diag([1, 1, 1])  # Setting the initial covariance matrix
        x = np.zeros((3, len(t)))  # The state vector, initialized as a 2D array of zeros
        x[:, 0] = true_initial_state  # Setting the initial state vector based on the true initial states
        x[2, 0] = 0  # Setting the initial acceleration to zero

        P_p = np.zeros(len(t))  # The variance of the position
        P_v = np.zeros(len(t))  # The variance of the velocity
        P_a = np.zeros(len(t))  # The variance of the acceleration
        dT = np.mean(np.diff(t))  # The time step
        P_p[0] = P[0, 0, 0]  # Setting the initial variance of the position
        P_v[0] = P[1, 1, 0]  # Setting the initial variance of the velocity
        P_a[0] = P[2, 2, 0]  # Setting the initial variance of the acceleration

        y_observed = y_true + np.random.normal(0, observation_noise_std, len(y_true))

        # Call the Kalman filter function
        x, P_p, P_v, P_a = kalman_filter(y_observed, x, P, F, H, G, Q, R, dT)

        # Generate the figures
        generate_figures(x, y_observed, y_true, t, plot_names_array, measurement_noise_factor, mes_noise)

        # RMSE Calculation
        rmse = np.sqrt(np.sum((x[0, :] - y_true) ** 2) / len(y_true))
        rmse_noisy = np.sqrt(np.sum((y_observed - y_true) ** 2) / len(y_true))

        print(f'RMSE PART2_TASK3: {rmse:.2f}')
        print(f'RMSE PART2_TASK3 Noisy: {rmse_noisy:.2f}')

    plt.show()

elif PART2_TASK4:
    for initial_cov in initial_covariance:

        # Set the initial parameters for the Kalman filter
        true_initial_state = np.array([8.7105, 38.8748, -9.7923])  # Initial state of the system
        F = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # The state transition matrix
        H = np.array([[1, 0, 0]])  # The observation matrix
        G = np.diag([0, 0, 1])  # The noise shaping matrix
        Q = system_noise_factor * np.diag([1, 1, 1])  # The system noise covariance matrix
        R = measurement_noise_factor * 1  # The measurement noise covariance matrix (in this case, just a scalar)
        P = np.zeros((3, 3, len(t)))  # The state covariance matrix, initialized as a 3D array of zeros
        P[:, :, 0] = initial_cov*np.diag([1, 1, 1])  # Setting the initial covariance matrix
        x = np.zeros((3, len(t)))  # The state vector, initialized as a 2D array of zeros
        x[:, 0] = true_initial_state  # Setting the initial state vector based on the true initial states
        x[2, 0] = 0  # Setting the initial acceleration to zero

        P_p = np.zeros(len(t))  # The variance of the position
        P_v = np.zeros(len(t))  # The variance of the velocity
        P_a = np.zeros(len(t))  # The variance of the acceleration
        dT = np.mean(np.diff(t))  # The time step
        P_p[0] = P[0, 0, 0]  # Setting the initial variance of the position
        P_v[0] = P[1, 1, 0]  # Setting the initial variance of the velocity
        P_a[0] = P[2, 2, 0]  # Setting the initial variance of the acceleration

        y_observed = y_true + np.random.normal(0, observation_noise_std, len(y_true))

        # Call the Kalman filter function
        x, P_p, P_v, P_a = kalman_filter(y_observed, x, P, F, H, G, Q, R, dT)

        # Generate the figures
        generate_figures(x, y_observed, y_true, t, plot_names_array, initial_covariance, initial_cov)

        # RMSE Calculation
        rmse = np.sqrt(np.sum((x[0, :] - y_true) ** 2) / len(y_true))
        rmse_noisy = np.sqrt(np.sum((y_observed - y_true) ** 2) / len(y_true))

        print(f'RMSE PART2_TASK4: {rmse:.2f}')
        print(f'RMSE PART2_TASK4 Noisy: {rmse_noisy:.2f}')

    plt.show()

