# All code within this script was written by Alexander Crain for the course SYSC5702. Use of this code is freely allowed with no limitations whatsoever.

#==============================================================================#
# PREAMBLE
#==============================================================================#

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from project_4_utils import import_project_data

#==============================================================================#
# DATA LOADING AND INITIALIZATION
#==============================================================================#

t, f_theta_noisy, f_a_noisy, East_GPS, North_GPS, Azimuth_Magnetometer, East_ref, North_ref, Azimuth_ref, initial_accel, initial_forward_speed = import_project_data(filepath='proj4_data.mat')

# Define parameters
system_noise_factor = 0.001  # Used to scale Q matrix
measurement_noise_factor = 1  # Used to scale R matrix
initial_covariance = 5  # Initial covariance value

# Calculate the time step for the imported data
dT = np.mean(np.diff(t))

# Set the number of states
n_states = 7

# Set the controls
SAVE_TASK2_PLOTS = True
SAVE_TASK5_PLOTS = True

#==============================================================================#
# TASK 1: DERIVE SYSTEM MODEL
#==============================================================================#

# The main difference between this projects model and the on in project #1 is the inclusion of the Azimuth as a state which can vary over time. In the previous project, we assumed that the Azimuth was a constant value, which meant the system was linear. When the Azimuth varies, the model contains sin and cos which are nonlinear.

# The state vector for this project is:

# x = [ E, N, V, a, theta, b_a, b_theta]

# And the controls for this system are defined as:

# u = [f_a, f_theta]

# However, these are measurements from imperfect sensors with additive random constant biases and zero-mean Gausian random noise. So in actuality:

# u = [f_a_noisy, f_theta_noisy]

# Where:

# f_a_noisy = f_a + b_a -- where b_a is a constant bias
# f_theta_noisy = f_theta + b_theta -- where b_theta is a constant bias

# So the system dynamics can be defined as:

# x_dot = [E_dot, N_dot, V_dot, a_dot, theta_dot, b_a_dot, b_theta_dot]

# And:

# E_dot = V*cos(theta)
# N_dot = V*sin(theta)
# V_dot = a
# a_dot = f_a
# theta_dot = f_theta
# b_a_dot = 0 --> Bias is constant
# b_theta_dot = 0 --> Bias is constant

# Let's now define a function for the system model which will calculate the state derivatives given the noisy control inputs
def system_dynamics(x, u):
    """
    This function calculates the state derivatives given the current state and the noisy control inputs. The state derivatives are calculated according to the model dynamics defined above. The noisy control inputs are first compensated for the biases before being used in the calculation of the state derivatives.

    Parameters:
    x (array_like): The current state vector [E, N, V, a, theta, b_a, b_theta]
    u (array_like): The noisy control inputs [f_a_noisy, f_theta_noisy]

    Returns:
    array_like: The state derivatives [E_dot, N_dot, V_dot, a_dot, theta_dot, b_a_dot, b_theta_dot]
    """   

    # First we extract the states
    E, N, V, a, theta, b_a, b_theta = x

    # Then we extract the control inputs
    f_a_noisy, f_theta_noisy = u

    # We have to calculate the true control inputs (compensating for biases)
    f_a = f_a_noisy - b_a
    f_theta = f_theta_noisy - b_theta

    # Now we calculate the state derivatives
    E_dot = V * np.cos(theta)
    N_dot = V * np.sin(theta)
    V_dot = a
    a_dot = f_a
    theta_dot = f_theta
    b_a_dot = 0
    b_theta_dot = 0

    # And finally we return the state derivatives
    return np.array([E_dot, N_dot, V_dot, a_dot, theta_dot, b_a_dot, b_theta_dot])

#==============================================================================#
# TASK 2: OPEN-LOOP MODEL PREDICTION
#==============================================================================#

# With the dynamics model defined, we can use it to calculate the open-loop predictions for the states -- noting that because of the noise and biases present in the control measurements, we would expect this open loop prediction to be very different from the true states.

# Let's now define a function for the open-loop prediction of the states given the noisy control inputs. for the sake of demonstrating the inpact of the noise we will use the measurements as initial conditions.

E = East_GPS[0]
N = North_GPS[0]
V = initial_forward_speed
a = initial_accel
theta = Azimuth_Magnetometer[0]
b_a = 0
b_theta = 0

# Set the initial states
x_init = np.array([E, N, V, a, theta, b_a, b_theta])

def open_loop_prediction(x_init, u):
    """
    This function performs an open-loop prediction of the states given the noisy control inputs.

    Parameters:
    x_init (array_like): The initial state vector [E, N, V, a, theta, b_a, b_theta]
    u (array_like): The control inputs [f_a_noisy, f_theta_noisy]

    Returns:
    array_like: The open-loop predictions of the states [E, N, V, a, theta, b_a, b_theta]
    """

    # We should also define storage for the results
    x_open_loop = np.zeros((n_states, len(t)))

    # And then replace the initial values:
    x_open_loop[:, 0] = x_init

    for i in range(1, len(t)):

        # Calculate the state derivative at this time step
        x_dot = system_dynamics(x_open_loop[:, i-1], u[:, i-1])

        # Integrate the state derivative to get the state at the next time step
        x_open_loop[:, i] = x_open_loop[:, i-1] + x_dot * dT

    return x_open_loop

# The control input is defined as:
u = np.vstack((f_a_noisy, f_theta_noisy))

# Run the open-loop prediction
x_open_loop = open_loop_prediction(x_init, u)

# Now we plot the results using matplotlib
fontsize = 30

# Show the plots
if SAVE_TASK2_PLOTS:
    plt.figure(figsize=(12, 10))
    plt.plot(t, East_GPS, '-k', markersize=2, label='GPS East Measurement', linewidth=2)
    plt.plot(t, x_open_loop[0, :], '--r', label='Open-Loop East Position', linewidth=2)
    plt.xlabel('Time [s]', fontsize=fontsize)
    plt.ylabel('East Position [m]', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout(pad=1.5)  # Adds extra padding around the plot
    plt.savefig('task2_open_loop_east.pdf')

    plt.figure(figsize=(12, 10))
    plt.plot(t, North_GPS, '-k', markersize=2, label='GPS North Measurement', linewidth=2)
    plt.plot(t, x_open_loop[1, :], '--r', label='Open-Loop North Position', linewidth=2)
    plt.xlabel('Time [s]', fontsize=fontsize)
    plt.ylabel('North Position [m]', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout(pad=1.5)  # Adds extra padding around the plot
    plt.savefig('task2_open_loop_north.pdf')

    plt.figure(figsize=(12, 10))
    plt.plot(t, Azimuth_Magnetometer, '-k', markersize=2, label='Magnetometer Measurement', linewidth=2)
    plt.plot(t, x_open_loop[4, :], '--r',label='Open-Loop Azimuth', linewidth=2)
    plt.xlabel('Time [s]', fontsize=fontsize)
    plt.ylabel('Azimuth [rad]', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout(pad=1.5)  # Adds extra padding around the plot
    plt.savefig('task2_open_loop_azimuth.pdf')


# We can also calculate the RMSE to gauge how poorly the results are, since the plots are very difficult to interpret.

# Calculate RMSE
rmse_E_open = np.sqrt(np.mean((x_open_loop[0, :] - East_GPS) ** 2))
print('Open-Loop RMSE for East position:', rmse_E_open)

rmse_N_open = np.sqrt(np.mean((x_open_loop[1, :] - North_GPS) ** 2))
print('Open-Loop RMSE for North position:', rmse_N_open)

rmse_theta_open = np.sqrt(np.mean((x_open_loop[4, :] - Azimuth_Magnetometer) ** 2))
print('Open-Loop RMSE for Azimuth:', rmse_theta_open)

#==============================================================================#
# TASK 3: Linearize the System Model
#==============================================================================#

# Given that we have a nonlinear system of equations, we can't simply use the state transition matrix F directly. Instead we need to linearize the dynamics model first. To do this we can calculate the Jacobian matrix for our dynamics. This matrix is the partial derivative of each state w.r.t. all other states.

def linearized_system_model(x):
    """
    Calculate the linearized system model.

    Parameters
    ----------
    x : 1D array of length 7
        The state vector [East, North, Velocity, acceleration, theta, bias acceleration, bias theta]

    Returns
    -------
    F : 2D array of size 7x7
        The state transition matrix
    G : 2D array of size 7x7
        The noise shaping matrix
    """
    # F = ∂f(x, u)/∂x
    # where f(x, u) is the system dynamics function
    # F[0,0] = ∂E_dot/∂E = 0
    # F[0,1] = ∂E_dot/∂N = 0
    # F[0,2] = ∂E_dot/∂V = cos(theta)
    # F[0,3] = ∂E_dot/∂a = 0
    # F[0,4] = ∂E_dot/∂theta = -V * sin(theta)
    # F[0,5] = ∂E_dot/∂b_a = 0
    # F[0,6] = ∂E_dot/∂b_theta = 0
    # F[1,0] = ∂N_dot/∂E = 0
    # F[1,1] = ∂N_dot/∂N = 0
    # F[1,2] = ∂N_dot/∂V = sin(theta)
    # F[1,3] = ∂N_dot/∂a = 0
    # F[1,4] = ∂N_dot/∂theta = V * cos(theta)
    # F[1,5] = ∂N_dot/∂b_a = 0
    # F[1,6] = ∂N_dot/∂b_theta = 0
    # F[2,0] = ∂V_dot/∂E = 0
    # F[2,1] = ∂V_dot/∂N = 0
    # F[2,2] = ∂V_dot/∂V = 0
    # F[2,3] = ∂V_dot/∂a = 1
    # F[2,4] = ∂V_dot/∂theta = 0
    # F[2,5] = ∂V_dot/∂b_a = 0
    # F[2,6] = ∂V_dot/∂b_theta = 0
    # F[3,0] = ∂a_dot/∂E = 0
    # F[3,1] = ∂a_dot/∂N = 0
    # F[3,2] = ∂a_dot/∂V = 0
    # F[3,3] = ∂a_dot/∂a = 0
    # F[3,4] = ∂a_dot/∂theta = 0
    # F[3,5] = ∂a_dot/∂b_a = -1
    # F[3,6] = ∂a_dot/∂b_theta = 0
    # F[4,0] = ∂theta_dot/∂E = 0
    # F[4,1] = ∂theta_dot/∂N = 0
    # F[4,2] = ∂theta_dot/∂V = 0
    # F[4,3] = ∂theta_dot/∂a = 0
    # F[4,4] = ∂theta_dot/∂theta = 0
    # F[4,5] = ∂theta_dot/∂b_a = 0
    # F[4,6] = ∂theta_dot/∂b_theta = -1
    # F[5,0] = ∂b_a_dot/∂E = 0
    # F[5,1] = ∂b_a_dot/∂N = 0
    # F[5,2] = ∂b_a_dot/∂V = 0
    # F[5,3] = ∂b_a_dot/∂a = 0
    # F[5,4] = ∂b_a_dot/∂theta = 0
    # F[5,5] = ∂b_a_dot/∂b_a = 0
    # F[5,6] = ∂b_a_dot/∂b_theta = 0
    # F[6,0] = ∂b_theta_dot/∂E = 0
    # F[6,1] = ∂b_theta_dot/∂N = 0
    # F[6,2] = ∂b_theta_dot/∂V = 0
    # F[6,3] = ∂b_theta_dot/∂a = 0
    # F[6,4] = ∂b_theta_dot/∂theta = 0
    # F[6,5] = ∂b_theta_dot/∂b_a = 0
    # F[6,6] = ∂b_theta_dot/∂b_theta = 0

    # Extract states
    E, N, V, a, theta, b_a, b_theta = x

    # So, based on the above we can define F as:
    F = np.zeros((7, 7))
    F[0, 2] = np.cos(theta)
    F[0, 4] = -V * np.sin(theta)
    F[1, 2] = np.sin(theta)
    F[1, 4] = V * np.cos(theta)
    F[2, 3] = 1
    F[3, 5] = -1
    F[4, 6] = -1

    # We also need to calculate the noise shaping matrix G
    G = np.eye(7)
    
    return F, G
    
#==============================================================================#
# TASK 4: OBSERVATION MATRIX
#==============================================================================#
# Now we need the observation matrix H
# The observation matrix maps state to measurement
# For observations [East_GPS, North_GPS, Azimuth_Magnetometer]

def calculate_H():
    """
    Calculate the observation matrix H for the system. The observation matrix maps state to measurement.

    For observations [East_GPS, North_GPS, Azimuth_Magnetometer], the observation matrix is:
    H = [[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0]]

    Returns:
    array_like: The observation matrix H
    """
    H = np.zeros((3, 7))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 4] = 1
    return H

#==============================================================================#
# TASK 5: EKF CODE
#==============================================================================#
# We can move on to implementing the full EKF code. First, we need to define the system noise and measurement noise covariance matrices

# Process noise covariance
Q = np.diag([0.1, 0.1, 1, 0.1, 0.001, 0.001, 0.001])

# Measurement noise covariance
R = np.diag([0.1, 0.1, 0.001]) 

# Now we can define the EKF function

def ekf(x, u, P, z, Q, R):
    """
    Implement the Extended Kalman Filter (EKF) for the system.

    Args:
    x (array_like): The current state estimate.
    u (array_like): The current control input.
    P (array_like): The current covariance matrix.
    z (array_like): The current measurement.
    Q (array_like): The process noise covariance matrix.
    R (array_like): The measurement noise covariance matrix.


    Returns:
    x (array_like): The updated state estimate.
    P (array_like): The updated covariance matrix.
    """

    # Get linearized system matrices
    F, G = linearized_system_model(x)
    
    # Discretize the continuous-time model
    Phi = np.eye(7) + F * dT
    Qd = dT**2 * G @ Q @ G.T
    
    # EKF Prediction Step
    x_dot = system_dynamics(x, u)
    x_predicted = x + x_dot * dT
    P_predicted = Phi @ P @ Phi.T + Qd

    # EKF Update Step
    # Get the observation matrix H
    H = calculate_H()
    
    # Compute the predicted measurement
    h_x = np.array([x_predicted[0], x_predicted[1], x_predicted[4]])
    
    # Compute the measurement residual
    y = z - h_x
    
    # Compute the covariance
    S = H @ P_predicted @ H.T + R
    
    # Compute the Kalman gain
    K = P_predicted @ H.T @ np.linalg.inv(S)
    
    # Update the state estimate
    x_updated = x_predicted + K @ y
    
    # Update the covariance matrix
    P_updated = (np.eye(7) - K @ H) @ P_predicted
    
    return x_updated, P_updated

# With the EKF implemented, we can now run the simulation

# Initialize state vector for EKF
x_ekf = np.zeros((7, len(t)))
x_ekf[:, 0] = x_open_loop[:, 0]  # Same initial conditions as open-loop

# Initialize covariance matrix
P = np.ones((7, 7, len(t)))
P[:, :, 0] = np.eye(7)*initial_covariance  # Initial covariance matrix
P[5, 5, 0] = 100  # Higher uncertainty for acceleration bias
P[6, 6, 0] = 100  # Higher uncertainty for steering bias

# Run the EKF
for i in range(len(t) - 1):

    # Get the noisy measurements
    z = np.array([East_GPS[i+1], North_GPS[i+1], Azimuth_Magnetometer[i+1]])

    # Get the noisy inputs
    u = np.array([f_a_noisy[i], f_theta_noisy[i]])

    # Run the EKF
    x_ekf[:, i+1], P[:, :, i+1] = ekf(x_ekf[:, i], u, P[:, :, i], z, Q, R)

if SAVE_TASK5_PLOTS:
    # Now we can actually plot the results to see the impact. Let's start with Azimuth.
    # Plot Azimuth
    plt.figure(figsize=(12, 10))
    plt.plot(t, Azimuth_Magnetometer, '-k', label='Azimuth Magnetometer', linewidth=2)
    plt.plot(t, x_ekf[4, :], '--r', label='Azimuth EKF', linewidth=2)
    plt.xlabel('Time [s]', fontsize=fontsize)
    plt.ylabel('Azimuth [rad]', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid(True)
    plt.tight_layout(pad=1.5)  # Adds extra padding around the plot
    plt.savefig('task5_azimuth_ekf.pdf')
    #plt.show()

    # Now let's plot the position
    # Plot Position
    plt.figure(figsize=(12, 10))
    plt.plot(t, East_GPS, '-k',label='East GPS', linewidth=2)
    plt.plot(t, x_ekf[0, :], '--r', label='East EKF', linewidth=2)
    plt.plot(t, North_GPS, 'b', label='North GPS', linewidth=2)
    plt.plot(t, x_ekf[1, :], '--m', label='North EKF', linewidth=2)
    plt.xlabel('Time [s]', fontsize=fontsize)
    plt.ylabel('Position [m]', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout(pad=1.5)  # Adds extra padding around the plot
    plt.savefig('task5_position_ekf.pdf')

    # Now let's plot the biases
    # Plot Biases
    plt.figure(figsize=(12, 10))
    plt.plot(t, x_ekf[5, :], 'k', label='Acceleration Bias', linewidth=2)
    plt.plot(t, x_ekf[6, :], 'r',label='Steering Bias', linewidth=2)
    plt.xlabel('Time [s]', fontsize=fontsize)
    plt.ylabel('Bias [m]', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout(pad=1.5)  # Adds extra padding around the plot
    plt.savefig('task5_biases_ekf.pdf')

    # Add a plot for the covariance matrix
    plt.figure(figsize=(12, 10))
    plt.plot(t, P[0, 0, :], 'k', label='East Position Covariance', linewidth=2)
    plt.plot(t, P[1, 1, :], 'r', label='North Position Covariance', linewidth=2)
    plt.plot(t, P[4, 4, :], 'b', label='Azimuth Covariance', linewidth=2)
    plt.axis([0, 0.1, -1, 10])
    plt.xlabel('Time [s]', fontsize=fontsize)
    plt.ylabel('Covariance [m^2]', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout(pad=1.5)  # Adds extra padding around the plot
    plt.savefig('task5_covariance_ekf.pdf')

# Finally, we can calculate and display the RMSE for the EKF results

# Calculate RMSE   
RMSE_EKF_EAST = np.sqrt(np.mean((x_ekf[0, :] - East_GPS) ** 2))
RMSE_EKF_NORTH = np.sqrt(np.mean((x_ekf[1, :] - North_GPS) ** 2))
RMSE_EKF_AZIMUTH = np.sqrt(np.mean((x_ekf[4, :] - Azimuth_Magnetometer) ** 2))

# Print RMSE
print('RMSE for East position:', RMSE_EKF_EAST)
print('RMSE for North position:', RMSE_EKF_NORTH)
print('RMSE for Azimuth:', RMSE_EKF_AZIMUTH)

# Print the steady state bias values
print('Acceleration Bias:', x_ekf[5, -1])
print('Steering Bias:', x_ekf[6, -1])


