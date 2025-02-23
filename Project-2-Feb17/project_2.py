#==============================================================================
# PREAMBLE
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from project_utils import convert_dcm_to_quat, euler_to_dcm

#==============================================================================
# LOAD DATA
#==============================================================================

# Load the MATLAB file (ensure 'trajecory1_data.mat' is in your working directory)
mat_data = loadmat('trajecory1_data.mat', squeeze_me=True, struct_as_record=False)
trajectory_info = mat_data['trajectory_info']

# Extract the data from the matlab file
time = trajectory_info.t  # Time (s)
lat = trajectory_info.lat  # Latitude (rad)
lon = trajectory_info.lon  # Longitude (rad)
h = trajectory_info.h  # Altitude (m)
vn = trajectory_info.vn  # Velocity North (m/s)
ve = trajectory_info.ve  # Velocity East (m/s)
vd = trajectory_info.vd  # Velocity Down (m/s)
roll = trajectory_info.roll  # Roll (rad)
pitch = trajectory_info.pitch  # Pitch (rad)
heading = trajectory_info.heading  # Heading (rad)
gyro = trajectory_info.gyro  # Angular rate (rad/s)
accel = trajectory_info.accel  # Acceleration (m/s^2)

# Change this to run the code using the second trajectory
DATASET1 = False
DATASET2 = True

# Load the second trajectory data and override the old data
if DATASET2 and not DATASET1:
    # Load the MATLAB file (ensure 'trajecory1_data.mat' is in your working directory)
    mat_data = loadmat('trajecory2_data.mat', squeeze_me=True, struct_as_record=False)
    trajectory_info = mat_data['trajectory_info']

    # Extract the data from the matlab file
    time = trajectory_info.t  # Time (s)
    lat = trajectory_info.lat  # Latitude (rad)
    lon = trajectory_info.lon  # Longitude (rad)
    h = trajectory_info.h  # Altitude (m)
    vn = trajectory_info.vn  # Velocity North (m/s)
    ve = trajectory_info.ve  # Velocity East (m/s)
    vd = trajectory_info.vd  # Velocity Down (m/s)
    roll = trajectory_info.roll  # Roll (rad)
    pitch = trajectory_info.pitch  # Pitch (rad)
    heading = trajectory_info.heading  # Heading (rad)
    gyro = trajectory_info.gyro  # Angular rate (rad/s)
    accel = trajectory_info.accel  # Acceleration (m/s^2)

#==============================================================================
# CONSTANTS
#==============================================================================

# Define a random seed for reproducibility
np.random.seed(64518)

# Define any constants needed for the analysis
D2R = np.pi / 180  # Degrees to radians

# Earth parameters
semimajor = 6378137                # Semi Major Axis of the Earth (in metres)
semiminor = 6356752.3142           # Semi Minor Axis of the Earth (in metres)
e2 = 1 - (semiminor**2) / (semimajor**2)     # Used in calculating radii of curvature (N & M)
ecc = np.sqrt((semimajor**2 - semiminor**2) / semimajor**2)
earth_rate = 7.292115e-5  # Earth rotation rate (rad/s)
g_L = np.array([0, 0, 9.793])  # Gravity in the L frame (m/s²)

# Compute the radii of curvature in the prime vertical (RN) and meridian (RM)
RN = semimajor / np.sqrt(1 - e2 * np.sin(lat)**2)
RM = semimajor * (1 - e2) / ((1 - e2 * np.sin(lat)**2)**1.5)

# Control which plots are displayed
NOISE_ACCEL_X = False
NOISE_ACCEL_Y = False
NOISE_ACCEL_Z = False
NOISE_GYRO_X = False
NOISE_GYRO_Y = False
NOISE_GYRO_Z = False

# Define the empty noise and bias arrays
accel_noise = np.zeros((len(time), 3))
accel_bias = np.array([0.0, 0.0, 0.0])
accel_scale = np.array([0.0, 0.0, 0.0])

gyro_noise = np.zeros((len(time), 3))
gyro_bias = np.array([0.0, 0.0, 0.0])
gyro_scale = np.array([0.0, 0.0, 0.0])

# Depending on which noise and bias is enabled, add to the acceleration data
if NOISE_ACCEL_X:
    accel_noise[:,0] = np.random.normal(0, 0.01, len(time))
    accel_bias[0] = 0.1
    accel_scale[0] = 0.0

if NOISE_ACCEL_Y:
    accel_noise[:,1] = np.random.normal(0, 0.01, len(time))
    accel_bias[1] = 0.1
    accel_scale[1] = 0.0

if NOISE_ACCEL_Z:
    accel_noise[:,2] = np.random.normal(0, 0.1, len(time))
    accel_bias[2] = 0.1
    accel_scale[2] = 0.0

# Depending on which noise and bias is enabled, add to the gyroscope data
if NOISE_GYRO_X:
    gyro_noise[:,0] = np.random.normal(0, 0.001, len(time))
    gyro_bias[0] = 0.01
    gyro_scale[0] = 0.00

if NOISE_GYRO_Y:
    gyro_noise[:,1] = np.random.normal(0, 0.001, len(time))
    gyro_bias[1] = 0.01
    gyro_scale[1] = 0.00

if NOISE_GYRO_Z:
    gyro_noise[:,2] = np.random.normal(0, 0.001, len(time))
    gyro_bias[2] = 0.01
    gyro_scale[2] = 0.00


#==============================================================================
# CALCULATIONS - INVERSE KINEMATICS
#==============================================================================

# Using the position, velocity, and attitude data, compute the inverse kinematics
# to determine the gyroscope and acceleration

# Given the velocity in the L frame (ECI), we can can calculate the acceleration in the L frame
# using the numerical derivative of the velocity
v_L = np.array([vn, ve, vd])

# Calculate acceleration using central differences (except endpoints)
a_Lx = np.zeros(len(time))
a_Ly = np.zeros(len(time))
a_Lz = np.zeros(len(time))

for i in range(1, len(time) - 1):
    dt_prev = time[i] - time[i-1]
    dt_next = time[i+1] - time[i]
    a_Lx[i] = (vn[i+1] - vn[i-1]) / (dt_prev + dt_next)
    a_Ly[i] = (ve[i+1] - ve[i-1]) / (dt_prev + dt_next)
    a_Lz[i] = (vd[i+1] - vd[i-1]) / (dt_prev + dt_next)

# Forward difference for the first element
a_Lx[0] = (vn[1] - vn[0]) / (time[1] - time[0])
a_Ly[0] = (ve[1] - ve[0]) / (time[1] - time[0])
a_Lz[0] = (vd[1] - vd[0]) / (time[1] - time[0])

# Backward difference for the last element
a_Lx[-1] = (vn[-1] - vn[-2]) / (time[-1] - time[-2])
a_Ly[-1] = (ve[-1] - ve[-2]) / (time[-1] - time[-2])
a_Lz[-1] = (vd[-1] - vd[-2]) / (time[-1] - time[-2])

# To get the final acceleration, we must compensate for gravity and the coriolis effect
a_L_corrected = np.zeros((3, len(time)))
for i in range(0, len(time)):

    omega_EL = np.array([ ve[i]/(RN[i] + h[i]), 
                         -vn[i]/(RM[i] + h[i]), 
                         -ve[i]*np.tan(lat[i])/(RN[i] + h[i])])
    omega_IE = earth_rate * np.array([np.cos(lat[i]), 
                                      0, 
                                      -np.sin(lat[i])])
    a_L = np.array([a_Lx[i], a_Ly[i], a_Lz[i]])
    a_L_corrected[:,i] = a_L - g_L + np.cross((omega_EL + 2*omega_IE),v_L[:,i])

# Calculate the direction cosine matrix (DCM) from the Euler angles
dcm = np.zeros((3, 3, len(time)))
for i in range(0, len(time)):
    dcm[:, :, i] = euler_to_dcm(roll[i], pitch[i], heading[i])

# Using the DCM, project the L frame accelerations into the body frame
a_B = np.zeros((len(time),3))
for i in range(0, len(time)):
    a_B[i,:] = dcm[:, :, i].T @ a_L_corrected[:,i]

# The raw gyroscope measurements can be calculated by taking the rotation difference between two consecutive DCMs and dividing by the time step
# Correct gyro calculation using actual time step
gyro_B = np.zeros((len(time), 3))
for i in range(1, len(time)):
    dt = time[i] - time[i-1]
    dcm_diff = (dcm[:, :, i-1].T @ dcm[:, :, i] - np.eye(3)) / dt
    gyro_B[i, :] = [dcm_diff[2, 1], dcm_diff[0, 2], dcm_diff[1, 0]]

# To get the correct gyro measurements, we must compensate for the earth's rate and the transport rate
gyro_corrected = np.zeros((len(time),3))
for i in range(0, len(time)):
    omega_IE = earth_rate * np.array([np.cos(lat[i]), 
                                      0, 
                                      -np.sin(lat[i])])
    omega_EL = np.array([ve[i]/(RN[i] + h[i]), 
                         -vn[i]/(RM[i] + h[i]), 
                         -ve[i]*np.tan(lat[i])/(RN[i] + h[i])])
    gyro_corrected[i,:] = gyro_B[i,:] + dcm[:, :, i].T @ (omega_EL + omega_IE)

# Print the individual RMSEs between the estimations and the datasets
accel_rmse = np.sqrt(np.mean((accel - a_B)**2))
gyro_rmse = np.sqrt(np.mean((gyro - gyro_corrected)**2))

print(f'Accelerometer RMSE: {accel_rmse}')
print(f'Gyroscope RMSE: {gyro_rmse}')

#==============================================================================
# CALCULATIONS - DYNAMICS
# CALCULATIONS - NOISE, BIAS, AND SCALE
#==============================================================================
# Now, we want to do the reverse and calculate the position, velocity, and orientation
# using the gyroscope and accelerometer results from the previous section

# In this section we also want to add noise and bias to the accelerometer and gyroscope data
# The instructions are unclear about any scale factors for task #3, so I will include them 
# in the calculations as an exercise in understanding their effect

# Initialize the latitude, longitude, and altitude arrays
lat_est = np.zeros(len(time))
lon_est = np.zeros(len(time))
h_est = np.zeros(len(time))

# Initialize the velocity arrays
vn_est = np.zeros(len(time))
ve_est = np.zeros(len(time))
vd_est = np.zeros(len(time))

# Initialize the roll, pitch, and heading arrays
roll_est = np.zeros(len(time))
pitch_est = np.zeros(len(time))
heading_est = np.zeros(len(time))

# Initialize RN and RM
RN_est = np.zeros(len(time))
RM_est = np.zeros(len(time))

# Initialize the acceleration arrays
accel_est = np.zeros((len(time),3))

# Initialize the gyro arrays
gyro_est = np.zeros((len(time),3))

# Initialize the velocity array
v_L_est = np.zeros((len(time),3))

# Initialize the quaternion arrays
q_LB = np.zeros((len(time),4))
q_LB_est = np.zeros((len(time),4))

# Set the initial value in each array
lat_est[0] = lat[0]
lon_est[0] = lon[0]
h_est[0] = h[0]
vn_est[0] = vn[0]
ve_est[0] = ve[0]
vd_est[0] = vd[0]
roll_est[0] = roll[0]
pitch_est[0] = pitch[0]
heading_est[0] = heading[0]
accel_est[0, :] = a_B[0, :]  
gyro_est[0,:] = gyro_corrected[0,:]
v_L_est[0,:] = np.array([vn_est[0], ve_est[0], vd_est[0]])

# Initialize the DCM array
dcm_est = np.zeros((3, 3, len(time)))
dcm_est[:, :, 0] = euler_to_dcm(roll_est[0], pitch_est[0], heading_est[0])

# Create the true quaternions for comparisons
for i in range(0, len(time)):
    q_LB[i,:] = convert_dcm_to_quat(dcm[:, :, i])

# Start a loop to calculate the position, velocity, and orientation
for i in range(0, len(time)-1):

    # Calculate the time difference
    if i == 0:
        dt = time[1] - time[0]
    else:
        dt = time[i] - time[i-1]

    # Update the radii
    RN_est[i] = semimajor / np.sqrt(1 - e2 * np.sin(lat_est[i])**2)
    RM_est[i] = semimajor * (1 - e2) / ((1 - e2 * np.sin(lat_est[i])**2)**1.5)

    # Get the quaternion from the DCM
    q_LB_est[i,:] = convert_dcm_to_quat(dcm_est[:, :, i])

    # Correct gyro compensation with DCM transformation
    omega_IE = earth_rate * np.array([np.cos(lat_est[i]), 0, -np.sin(lat_est[i])])
    omega_EL = np.array([
        ve_est[i] / (RN_est[i] + h_est[i]),
        -vn_est[i] / (RM_est[i] + h_est[i]),
        -ve_est[i] * np.tan(lat_est[i]) / (RN_est[i] + h_est[i])
    ])
    omega_IL = omega_IE + omega_EL  # This is in the L frame

    # Subtract omega_IL transformed to body frame and consider the noise, bias, and scale
    gyro_est[i, :] = (gyro_corrected[i, :] - gyro_bias)/(1 + gyro_scale) - dcm_est[:, :, i].T @ omega_IL + gyro_noise[i, :]

    # Subtract the bias and scale and add the noise
    accel_est[i+1, :] = ((a_B[i+1, :]) - accel_bias) / (1 + accel_scale) + accel_noise[i+1, :]

    # Calculate q_LB_est_dot using the gyro calculations from part 1
    omega_LB_cross = np.array([[0, -gyro_est[i, 0], -gyro_est[i, 1], -gyro_est[i, 2]],
                            [gyro_est[i, 0], 0, gyro_est[i, 2], -gyro_est[i, 1]],
                            [gyro_est[i, 1], -gyro_est[i, 2], 0, gyro_est[i, 0]],
                            [gyro_est[i, 2], gyro_est[i, 1], -gyro_est[i, 0], 0]])

    # Calculate the quaternion derivative
    q_LB_est_dot = 0.5 * omega_LB_cross @ q_LB_est[i,:].T
    
    # Numerically integrate to get the next quaternion
    q_LB_est[i+1,:] = q_LB_est[i,:] + q_LB_est_dot * dt

    # Normalize the quaternion 
    q_LB_est[i+1,:] = q_LB_est[i+1,:] / np.sqrt(q_LB_est[i+1,0]**2 + q_LB_est[i+1,1]**2 + q_LB_est[i+1,2]**2 + q_LB_est[i+1,3]**2)
    
    # Extract a, b, c, and d from the quaternion
    a, b, c, d = q_LB_est[i+1,:]

    # Convert the quaternion to a DCM
    dcm_est[0, 0, i+1] = a**2 + b**2 - c**2 - d**2
    dcm_est[0, 1, i+1] = 2 * (b * c - a * d)
    dcm_est[0, 2, i+1] = 2 * (b * d + a * c)
    dcm_est[1, 0, i+1] = 2 * (b * c + a * d)
    dcm_est[1, 1, i+1] = a**2 - b**2 + c**2 - d**2
    dcm_est[1, 2, i+1] = 2 * (c * d - a * b)
    dcm_est[2, 0, i+1] = 2 * (b * d - a * c)
    dcm_est[2, 1, i+1] = 2 * (c * d + a * b)
    dcm_est[2, 2, i+1] = a**2 - b**2 - c**2 + d**2

    # Use estimated velocities for Coriolis calculation
    v_L_dot = (dcm_est[:, :, i] @ accel_est[i, :]) + g_L - np.cross((omega_EL + 2 * omega_IE), v_L_est[i,:])

    # Integrate using estimated velocity
    vn_est[i+1] = vn_est[i] + v_L_dot[0] * dt
    ve_est[i+1] = ve_est[i] + v_L_dot[1] * dt
    vd_est[i+1] = vd_est[i] + v_L_dot[2] * dt

    # Calculate the positions
    r_dot = np.array([
        vn_est[i] / (RM_est[i] + h_est[i]),
        ve_est[i] / ((RN_est[i] + h_est[i]) * np.cos(lat_est[i])),
        -vd_est[i]
    ])

    # Integrate to get the next position
    lat_est[i+1] = lat_est[i] + r_dot[0] * dt
    lon_est[i+1] = lon_est[i] + r_dot[1] * dt
    h_est[i+1] = h_est[i] + r_dot[2] * dt

    # Convert the DCM to Euler angles
    pitch_est[i+1] = -np.arcsin(dcm_est[2,0,i+1])
    roll_est[i+1] = np.arctan2(dcm_est[2,1,i+1], dcm_est[2,2,i+1])
    heading_est[i+1] = np.arctan2(dcm_est[1,0,i+1], dcm_est[0,0,i+1])

    # Update v_L_est
    v_L_est[i+1,:] = np.array([vn_est[i+1], ve_est[i+1], vd_est[i+1]])


# Print the individual RMSEs between the estimations and the datasets
lat_rmse = np.sqrt(np.mean((lat - lat_est)**2))
lon_rmse = np.sqrt(np.mean((lon - lon_est)**2))
alt_rmse = np.sqrt(np.mean((h - h_est)**2))
vn_rmse = np.sqrt(np.mean((vn - vn_est)**2))
ve_rmse = np.sqrt(np.mean((ve - ve_est)**2))
vd_rmse = np.sqrt(np.mean((vd - vd_est)**2))
roll_rmse = np.sqrt(np.mean((roll - roll_est)**2))
pitch_rmse = np.sqrt(np.mean((pitch - pitch_est)**2))
heading_rmse = np.sqrt(np.mean((heading - heading_est)**2))

print(f'Latitude RMSE: {lat_rmse}')
print(f'Longitude RMSE: {lon_rmse}')
print(f'Altitude RMSE: {alt_rmse}')
print(f'Velocity North RMSE: {vn_rmse}')
print(f'Velocity East RMSE: {ve_rmse}')
print(f'Velocity Down RMSE: {vd_rmse}')
print(f'Roll RMSE: {roll_rmse}')
print(f'Pitch RMSE: {pitch_rmse}')
print(f'Heading RMSE: {heading_rmse}')

#==============================================================================
# PLOTTING
#==============================================================================

if DATASET1:
    # Plot the accelerometer data
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, accel[:, i], label='Reference', color='black')
        plt.plot(time, a_B[:, i], '--', label='Computed', color='red')
        plt.ylabel(f'Accel {["X", "Y", "Z"][i]} (m/s²)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
    plt.suptitle('Accelerometer Data Comparison',fontsize=14)
    plt.xlabel('Time (s)',fontsize=14)
    plt.tight_layout()
    plt.savefig('inverse_kinematics_acc_dataset1.pdf')

    # Plot the gyroscope data
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, gyro[:, i], label='Reference', color='black')
        plt.plot(time, gyro_corrected[:, i], '--', label='Computed', color='red')
        plt.ylabel(f'Gyro {["X", "Y", "Z"][i]} (rad/s)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
    plt.suptitle('Gyroscope Data Comparison',fontsize=14)
    plt.xlabel('Time (s)',fontsize=14)
    plt.tight_layout()
    plt.savefig('inverse_kinematics_gyro_dataset1.pdf')

    # Compare the estimations and the data
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, lat, label='Reference', color='black')
    plt.plot(time, lat_est, '--', label='Estimated', color='red')
    plt.ylabel('Latitude (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, lon, label='Reference', color='black')
    plt.plot(time, lon_est, '--', label='Estimated', color='red')
    plt.ylabel('Longitude (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, h, label='Reference', color='black')
    plt.plot(time, h_est, '--', label='Estimated', color='red')
    plt.ylabel('Altitude (m)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.suptitle('Position Estimation Comparison', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_position_estimation_dataset1_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_position_estimation_dataset1_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_position_estimation_dataset1_az.pdf')
    elif NOISE_GYRO_X:
        plt.savefig('dynamics_position_estimation_dataset1_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_position_estimation_dataset1_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_position_estimation_dataset1_gz.pdf')
    else:
        plt.savefig('dynamics_position_estimation_dataset1_no_noise.pdf')

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, vn, label='Reference', color='black')
    plt.plot(time, vn_est, '--', label='Estimated', color='red')
    plt.ylabel('Velocity North (m/s)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, ve, label='Reference', color='black')
    plt.plot(time, ve_est, '--', label='Estimated', color='red')
    plt.ylabel('Velocity East (m/s)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, vd, label='Reference', color='black')
    plt.plot(time, vd_est, '--', label='Estimated', color='red')
    plt.ylabel('Velocity Down (m/s)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.suptitle('Velocity Estimation Comparison', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_velocity_estimation_dataset1_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_velocity_estimation_dataset1_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_velocity_estimation_dataset1_az.pdf')
    elif NOISE_GYRO_X:
        plt.savefig('dynamics_velocity_estimation_dataset1_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_velocity_estimation_dataset1_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_velocity_estimation_dataset1_gz.pdf')
    else:
        plt.savefig('dynamics_velocity_estimation_dataset1_no_noise.pdf')

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, roll, label='Reference', color='black')
    plt.plot(time, roll_est, '--', label='Estimated', color='red')
    plt.ylabel('Roll (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, pitch, label='Reference', color='black')
    plt.plot(time, pitch_est, '--', label='Estimated', color='red')
    plt.ylabel('Pitch (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, heading, label='Reference', color='black')
    plt.plot(time, np.unwrap(heading_est), '--', label='Estimated', color='red')
    plt.ylabel('Heading (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.suptitle('Orientation Estimation Comparison', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_orientation_estimation_dataset1_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_orientation_estimation_dataset1_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_orientation_estimation_dataset1_az.pdf')
    elif NOISE_GYRO_X:
        plt.savefig('dynamics_orientation_estimation_dataset1_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_orientation_estimation_dataset1_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_orientation_estimation_dataset1_gz.pdf')
    else:
        plt.savefig('dynamics_orientation_estimation_dataset1_no_noise.pdf')


    # Plot the quaternion components
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(time, q_LB[:, 0], label='Reference', color='black')
    plt.plot(time, q_LB_est[:, 0], '--', label='Estimated', color='red')
    plt.ylabel('q0', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(4, 1, 2)
    plt.plot(time, q_LB[:, 1], label='Reference', color='black')
    plt.plot(time, q_LB_est[:, 1], '--', label='Estimated', color='red')
    plt.ylabel('q1', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(4, 1, 3)
    plt.plot(time, q_LB[:, 2], label='Reference', color='black')
    plt.plot(time, q_LB_est[:, 2], '--', label='Estimated', color='red')
    plt.ylabel('q2', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(4, 1, 4)
    plt.plot(time, q_LB[:, 3], label='Reference', color='black')
    plt.plot(time, q_LB_est[:, 3], '--', label='Estimated', color='red')
    plt.ylabel('q3', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.suptitle('Quaternion Estimation Comparison', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_quaternion_estimation_dataset1_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_quaternion_estimation_dataset1_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_quaternion_estimation_dataset1_az.pdf')
    elif NOISE_GYRO_X:
        plt.savefig('dynamics_quaternion_estimation_dataset1_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_quaternion_estimation_dataset1_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_quaternion_estimation_dataset1_gz.pdf')
    else:
        plt.savefig('dynamics_quaternion_estimation_dataset1_no_noise.pdf')

    # Plot the acceleration and gyro data to show the noise and bias
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, accel[:, i], label='Reference', color='black')
        plt.plot(time, accel_est[:, i], '--', label='Estimated', color='red')
        plt.ylabel(f'Accel {["X", "Y", "Z"][i]} (m/s²)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
    plt.suptitle('Accelerometer Data Comparison with Noise and Bias', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_accel_estimation_dataset1_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_accel_estimation_dataset1_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_accel_estimation_dataset1_az.pdf')
    else:
        plt.savefig('dynamics_accel_estimation_dataset1_no_noise.pdf')


    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, gyro[:, i], label='Reference', color='black')
        plt.plot(time, gyro_est[:, i], '--', label='Estimated', color='red')
        plt.ylabel(f'Gyro {["X", "Y", "Z"][i]} (rad/s)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
    plt.suptitle('Gyroscope Data Comparison with Noise and Bias', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_GYRO_X:
        plt.savefig('dynamics_gyro_estimation_dataset1_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_gyro_estimation_dataset1_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_gyro_estimation_dataset1_gz.pdf')
    else:
        plt.savefig('dynamics_gyro_estimation_dataset1_no_noise.pdf')
    plt.show()

elif DATASET2:
    # Plot the accelerometer data
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, accel[:, i], label='Reference', color='black')
        plt.plot(time, a_B[:, i], '--', label='Computed', color='red')
        plt.ylabel(f'Accel {["X", "Y", "Z"][i]} (m/s²)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
    plt.suptitle('Accelerometer Data Comparison',fontsize=14)
    plt.xlabel('Time (s)',fontsize=14)
    plt.tight_layout()
    plt.savefig('inverse_kinematics_acc_dataset2.pdf')

    # Plot the gyroscope data
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, gyro[:, i], label='Reference', color='black')
        plt.plot(time, gyro_corrected[:, i], '--', label='Computed', color='red')
        plt.ylabel(f'Gyro {["X", "Y", "Z"][i]} (rad/s)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
    plt.suptitle('Gyroscope Data Comparison',fontsize=14)
    plt.xlabel('Time (s)',fontsize=14)
    plt.tight_layout()
    plt.savefig('inverse_kinematics_gyro_dataset2.pdf')

    # Compare the estimations and the data
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, lat, label='Reference', color='black')
    plt.plot(time, lat_est, '--', label='Estimated', color='red')
    plt.ylabel('Latitude (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, lon, label='Reference', color='black')
    plt.plot(time, lon_est, '--', label='Estimated', color='red')
    plt.ylabel('Longitude (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, h, label='Reference', color='black')
    plt.plot(time, h_est, '--', label='Estimated', color='red')
    plt.ylabel('Altitude (m)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.suptitle('Position Estimation Comparison', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_position_estimation_dataset2_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_position_estimation_dataset2_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_position_estimation_dataset2_az.pdf')
    elif NOISE_GYRO_X:
        plt.savefig('dynamics_position_estimation_dataset2_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_position_estimation_dataset2_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_position_estimation_dataset2_gz.pdf')
    else:
        plt.savefig('dynamics_position_estimation_dataset2_no_noise.pdf')

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, vn, label='Reference', color='black')
    plt.plot(time, vn_est, '--', label='Estimated', color='red')
    plt.ylabel('Velocity North (m/s)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, ve, label='Reference', color='black')
    plt.plot(time, ve_est, '--', label='Estimated', color='red')
    plt.ylabel('Velocity East (m/s)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, vd, label='Reference', color='black')
    plt.plot(time, vd_est, '--', label='Estimated', color='red')
    plt.ylabel('Velocity Down (m/s)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.suptitle('Velocity Estimation Comparison', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_velocity_estimation_dataset2_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_velocity_estimation_dataset2_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_velocity_estimation_dataset2_az.pdf')
    elif NOISE_GYRO_X:
        plt.savefig('dynamics_velocity_estimation_dataset2_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_velocity_estimation_dataset2_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_velocity_estimation_dataset2_gz.pdf')
    else:
        plt.savefig('dynamics_velocity_estimation_dataset2_no_noise.pdf')

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, roll, label='Reference', color='black')
    plt.plot(time, roll_est, '--', label='Estimated', color='red')
    plt.ylabel('Roll (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, pitch, label='Reference', color='black')
    plt.plot(time, pitch_est, '--', label='Estimated', color='red')
    plt.ylabel('Pitch (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, heading, label='Reference', color='black')
    plt.plot(time, np.unwrap(heading_est), '--', label='Estimated', color='red')
    plt.ylabel('Heading (rad)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.suptitle('Orientation Estimation Comparison', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_orientation_estimation_dataset2_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_orientation_estimation_dataset2_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_orientation_estimation_dataset2_az.pdf')
    elif NOISE_GYRO_X:
        plt.savefig('dynamics_orientation_estimation_dataset2_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_orientation_estimation_dataset2_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_orientation_estimation_dataset2_gz.pdf')
    else:
        plt.savefig('dynamics_orientation_estimation_dataset2_no_noise.pdf')

    # Plot the quaternion components
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(time, q_LB[:, 0], label='Reference', color='black')
    plt.plot(time, q_LB_est[:, 0], '--', label='Estimated', color='red')
    plt.ylabel('q0', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(4, 1, 2)
    plt.plot(time, q_LB[:, 1], label='Reference', color='black')
    plt.plot(time, q_LB_est[:, 1], '--', label='Estimated', color='red')
    plt.ylabel('q1', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(4, 1, 3)
    plt.plot(time, q_LB[:, 2], label='Reference', color='black')
    plt.plot(time, q_LB_est[:, 2], '--', label='Estimated', color='red')
    plt.ylabel('q2', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.subplot(4, 1, 4)
    plt.plot(time, q_LB[:, 3], label='Reference', color='black')
    plt.plot(time, q_LB_est[:, 3], '--', label='Estimated', color='red')
    plt.ylabel('q3', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.suptitle('Quaternion Estimation Comparison', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_quaternion_estimation_dataset2_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_quaternion_estimation_dataset2_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_quaternion_estimation_dataset2_az.pdf')
    elif NOISE_GYRO_X:
        plt.savefig('dynamics_quaternion_estimation_dataset2_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_quaternion_estimation_dataset2_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_quaternion_estimation_dataset2_gz.pdf')
    else:
        plt.savefig('dynamics_quaternion_estimation_dataset2_no_noise.pdf')

    # Plot the acceleration and gyro data to show the noise and bias
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, accel[:, i], label='Reference', color='black')
        plt.plot(time, accel_est[:, i], '--', label='Estimated', color='red')
        plt.ylabel(f'Accel {["X", "Y", "Z"][i]} (m/s²)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
    plt.suptitle('Accelerometer Data Comparison with Noise and Bias', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_ACCEL_X:
        plt.savefig('dynamics_accel_estimation_dataset2_ax.pdf')
    elif NOISE_ACCEL_Y:
        plt.savefig('dynamics_accel_estimation_dataset2_ay.pdf')
    elif NOISE_ACCEL_Z:
        plt.savefig('dynamics_accel_estimation_dataset2_az.pdf')
    else:
        plt.savefig('dynamics_accel_estimation_dataset2_no_noise.pdf')

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, gyro[:, i], label='Reference', color='black')
        plt.plot(time, gyro_est[:, i], '--', label='Estimated', color='red')
        plt.ylabel(f'Gyro {["X", "Y", "Z"][i]} (rad/s)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
    plt.suptitle('Gyroscope Data Comparison with Noise and Bias', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    if NOISE_GYRO_X:
        plt.savefig('dynamics_gyro_estimation_dataset2_gx.pdf')
    elif NOISE_GYRO_Y:
        plt.savefig('dynamics_gyro_estimation_dataset2_gy.pdf')
    elif NOISE_GYRO_Z:
        plt.savefig('dynamics_gyro_estimation_dataset2_gz.pdf')
    else:
        plt.savefig('dynamics_gyro_estimation_dataset2_no_noise.pdf')
    plt.show()