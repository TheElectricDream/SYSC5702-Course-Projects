#==============================================================================
# PREAMBLE
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import project_3_utils as utils
from scipy.interpolate import interp1d

#==============================================================================
# LOAD DATA
#==============================================================================

# Load in all project datasets
BGP_Alt, BGP_Lat, BGP_Long, BGP_second, BGP_Lat_std, BGP_Long_std, BGP_Alt_std, BGV_second, BGV_hor_spd, BGV_az, BGV_vu, BGV_ve, BGV_vn, pos_time_XYZ, no_of_sv_XYZ, Sat_Num_XYZ, Sat_X, Sat_Y, Sat_Z, Sat_Clk_Corr, Sat_Ion_Corr, Sat_Tro_Corr, no_of_GPS, GPS_week, sat_x, sat_y, sat_z, sat_vx, sat_vy, sat_vz, dt_sv, sv_no, rho_C1, dopp_L1, dopp_L2, phase_L1, phase_L2, GPS_sec, rho_C1_std, phase_L1_std, phase_L2_std = utils.loadProjectMATLABFiles()

#==============================================================================
# CONSTANTS
#==============================================================================

# Define the constants needed to control the main loop
MAX_ITR  = 10  # This is the maximum number of iterations allowed
CVG_TRSH = 0.005  # This is the convergence threshold, in meters

# Constants for velocity estimation
SPEED_OF_LIGHT = 299792458.0  # m/s
L1_FREQUENCY = 1575.42e6      # Hz
LAMBDA_L1 = SPEED_OF_LIGHT / L1_FREQUENCY  # Wavelength for L1 frequency

#==============================================================================
# INITIALIZATION
#==============================================================================

# Initialize arrays to store results
# First we initialize the position estimate arrays
num_eph = len(GPS_sec)  # This is the number of epochs, or timesteps
rcvr_pos_ecef = np.zeros((num_eph, 3))  # [X, Y, Z] positions in the ECEF frame
rcvr_pos_geo = np.zeros((num_eph, 3))   # [Lat, Long, Height] in geodetic coordinates
rcvr_clock_bias = np.zeros(num_eph)     # Clock bias

# And then we initialize the velocity estimate arrays
rcvr_vel_ecef = np.zeros((num_eph, 3))  # [Vx, Vy, Vz] velocities in the ECEF frame
rcvr_vel_enu = np.zeros((num_eph, 3))   # [Ve, Vn, Vu] velocities in the ENU frame
rcvr_clock_drift = np.zeros(num_eph)    # Clock drift rate

#==============================================================================
# MAIN LOOP
#==============================================================================
# First, we need to use iterative least-squares method and the given pseudorange measurements to estimate the receiver's 3D position in ECEF (X, Y, Z) for each time epoch.
print("Estimating receiver 3D position in ECEF...")

# Start a loop that will iterate through all of the timesteps in the dataset
for epoch in range(num_eph):

    # For each timestep, we first need to extract the data we need
    # First, the current GPS time in seconds of the week as well as 
    # The number of satellites available for thie timestep
    t_now = GPS_sec[epoch]
    n_sat = int(no_of_GPS[epoch])

    # We need to check that there are at least 4 satellites available for the estimation
    if n_sat < 4:

        print(f"Insufficient satellites for epoch: {t_now}")

        # If there are not enough satellites, then we will keep the previous timestep's values
        # and we will skip to the next timestep
        rcvr_pos_ecef[epoch, :] = rcvr_pos_ecef[epoch - 1, :]
        rcvr_pos_geo[epoch, :] = rcvr_pos_geo[epoch - 1, :]
        rcvr_clock_bias[epoch] = rcvr_clock_bias[epoch - 1]

        continue

    # If we have enough visible satellites, we also need to check that the same sattelite data is 
    # also available in the corrections file

    prn_nums = sv_no[epoch].flatten()  # PRN numbers for this timestep
    pseudoranges = rho_C1[epoch].flatten()  # Pseudoranges for this timestep
    pseudorange_stds = rho_C1_std[epoch].flatten()  # Pseudorange standard deviations for this timestep
    
    # Extract the satellite positions for this timestep
    sat_pos_x = sat_x[epoch].flatten()
    sat_pos_y = sat_y[epoch].flatten()
    sat_pos_z = sat_z[epoch].flatten()
    
    # We can now look at the time in the correction data to find the matching corrections at t_now
    matched_idx = np.where(pos_time_XYZ == t_now)[0]
    
    # Of course, if there is no matches, then keep the previous values and skip this timestep
    # This check should not trigger, but likely good to have anyways
    if len(matched_idx) == 0:

        print(f"No matching corrections for epoch: {t_now}")

        rcvr_pos_ecef[epoch, :] = rcvr_pos_ecef[epoch - 1, :]
        rcvr_pos_geo[epoch, :] = rcvr_pos_geo[epoch - 1, :]
        rcvr_clock_bias[epoch] = rcvr_clock_bias[epoch - 1]

        continue
    
    # Extract the index
    matched_idx = matched_idx[0]

    # Check that there are enough satellites in the correction data
    if isinstance(Sat_Num_XYZ[matched_idx], int):
        n_sat_corr = 1  # When there is a single satellite, it is an integer and thus len() does not work
    else:
        n_sat_corr = len(Sat_Num_XYZ[matched_idx])  # Gets the number of satellites in the correction data
    
    # If there are not enough satellites in the correction data, then we also skip this timestep
    # while keeping the previous values
    if n_sat_corr < 4:

        print(f"Insufficient satellites in correction data for epoch: {t_now}")

        rcvr_pos_ecef[epoch, :] = rcvr_pos_ecef[epoch - 1, :]
        rcvr_pos_geo[epoch, :] = rcvr_pos_geo[epoch - 1, :]
        rcvr_clock_bias[epoch] = rcvr_clock_bias[epoch - 1]

        continue
    
    # Now we can extract the correction data for the matching timestep
    corr_sat_prns = Sat_Num_XYZ[matched_idx].flatten()
    clock_corrections = Sat_Clk_Corr[matched_idx].flatten()
    iono_corrections = Sat_Ion_Corr[matched_idx].flatten()
    tropo_corrections = Sat_Tro_Corr[matched_idx].flatten()
    
    # However, just because there are enough satellites in both datasets, it does
    # not mean that the SAME satellites are avilable in both datasets; so, we must check that
    # the satellites available are the same in both sets
    common_sats = []
    idx_meas = []
    idx_corr = []
    
    for i, prn in enumerate(prn_nums):
        if prn in corr_sat_prns:
            common_sats.append(prn)
            idx_meas.append(i)
            idx_corr.append(np.where(corr_sat_prns == prn)[0][0])
    
    # Finally, if there are less then 4 commong satellites -- the minimum needed for the estimation --
    # then we keep the old values and skip this timestep
    if len(common_sats) < 4:

        print(f"Insufficient satellites for epoch: {t_now}")

        rcvr_pos_ecef[epoch, :] = rcvr_pos_ecef[epoch - 1, :]
        rcvr_pos_geo[epoch, :] = rcvr_pos_geo[epoch - 1, :]
        rcvr_clock_bias[epoch] = rcvr_clock_bias[epoch - 1]

        continue
    
    # With the alignment of the data complete, we can now extract the common data
    sat_x_common = sat_pos_x[idx_meas]
    sat_y_common = sat_pos_y[idx_meas]
    sat_z_common = sat_pos_z[idx_meas]
    pseudoranges_common = pseudoranges[idx_meas]
    std_common = pseudorange_stds[idx_meas]
    clock_corr_common = clock_corrections[idx_corr]
    iono_corr_common = iono_corrections[idx_corr]
    tropo_corr_common = tropo_corrections[idx_corr]
    
    # Number of common satellites
    n = len(common_sats)
    
    # Zero initial guess for the receiver position and clock bias
    x_est = 0
    y_est = 0
    z_est = 0
    b_est = 0  
    
    # Now we enter another loop that will iterate through the least-squares method
    for iter in range(MAX_ITR):  # Looping until we converge or hit the maximum iterations
        
        # Initialize H matrix and residuals vector
        H = np.zeros((n, 4))
        delta_rho = np.zeros(n)
        
        # Calculate estimated ranges based on current position estimate
        for i in range(n):
            
            # Geometric range
            rho_geometric = np.sqrt((x_est - sat_x_common[i])**2 + 
                                    (y_est - sat_y_common[i])**2 + 
                                    (z_est - sat_z_common[i])**2)
            
            # Estimated pseudorange with corrections
            rho_est = rho_geometric + b_est - clock_corr_common[i] + \
                     iono_corr_common[i] + tropo_corr_common[i]
            
            # Measurement residual
            delta_rho[i] = pseudoranges_common[i] - rho_est
            
            # Line of sight unit vector (partial derivatives for x,y,z)
            H[i, 0] = (x_est - sat_x_common[i]) / rho_geometric
            H[i, 1] = (y_est - sat_y_common[i]) / rho_geometric
            H[i, 2] = (z_est - sat_z_common[i]) / rho_geometric
            H[i, 3] = 1  # Partial derivative for clock bias
        
        # Weight matrix (inverse of measurement covariance)
        R = np.diag(std_common)
        W = np.linalg.inv(R)
        
        # Solve for state correction
        solution = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ delta_rho
        
        # Update state estimate
        x_est += solution[0]
        y_est += solution[1]
        z_est += solution[2]
        b_est += solution[3]
        
        # Check for convergence
        if np.linalg.norm(solution[:3]) < CVG_TRSH:
            print(f"Epoch {epoch} converged after {iter+1} iterations")
            break
    
    # Store result
    rcvr_pos_ecef[epoch, :] = [x_est, y_est, z_est]
    rcvr_clock_bias[epoch] = b_est
    
    # Convert ECEF to geodetic (lat, long, height) using the function provided 
    # by professor Atia
    lat, lon, h = utils.xyz2plh(x_est, y_est, z_est)
    rcvr_pos_geo[epoch, :] = [lat, lon, h]

#==============================================================================
# VELOCITY ESTIMATION
#==============================================================================

print("\nEstimating receiver velocity using Doppler measurements...")

# Loop through all epochs to estimate velocity using Doppler measurements
for epoch in range(num_eph):

    # Extract the current GPS time and number of satellites for this epoch
    t_now = GPS_sec[epoch]
    n_sat = int(no_of_GPS[epoch])
    
    # Check that there are at least 4 satellites available for the estimation
    if n_sat < 4:
        print(f"Insufficient satellites for velocity estimation at epoch: {t_now}")
        
        # If there are not enough satellites, keep previous timestep's values
        if epoch > 0:
            rcvr_vel_ecef[epoch, :] = rcvr_vel_ecef[epoch - 1, :]
            rcvr_vel_enu[epoch, :] = rcvr_vel_enu[epoch - 1, :]
            rcvr_clock_drift[epoch] = rcvr_clock_drift[epoch - 1]
        continue
    
    # Extract satellite data for this timestep
    prn_nums = sv_no[epoch].flatten()  # PRN numbers for this timestep
    doppler_measurements = dopp_L1[epoch].flatten()  # L1 Doppler measurements
    
    # Extract satellite positions and velocities for this timestep
    sat_pos_x = sat_x[epoch].flatten()
    sat_pos_y = sat_y[epoch].flatten()
    sat_pos_z = sat_z[epoch].flatten()
    sat_vel_x = sat_vx[epoch].flatten()
    sat_vel_y = sat_vy[epoch].flatten()
    sat_vel_z = sat_vz[epoch].flatten()
    
    # Get the current receiver position (already calculated)
    rcvr_x = rcvr_pos_ecef[epoch, 0]
    rcvr_y = rcvr_pos_ecef[epoch, 1]
    rcvr_z = rcvr_pos_ecef[epoch, 2]
    
    # We can now look at the time in the correction data to find the matching corrections
    matched_idx = np.where(pos_time_XYZ == t_now)[0]
    
    # If there is no match, keep the previous values and skip this timestep
    if len(matched_idx) == 0:
        print(f"No matching corrections for velocity estimation at epoch: {t_now}")
        
        if epoch > 0:
            rcvr_vel_ecef[epoch, :] = rcvr_vel_ecef[epoch - 1, :]
            rcvr_vel_enu[epoch, :] = rcvr_vel_enu[epoch - 1, :]
            rcvr_clock_drift[epoch] = rcvr_clock_drift[epoch - 1]
        continue
    
    # Extract the index
    matched_idx = matched_idx[0]
    
    # Check that there are enough satellites in the correction data
    if isinstance(Sat_Num_XYZ[matched_idx], int):
        n_sat_corr = 1
    else:
        n_sat_corr = len(Sat_Num_XYZ[matched_idx])
    
    # If there are not enough satellites in the correction data, skip this timestep
    if n_sat_corr < 4:
        print(f"Insufficient satellites in correction data for velocity at epoch: {t_now}")
        
        if epoch > 0:
            rcvr_vel_ecef[epoch, :] = rcvr_vel_ecef[epoch - 1, :]
            rcvr_vel_enu[epoch, :] = rcvr_vel_enu[epoch - 1, :]
            rcvr_clock_drift[epoch] = rcvr_clock_drift[epoch - 1]
        continue
    
    # Extract the correction data for the matching timestep
    corr_sat_prns = Sat_Num_XYZ[matched_idx].flatten()
    
    # Find common satellites between measurements and corrections
    common_sats = []
    idx_meas = []
    idx_corr = []
    
    for i, prn in enumerate(prn_nums):
        if prn in corr_sat_prns:
            common_sats.append(prn)
            idx_meas.append(i)
            idx_corr.append(np.where(corr_sat_prns == prn)[0][0])
    
    # Check if there are enough common satellites for estimation
    if len(common_sats) < 4:
        print(f"Insufficient common satellites for velocity at epoch: {t_now}")
        
        if epoch > 0:
            rcvr_vel_ecef[epoch, :] = rcvr_vel_ecef[epoch - 1, :]
            rcvr_vel_enu[epoch, :] = rcvr_vel_enu[epoch - 1, :]
            rcvr_clock_drift[epoch] = rcvr_clock_drift[epoch - 1]
        continue
    
    # Extract the common data
    sat_x_common = sat_pos_x[idx_meas]
    sat_y_common = sat_pos_y[idx_meas]
    sat_z_common = sat_pos_z[idx_meas]
    sat_vx_common = sat_vel_x[idx_meas]
    sat_vy_common = sat_vel_y[idx_meas]
    sat_vz_common = sat_vel_z[idx_meas]
    doppler_common = doppler_measurements[idx_meas]
    
    # Filter out any NaN Doppler measurements
    valid_indices = ~np.isnan(doppler_common)
    
    if np.sum(valid_indices) < 4:
        print(f"Not enough valid Doppler measurements at epoch: {t_now}")
        
        if epoch > 0:
            rcvr_vel_ecef[epoch, :] = rcvr_vel_ecef[epoch - 1, :]
            rcvr_vel_enu[epoch, :] = rcvr_vel_enu[epoch - 1, :]
            rcvr_clock_drift[epoch] = rcvr_clock_drift[epoch - 1]
        continue
    
    # Apply the valid indices filter to all data
    valid_doppler = doppler_common[valid_indices]
    valid_sat_x = sat_x_common[valid_indices]
    valid_sat_y = sat_y_common[valid_indices]
    valid_sat_z = sat_z_common[valid_indices]
    valid_sat_vx = sat_vx_common[valid_indices]
    valid_sat_vy = sat_vy_common[valid_indices]
    valid_sat_vz = sat_vz_common[valid_indices]
    
    # Number of valid measurements
    n_valid = len(valid_doppler)
    
    # Prepare matrices for least squares estimation
    H = np.zeros((n_valid, 4))
    doppler_adjusted = np.zeros(n_valid)
    
    for i in range(n_valid):
        # Calculate geometric range
        delta_x = rcvr_x - valid_sat_x[i]
        delta_y = rcvr_y - valid_sat_y[i]
        delta_z = rcvr_z - valid_sat_z[i]
        range_sat = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        
        # Line of sight unit vector
        e_x = delta_x / range_sat
        e_y = delta_y / range_sat
        e_z = delta_z / range_sat
        
        # Fill H matrix with direction cosines
        H[i, 0] = e_x
        H[i, 1] = e_y
        H[i, 2] = e_z
        H[i, 3] = 1  # For clock drift
        
        # Adjust Doppler measurements 
        # Satellite contribution to range rate
        sat_range_rate = e_x * valid_sat_vx[i] + e_y * valid_sat_vy[i] + e_z * valid_sat_vz[i]
        
        # Convert Doppler to range rate 
        measured_range_rate = valid_doppler[i] * LAMBDA_L1 
        
        # Adjusted measurements (what we want to estimate is the receiver velocity and clock drift)
        doppler_adjusted[i] = measured_range_rate - sat_range_rate
    
    # Solve for velocity and clock drift using least squares
    solution = np.linalg.inv(H.T @ H) @ H.T @ doppler_adjusted
    
    # Store velocity in ECEF coordinates and clock drift
    rcvr_vel_ecef[epoch, 0] = solution[0]  # vx
    rcvr_vel_ecef[epoch, 1] = solution[1]  # vy
    rcvr_vel_ecef[epoch, 2] = solution[2]  # vz
    rcvr_clock_drift[epoch] = solution[3]  # clock drift rate
    
    # Create rotation matrix from ECEF to ENU
    sin_lat = np.sin(rcvr_pos_geo[epoch, 0])
    cos_lat = np.cos(rcvr_pos_geo[epoch, 0])
    sin_lon = np.sin(rcvr_pos_geo[epoch, 1])
    cos_lon = np.cos(rcvr_pos_geo[epoch, 1])
    
    # Direction cosine matrix for ECEF to ENU transformation
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    ])
    
    # Apply rotation to get ENU velocity
    rcvr_vel_enu[epoch, :] = -R @ rcvr_vel_ecef[epoch, :]

#==============================================================================
# CALCULATE RMSE
#==============================================================================

# Interpolation functions for position
f_lat = interp1d(BGP_second, BGP_Lat, bounds_error=False, fill_value=np.nan)
f_lon = interp1d(BGP_second, BGP_Long, bounds_error=False, fill_value=np.nan)
f_alt = interp1d(BGP_second, BGP_Alt, bounds_error=False, fill_value=np.nan)

# Interpolation functions for velocity
f_vn = interp1d(BGV_second, BGV_vn, bounds_error=False, fill_value=np.nan)
f_ve = interp1d(BGV_second, BGV_ve, bounds_error=False, fill_value=np.nan)
f_vu = interp1d(BGV_second, BGV_vu, bounds_error=False, fill_value=np.nan)

# Interpolate reference data to GPS_sec times
ref_lat = f_lat(GPS_sec)
ref_lon = f_lon(GPS_sec)
ref_alt = f_alt(GPS_sec)
ref_vn = f_vn(GPS_sec)
ref_ve = f_ve(GPS_sec)
ref_vu = f_vu(GPS_sec)

# Calculate RMSE for latitude, longitude, altitude
valid_lat = ~np.isnan(ref_lat) & ~np.isnan(rcvr_pos_geo[:, 0])
valid_lon = ~np.isnan(ref_lon) & ~np.isnan(rcvr_pos_geo[:, 1])
valid_alt = ~np.isnan(ref_alt) & ~np.isnan(rcvr_pos_geo[:, 2])

rmse_lat = np.sqrt(np.mean((rcvr_pos_geo[valid_lat, 0] - ref_lat[valid_lat]*np.pi/180)**2))
rmse_lon = np.sqrt(np.mean((rcvr_pos_geo[valid_lon, 1] - ref_lon[valid_lon]*np.pi/180)**2))
rmse_alt = np.sqrt(np.mean((rcvr_pos_geo[valid_alt, 2] - ref_alt[valid_alt])**2))

# Calculate RMSE for velocity components
valid_vn = ~np.isnan(ref_vn) & ~np.isnan(rcvr_vel_enu[:, 1])
valid_ve = ~np.isnan(ref_ve) & ~np.isnan(rcvr_vel_enu[:, 0])
valid_vu = ~np.isnan(ref_vu) & ~np.isnan(-rcvr_vel_enu[:, 2])  # Note the negative sign for Up vs Down

rmse_vn = np.sqrt(np.mean((rcvr_vel_enu[valid_vn, 1] - ref_vn[valid_vn])**2))
rmse_ve = np.sqrt(np.mean((rcvr_vel_enu[valid_ve, 0] - ref_ve[valid_ve])**2))
rmse_vu = np.sqrt(np.mean((rcvr_vel_enu[valid_vu, 2] - ref_vu[valid_vu])**2))

print("\nRMSE Results:")
print("====================")
print(f"  Latitude:  {rmse_lat*180/np.pi:.6f} degrees")
print(f"  Longitude: {rmse_lon*180/np.pi:.6f} degrees")
print(f"  Altitude:  {rmse_alt:.2f} meters")
print(f"  North Velocity: {rmse_vn:.2f} m/s")
print(f"  East Velocity:  {rmse_ve:.2f} m/s")
print(f"  Down Velocity:  {rmse_vu:.2f} m/s")
print("====================")

#==============================================================================
# PLOT RESULTS
#==============================================================================

# Time vector for x-axis
time = GPS_sec.copy()

# Plot the ECEF positions (not reference data)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, rcvr_pos_ecef[:, 0], color='black')
plt.ylabel('X Position (m)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.suptitle('Receiver ECEF Coordinates (Estimated)', fontsize=14)
plt.subplot(3, 1, 2)
plt.plot(time, rcvr_pos_ecef[:, 1], color='black')
plt.ylabel('Y Position (m)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(time, rcvr_pos_ecef[:, 2], color='black')
plt.ylabel('Z Position (m)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.xlabel('GPS Time (s)', fontsize=14)
plt.tight_layout()
plt.savefig('gps_receiver_ecef_task1.pdf')

# Plot the latitude, longitude, and altitude against the reference data
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, ref_lat, label='Reference', color='black')
plt.plot(time, rcvr_pos_geo[:, 0]*180/np.pi, '--', label='Estimated', color='red')
plt.ylabel('Latitude (degrees)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid()
plt.suptitle('GPS Receiver Geodetic Coordinates (Estimated vs. Reference)', fontsize=14)
plt.subplot(3, 1, 2)
plt.plot(time, ref_lon, label='Reference', color='black')
plt.plot(time, rcvr_pos_geo[:, 1]*180/np.pi, '--', label='Estimated', color='red')
plt.ylabel('Longitude (degrees)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(time, ref_alt, label='Reference', color='black')
plt.plot(time, rcvr_pos_geo[:, 2], '--', label='Estimated', color='red')
plt.ylabel('Altitude (m)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid()
plt.xlabel('GPS Time (s)', fontsize=14)
plt.tight_layout()
plt.savefig('gps_receiver_geodetic_task1.pdf')

# Plot the ECEF velocities (not reference data)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, rcvr_vel_ecef[:, 0], color='black')
plt.ylabel('X Velocity (m/s)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.suptitle('GPS Receiver ECEF Velocities (Estimated)', fontsize=14)
plt.subplot(3, 1, 2)
plt.plot(time, rcvr_vel_ecef[:, 1], color='black')
plt.ylabel('Y Velocity (m/s)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(time, rcvr_vel_ecef[:, 2], color='black')
plt.ylabel('Z Velocity (m/s)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.xlabel('GPS Time (s)', fontsize=14)
plt.tight_layout()
plt.savefig('gps_receiver_ecef_velocity_task2.pdf')

# Plot the east, north, and down velocity components against the reference data
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, ref_ve, label='Reference', color='black')
plt.plot(time, rcvr_vel_enu[:, 0], '--', label='Estimated', color='red')
plt.ylabel('East Velocity (m/s)', fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.suptitle('Receiver ENU Velocities (Estimated vs. Reference)', fontsize=14)
plt.subplot(3, 1, 2)
plt.plot(time, ref_vn, label='Reference', color='black')
plt.plot(time, rcvr_vel_enu[:, 1], '--', label='Estimated', color='red')
plt.ylabel('North Velocity (m/s)', fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplot(3, 1, 3)
plt.plot(time, ref_vu, label='Reference', color='black')
plt.plot(time, rcvr_vel_enu[:, 2], '--', label='Estimated', color='red')
plt.ylabel('Up Velocity (m/s)', fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.xlabel('GPS Time (s)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('gps_receiver_enu_velocity_task2.pdf')


