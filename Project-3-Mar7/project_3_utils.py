from scipy.io import loadmat
import math

def calcN(lat, a=6378137.0, b=6356752.31425):
    """
    Calculates the prime vertical radius of curvature at a given latitude.

    Parameters:
        lat : float
            Latitude in radians.
        a : float, optional
            Semi-major axis of the ellipsoid (default is WGS84: 6378137.0).
        b : float, optional
            Semi-minor axis of the ellipsoid (default is WGS84: 6356752.31425).

    Returns:
        N : float
            The prime vertical radius of curvature.
    """
    e2 = (a**2 - b**2) / (a**2)
    W = math.sqrt(1.0 - e2 * (math.sin(lat)**2))
    N = a / W
    return N

def xyz2plh(*args):
    """
    Converts Cartesian coordinates (x, y, z) to geodetic coordinates (latitude, longitude, height).
    
    The function can be called in three ways:
    
      1. xyz2plh(xyz) 
         where xyz is an iterable (e.g. list or tuple) containing [x, y, z].
      2. xyz2plh(x, y, z)
         where x, y, z are provided as separate arguments.
      3. xyz2plh(x, y, z, a, b)
         where a and b are the ellipsoid parameters.
    
    The default ellipsoid parameters are those of WGS84.
    
    Returns:
        A tuple (lat, lon, h) where:
          lat : Latitude in radians.
          lon : Longitude in radians.
          h   : Height above the ellipsoid.
    """
    # Handle different input types
    if len(args) == 1:
        xyz = args[0]
        if len(xyz) < 3:
            raise ValueError("Input vector must have at least three elements [x, y, z].")
        x, y, z = xyz[0], xyz[1], xyz[2]
        a = 6378137.0
        b = 6356752.31425
    elif len(args) == 3:
        x, y, z = args
        a = 6378137.0
        b = 6356752.31425
    elif len(args) == 5:
        x, y, z, a, b = args
    else:
        raise ValueError("Invalid number of arguments. Expected 1, 3, or 5 arguments.")
    
    e2 = (a**2 - b**2) / (a**2)
    p = math.sqrt(x**2 + y**2)
    
    # Check for singularity (near the z-axis)
    if p <= 1.0e-6:
        if z > 0:
            lat = math.pi / 2.0
        else:
            lat = -math.pi / 2.0
        lon = 0.0  # longitude is undefined when p is nearly zero
        h = abs(z) - b
    else:
        N0 = 0
        h0 = 0
        # Initial guess for latitude (phi)
        phi = math.atan((z / p) / (1 - e2))
        N1 = calcN(phi, a, b)
        h1 = p / math.cos(phi) - N1
        phi = math.atan((z / p) / (1 - e2 * N1 / (N1 + h1)))
        
        # Iterate until convergence
        while abs(N1 - N0) >= 0.01 and abs(h1 - h0) >= 0.01:
            N0 = N1
            h0 = h1
            N1 = calcN(phi, a, b)
            h1 = p / math.cos(phi) - N1
            phi = math.atan((z / p) / (1 - e2 * N1 / (N1 + h1)))
        
        lat = phi
        lon = math.atan2(y, x)
        h = h1
    
    return (lat, lon, h)

def loadProjectMATLABFiles():
    """
    Load the MATLAB files for Project #3

    Returns
    -------
    BGP_Alt : numpy.ndarray
        Geoidal Height (meters)
    BGP_Lat : numpy.ndarray
        Latitude (degrees)
    BGP_Long : numpy.ndarray
        Longitude (degrees)
    BGP_second : numpy.ndarray
        Second within the GPS week in each epoch
    BGP_Lat_std : numpy.ndarray
        Latitude standard deviation (degrees)
    BGP_Long_std : numpy.ndarray
        Longitude standard deviation (degrees)
    BGP_Alt_std : numpy.ndarray
        Geoidal Height standard deviation (meters)
    BGV_second : numpy.ndarray
        Second within the GPS week in each epoch
    BGV_hor_spd : numpy.ndarray
        Horizontal speed (m/s)
    BGV_az : numpy.ndarray
        Azimuth (degrees)
    BGV_vu : numpy.ndarray
        Up velocity (m/s)
    BGV_ve : numpy.ndarray
        East velocity (m/s)
    BGV_vn : numpy.ndarray
        North velocity (m/s)
    pos_time_XYZ : numpy.ndarray
        Second within the GPS week in each epoch
    no_of_sv_XYZ : numpy.ndarray
        Number of visible satellites in each epoch
    Sat_Num_XYZ : numpy.ndarray
        Available satellites PRN in each epoch
    Sat_X : numpy.ndarray
        Satellites positions in ECEF coordinate frame (meters)
    Sat_Y : numpy.ndarray
        Satellites positions in ECEF coordinate frame (meters)
    Sat_Z : numpy.ndarray
        Satellites positions in ECEF coordinate frame (meters)
    Sat_Clk_Corr : numpy.ndarray
        Satellites clocks errors (meters)
    Sat_Ion_Corr : numpy.ndarray
        Ionospheric errors (meters)
    Sat_Tro_Corr : numpy.ndarray
        Tropospheric errors (meters)
    no_of_GPS : numpy.ndarray
        Number of visible satellites in each epoch
    GPS_week : numpy.ndarray
        Week number since 21-22 August 1999
    sat_x : numpy.ndarray
        Satellites positions in ECEF coordinate frame (meters)
    sat_y : numpy.ndarray
        Satellites positions in ECEF coordinate frame (meters)
    sat_z : numpy.ndarray
        Satellites positions in ECEF coordinate frame (meters)
    sat_vx : numpy.ndarray
        Satellites velocities in ECEF coordinate frame (m/s)
    sat_vy : numpy.ndarray
        Satellites velocities in ECEF coordinate frame (m/s)
    sat_vz : numpy.ndarray
        Satellites velocities in ECEF coordinate frame (m/s)
    dt_sv : numpy.ndarray
        Satellite clock bias (seconds)
    sv_no : numpy.ndarray
        PRNs of available satellites in each epoch
    rho_C1 : numpy.ndarray
        Pseudorange (meters)
    dopp_L1 : numpy.ndarray
        Doppler of L1 frequency (Hz)
    dopp_L2 : numpy.ndarray
        Doppler of L2 frequency (Hz)
    phase_L1 : numpy.ndarray
        Carrier phase of L1 frequency (cycles)
    phase_L2 : numpy.ndarray
        Carrier phase of L2 frequency (cycles)
    GPS_sec : numpy.ndarray
        Second within the GPS week in each epoch
    rho_C1_std : numpy.ndarray
        Pseudorange standard deviation (meters)
    phase_L1_std : numpy.ndarray
        Carrier phase standard deviation of L1 frequency (cycles)
    phase_L2_std : numpy.ndarray
        Carrier phase standard deviation of L2 frequency (cycles)
    """

    
    # Load the BESTGPSPOS MATLAB file
    BESTGPSPOS = loadmat('BESTGPSPOS.mat', squeeze_me=True, struct_as_record=False)

    # Extract the variables from the MATLAB file
    BGP_Alt = BESTGPSPOS['BGP_Alt']  # Geoidal Height (meters)
    BGP_Lat = BESTGPSPOS['BGP_Lat']  # Latitude (degrees)
    BGP_Long = BESTGPSPOS['BGP_Long']  # Longitude (degrees)
    BGP_second = BESTGPSPOS['BGP_second']  # Second within the GPS week in each epoch
    BGP_Lat_std = BESTGPSPOS['BGP_Lat_std']  # Latitude standard deviation (degrees)
    BGP_Long_std = BESTGPSPOS['BGP_Long_std']  # Longitude standard deviation (degrees)
    BGP_Alt_std = BESTGPSPOS['BGP_Alt_std']  # Geoidal Height standard deviation (meters)

    # Load the BESTGPSVEL MATLAB file
    BESTGPSVEL = loadmat('BESTGPSVEL.mat', squeeze_me=True, struct_as_record=False)

    # Extract the variables from the MATLAB file
    BGV_second = BESTGPSVEL['BGV_second']  # Second within the GPS week in each epoch
    BGV_hor_spd = BESTGPSVEL['BGV_hor_spd']  # Horizontal speed (m/s)
    BGV_az = BESTGPSVEL['BGV_az']  # Azimuth (degrees)
    BGV_vu = BESTGPSVEL['BGV_vu']  # Up velocity (m/s)
    BGV_ve = BESTGPSVEL['BGV_ve']  # East velocity (m/s)
    BGV_vn = BESTGPSVEL['BGV_vn']  # North velocity (m/s)

    # Load the SATXYZ MATLAB file
    SATXYZ = loadmat('SATXYZ.mat', squeeze_me=True, struct_as_record=False)

    # Extract the variables from the MATLAB file
    pos_time_XYZ = SATXYZ['pos_time_XYZ']  # Second within the GPS week in each epoch
    no_of_sv_XYZ = SATXYZ['no_of_sv_XYZ']  # Number of visible satellites in each epoch
    Sat_Num_XYZ = SATXYZ['Sat_Num_XYZ']  # Available satellites PRN in each epoch
    Sat_X = SATXYZ['Sat_X']  # Satellites positions in ECEF coordinate frame (meters)
    Sat_Y = SATXYZ['Sat_Y']  # Satellites positions in ECEF coordinate frame (meters)
    Sat_Z = SATXYZ['Sat_Z']  # Satellites positions in ECEF coordinate frame (meters)
    Sat_Clk_Corr = SATXYZ['Sat_Clk_Corr']  # Satellites clocks errors (meters)
    Sat_Ion_Corr = SATXYZ['Sat_Ion_Corr']  # Ionospheric errors (meters)
    Sat_Tro_Corr = SATXYZ['Sat_Tro_Corr']  # Tropospheric errors (meters)

    # Load the GPS_measurements MATLAB file
    GPS_measurements = loadmat('GPS_measuerments.mat', squeeze_me=True, struct_as_record=False)

    # Extract the variables from the MATLAB file
    no_of_GPS = GPS_measurements['no_of_GPS']  # Number of visible satellites in each epoch
    GPS_week = GPS_measurements['GPS_week']  # Week number since 21-22 August 1999
    sat_x = GPS_measurements['sat_x']  # Satellites positions in ECEF coordinate frame (meters)
    sat_y = GPS_measurements['sat_y']  # Satellites positions in ECEF coordinate frame (meters)
    sat_z = GPS_measurements['sat_z']  # Satellites positions in ECEF coordinate frame (meters)
    sat_vx = GPS_measurements['sat_vx']  # Satellites velocities in ECEF coordinate frame (m/s)
    sat_vy = GPS_measurements['sat_vy']  # Satellites velocities in ECEF coordinate frame (m/s)
    sat_vz = GPS_measurements['sat_vz']  # Satellites velocities in ECEF coordinate frame (m/s)
    dt_sv = GPS_measurements['dt_sv']  # Satellite clock bias (seconds)
    sv_no = GPS_measurements['sv_no']  # PRNs of available satellites in each epoch
    rho_C1 = GPS_measurements['rho_C1']  # Pseudorange (meters)
    dopp_L1 = GPS_measurements['dopp_L1']  # Doppler of L1 frequency (Hz)
    dopp_L2 = GPS_measurements['dopp_L2']  # Doppler of L2 frequency (Hz)
    phase_L1 = GPS_measurements['phase_L1']  # Carrier phase of L1 frequency (cycles)
    phase_L2 = GPS_measurements['phase_L2']  # Carrier phase of L2 frequency (cycles)
    GPS_sec = GPS_measurements['GPS_sec']  # Second within the GPS week in each epoch
    rho_C1_std = GPS_measurements['rho_C1_std']  # Pseudorange standard deviation (meters)
    phase_L1_std = GPS_measurements['phase_L1_std']  # Carrier phase standard deviation of L1 frequency (cycles)
    phase_L2_std = GPS_measurements['phase_L2_std']  # Carrier phase standard deviation of L2 frequency (cycles)

    return BGP_Alt, BGP_Lat, BGP_Long, BGP_second, BGP_Lat_std, BGP_Long_std, BGP_Alt_std, BGV_second, BGV_hor_spd, BGV_az, BGV_vu, BGV_ve, BGV_vn, pos_time_XYZ, no_of_sv_XYZ, Sat_Num_XYZ, Sat_X, Sat_Y, Sat_Z, Sat_Clk_Corr, Sat_Ion_Corr, Sat_Tro_Corr, no_of_GPS, GPS_week, sat_x, sat_y, sat_z, sat_vx, sat_vy, sat_vz, dt_sv, sv_no, rho_C1, dopp_L1, dopp_L2, phase_L1, phase_L2, GPS_sec, rho_C1_std, phase_L1_std, phase_L2_std

