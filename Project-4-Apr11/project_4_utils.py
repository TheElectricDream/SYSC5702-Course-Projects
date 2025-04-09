from scipy.io import loadmat

def import_project_data(filepath=None):
    '''
    This function loads in the data for project #4 and returns the imported data as a tuple.

    Inputs
        filepath: The path to the file that will be imported.

    Outputs
        t: Time vector in seconds
        f_theta_noisy: Noisy biased Azimuth angle control input (rad)
        f_a_noisy: Noisy biased forward acceleration control input (m/s3)
        East_GPS: Noisy GPS measurement of east position (m)
        North_GPS: Noisy GPS measurement of noth position (m)
        Azimuth_Magnetometer: Noisy magnetometer measurement of azimuth (rad)
        East_ref: Reference east position (m)
        North_ref: Reference north position (m)
        Azimuth_ref: Reference azimuth position (rad)
        initial_accel: Initial acceleration (m/s2)
        initial_forward_speed: Initial forward speed (m/s)    

    '''

    # Check for valid filepath
    if filepath is None:
        raise ValueError('Error: filepath input cannot be None')

    # Load the data from the provided MAT file
    data = loadmat(filepath)

    # Extract data from the loaded file
    t = data['t'].flatten()  
    f_theta_noisy = data['f_theta_noisy'].flatten()
    f_a_noisy = data['f_a_noisy'].flatten()  
    East_GPS = data['East_GPS'].flatten()  
    North_GPS = data['North_GPS'].flatten() 
    Azimuth_Magnetometer = data['Azimuth_Magnetometer'].flatten()  
    East_ref = data['East_ref'].flatten() 
    North_ref = data['North_ref'].flatten() 
    Azimuth_ref = data['Azimuth_ref'].flatten() 
    initial_accel = data['initial_accel'].flatten()[0]  
    initial_forward_speed = data['initial_forward_speed'].flatten()[0] 

    # Return the loaded data
    return t, f_theta_noisy, f_a_noisy, East_GPS, North_GPS, Azimuth_Magnetometer, East_ref, North_ref, Azimuth_ref, initial_accel, initial_forward_speed