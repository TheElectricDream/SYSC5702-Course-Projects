import numpy as np

def euler_to_dcm(roll, pitch, azimuth):
    """
    Converts Euler angles (roll, pitch, yaw) to a Direction Cosine Matrix (DCM).

    Parameters:
        roll (float): The roll angle in radians.
        pitch (float): The pitch angle in radians.
        azimuth (float): The azimuth angle in radians.

    Returns:
        np.ndarray: A 3x3 direction cosine matrix representing the rotation.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cA, sA = np.cos(azimuth), np.sin(azimuth)
    
    dcm = np.array([[cp*cA, -cr*sA+sr*sp*cA, sr*sA+cr*sp*cA],
                    [cp*sA, cr*cA+sr*sp*sA, -sr*cA+cr*sp*sA],
                    [-sp, sr*cp, cr*cp]])
    
    return dcm

def convert_dcm_to_quat(C):
    """
    Convert a Direction Cosine Matrix (DCM) to a quaternion.
    
    Parameters:
        C (numpy.ndarray): A 3x3 rotation matrix.
        
    Returns:
        numpy.ndarray: A 4-element quaternion vector [a, b, c, d] 
                       with a positive scalar component.
    """
    # Compute trace and intermediary terms
    Tr = np.trace(C)
    Pa = 1 + Tr
    Pb = 1 + 2 * C[0, 0] - Tr
    Pc = 1 + 2 * C[1, 1] - Tr
    Pd = 1 + 2 * C[2, 2] - Tr

    # Determine the maximum term. In MATLAB, [m,i]=max([Pa Pb Pc Pd]);
    # MATLAB indices: 1,2,3,4 correspond to Python indices: 0,1,2,3.
    values = [Pa, Pb, Pc, Pd]
    i = np.argmax(values)  # i will be 0, 1, 2, or 3

    if i == 0:
        a = 0.5 * np.sqrt(Pa)
        b = (C[2, 1] - C[1, 2]) / (4 * a)
        c = (C[0, 2] - C[2, 0]) / (4 * a)
        d = (C[1, 0] - C[0, 1]) / (4 * a)
    elif i == 1:
        b = 0.5 * np.sqrt(Pb)
        c = (C[1, 0] + C[0, 1]) / (4 * b)
        d = (C[0, 2] + C[2, 0]) / (4 * b)
        a = (C[2, 1] - C[1, 2]) / (4 * b)
    elif i == 2:
        c = 0.5 * np.sqrt(Pc)
        d = (C[2, 1] + C[1, 2]) / (4 * c)
        a = (C[0, 2] - C[2, 0]) / (4 * c)
        b = (C[1, 0] + C[0, 1]) / (4 * c)
    elif i == 3:
        d = 0.5 * np.sqrt(Pd)
        a = (C[1, 0] - C[0, 1]) / (4 * d)
        b = (C[0, 2] + C[2, 0]) / (4 * d)
        c = (C[2, 1] + C[1, 2]) / (4 * d)

    # Ensure the scalar component 'a' is positive.
    if a <= 0:
        a, b, c, d = -a, -b, -c, -d

    quat_vector = np.array([a, b, c, d])
    return quat_vector
