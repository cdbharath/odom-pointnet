'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import os
import numpy as np
import math

def quaterntion_to_axis_angle(q):
    theta = 2 * np.arccos(q[0])
    u1 = q[1] / np.sqrt(1 - q[0]**2)
    u2 = q[2] / np.sqrt(1 - q[0]**2)
    u3 = q[3] / np.sqrt(1 - q[0]**2)
    return theta, np.array([u1, u2, u3])
    
def axis_angle_to_quaternion(theta, axis):
    qw = np.cos(theta / 2)
    qx = axis[0] * np.sin(theta / 2)
    qy = axis[1] * np.sin(theta / 2)
    qz = axis[2] * np.sin(theta / 2)
    return np.array([qw, qx, qy, qz])

def rotation_matrix_to_axis_angle(R):
    R = normalize_rotation_matrix(R)

    trace_R = np.trace(R)
    angle = np.arccos((trace_R - 1) / 2)
    angle = wrap_angle(angle)
    
    if np.isclose(angle, 0.0):
        # If the angle is close to zero, the rotation is negligible, and the axis can be any unit vector.
        axis = np.array([1.0, 0.0, 0.0])  # Default axis
    else:
        axis = (1 / (2 * np.sin(angle))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / np.linalg.norm(axis)
        
        # Ensure a consistent axis direction
        if axis[0] < 0:
            axis = -axis
            angle = -angle  # Adjust the angle
            
    return np.concatenate(([angle], axis))

def axis_angle_to_rotation_matrix(axis_angle):
    axis = axis_angle[1:]
    angle = axis_angle[0]
    
    axis = axis / np.linalg.norm(axis)
    angle = wrap_angle(angle)

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis

    R = np.array([[t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                  [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                  [t * x * z - y * s, t * y * z + x * s, t * z * z + c]])
    
    R = normalize_rotation_matrix(R)
    return R

def quaternion_from_rotation(R):
    """
    Convert a 3x3 rotation matrix to a unit quaternion.
    """
    q = np.empty((4,))
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
        
    return q / np.linalg.norm(q) # return the normalized quaternion

def rotation_from_quaternion(q):
    """
    Convert a unit quaternion to a 3x3 rotation matrix.
    """
    q = q / np.linalg.norm(q)
    R = np.empty((3, 3))
    
    R[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    R[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    R[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    
    R[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    R[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    R[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    
    R[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    R[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    R[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    
    return R

def is_rotation_matrix(matrix):
    # Check if matrix is a square matrix
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check orthogonality
    if not np.allclose(np.dot(matrix.T, matrix), np.eye(matrix.shape[0])):
        return False

    # Check determinant
    if not np.isclose(np.linalg.det(matrix), 1):
        return False

    return True
 
def normalize_rotation_matrix(matrix):
    # Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(matrix)

    # Ensure proper orientation of the rotation matrix
    det = np.linalg.det(U @ Vt)
    correction_matrix = np.eye(3)
    correction_matrix[2, 2] = np.sign(det)

    # Construct the rotation matrix
    rotation_matrix = U @ correction_matrix @ Vt

    return rotation_matrix
        
def w_from_xyz(xyz):
    '''
    Gets w from x, y, z of a quaternion, on assumption that it is a unit quaternion
    '''
    return np.sqrt(1 - xyz[0]**2 - xyz[1]**2 - xyz[2]**2)

def repeat_and_concatenate_points(x, n):
    '''
    Repeats last point n times and concatenates them
    '''
    x_repeat = np.repeat([x[-1]], repeats=n - x.shape[0], axis=0)
    x = np.vstack((x, x_repeat))
    return x
    
def sample_and_concatenate_points(x, n):
    '''
    Randomly sample n points and concatenates them
    '''
    if x.shape[0] > n/2:
        new_samples = x[np.random.choice(x.shape[0], n - x.shape[0], replace=False)]
    else:
        new_samples = x[np.random.choice(x.shape[0], n - x.shape[0], replace=True)]

    x = np.vstack((x, new_samples))
    return x

def homo_trans_mat_from_rot_trans(rot, trans):
    '''
    Returns a homogeneous transformation matrix combining rotation matrix and translation vector
    '''
    assert rot.shape == (3, 3)
    assert trans.shape == (3, 1)
    return np.hstack((np.vstack((rot, np.array([[0, 0, 0]]))), 
                      np.vstack((trans, np.array([[1]])))))
                
def homo_trans_mat_inv(homo_mat):
    '''
    Computes inverse of a homogeneous transformation matrix
    '''
    R = homo_mat[:3, :3].copy()
    v = np.expand_dims(homo_mat[:3, 3].copy(), axis=1)
    
    inv_homo_mat = np.vstack((np.hstack((R.T, -R.T@v)),
                              np.array([[0, 0, 0, 1]])))
    return inv_homo_mat
    
def convert_to_homo_coords(x):
    '''
    Converts an array of points to homogeneous coordinates
    '''
    return np.hstack((x, np.ones((x.shape[0], 1))))

def wrap_angle(angle):
    '''
    wraps angle to the range of [-180, 180]
    '''
    return ((angle + np.pi) % (2*np.pi)) - np.pi

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    rot = np.array([[0,  0, -1],
                    [1,  0,  0],
                    [0, -1,  0]])
    
    trans = np.array([[1], 
                      [1], 
                      [1]])

    homo_trans = homo_trans_mat_from_rot_trans(rot, trans)
    print("Homogeneous transformation matrix \n", homo_trans)
                    
    quat = quaternion_from_rotation(rot)
    print("Rotation Matrix to Quaternion \n", quat)

    inv_homo_trans = homo_trans_mat_inv(homo_trans)
    print("Inverse of Homogeneous transformation matrix \n", inv_homo_trans)
    
    point_in_frame1 = np.array([[2],
                                [3],
                                [4],
                                [1]])
    
    print("Point in frame 1: \n", point_in_frame1)
    
    point_in_frame2 = homo_trans@point_in_frame1
    print("Point in frame 2: \n", point_in_frame2)
    
    point_in_frame1 = homo_trans_mat_inv(homo_trans)@point_in_frame2
    print("Point back in frame 1: \n", point_in_frame1)
    
    x = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])
                 
    print("Before repeat and concatenate \n", x)
    
    repeated_x = repeat_and_concatenate_points(x, 5)
    print("After repeat and concatenate \n", repeated_x)
    
    sampled_x = sample_and_concatenate_points(x, 5)
    print("After sample and concatenate \n", sampled_x)
    
    print("---")
    # quaternion test
    roll = 0
    pitch = 0
    yaw = -2
    R = euler_to_rotation_matrix([roll, pitch, yaw])
    print("rotation before: ", R)
    
    quat = quaternion_from_rotation(R)
    print("quaternion: ", np.array([0.732, -0.188, 0.612, 0.232]))
    
    R = rotation_from_quaternion(quat)
    print("Rotation after: ", R)
    
    print("---")
    # axis angle test
    print("rotation before: ", R)
    
    axis_angle = rotation_matrix_to_axis_angle(R)
    print("axis angle: ", axis_angle)
    
    R = axis_angle_to_rotation_matrix(axis_angle[0], axis_angle[1])
    print("Rotation after: ", R)