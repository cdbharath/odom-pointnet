'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import numpy as np
import cv2
from scipy.spatial import KDTree as KDTree

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)


def fill_in_nearest_bfs(image):
    '''
    Fills the nan values with the nearest (based on bfs) valid pixels
    '''
    
    def nearest_bfs(x, y, image):
        neighbors = [[-1, 0], [0, -1], [0, 1], [1, 0], [1, -1], [-1, -1], [-1, 1], [1, 1]]
        queue = [[x, y]]
        
        while len(queue):
            cur_x, cur_y = queue.pop(0)
            
            if image[cur_x][cur_y]:
                print(x, y, cur_x, cur_y)
                return image[cur_x][cur_y]
            
            for neighbor in neighbors:
                x = cur_x + neighbor[0]
                y = cur_y + neighbor[1]
                if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    queue.append([x, y])
        
        return image[x][y]
    
    # Calculate the valid distances of the entire matrix 
    invalid_idx = np.argwhere(image == 0)
 
    nearest_val = []
    for i in range(invalid_idx.shape[0]):
        nearest_val_ = nearest_bfs(invalid_idx[i][0], invalid_idx[i][1], image)
        nearest_val.append(nearest_val_)

    nearest_val = np.array(nearest_val)
    image[invalid_idx[:, 0], invalid_idx[:, 1]] = nearest_val

    return image


def fill_in_nearest(image):
    '''
    Fills the nan values with the nearest (based on distance) valid pixels
    TODO this might not be very effecient
    '''

    def nearest_to_nan(x, y, r, c):
        min_idx = ((r - x)**2 + (c - y)**2).argmin()
        return r[min_idx], c[min_idx]
        
    # Calculate the valid distances of the entire matrix 
    invalid_idx = np.argwhere(image == 0)
    valid_idx = np.argwhere(image != 0)
 
    nearest_val = []
    for i in range(invalid_idx.shape[0]):
        nearest_x, nearest_y = nearest_to_nan(invalid_idx[i][0], invalid_idx[i][1], valid_idx[:, 0], valid_idx[:, 1])
        nearest_val.append(image[nearest_x, nearest_y])

    nearest_val = np.array(nearest_val)

    # Using only numpy operations (even more ineffecient)
    # x_diff = np.expand_dims(valid_idx[: , 0], 0) - np.expand_dims(invalid_idx[:, 0], 1) 
    # y_diff = np.expand_dims(valid_idx[: , 1], 0) - np.expand_dims(invalid_idx[:, 1], 1) 

    # d = np.absolute(x_diff) + np.absolute(y_diff)
    # min_pidx = d.argmin(axis=1)
    # nearest_val = image[valid_idx[:, 0][min_pidx], valid_idx[:, 1][min_pidx]]

    image[invalid_idx[:, 0], invalid_idx[:, 1]] = nearest_val

    return image

    
def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.
    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE
    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def cv_inpaint(image, row_scale=64, col_scale=1024, blur=False):
    '''
    Use opencv inpaint for depth completion
    '''
    
    mask = np.full([row_scale, col_scale], 0, dtype=np.uint8)
    mask[image == 0] = 1
    image = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
    
    if blur:
        image = cv2.bilateralFilter(image, 3, 75, 75)

    return image
    

def cv_dilate(image):
    '''
    Use opencv dilate for depth completion
    '''
    
    return cv2.dilate(image, FULL_KERNEL_5)
    
    
def nearest_kdtree(image):
    '''
    replaces nonzero pixels with the nearest nonzero pixel
    '''
    
    valid_pixels = np.argwhere(image != 0)
    invalid_pixels = np.argwhere(image == 0)
    
    kdtree = KDTree(valid_pixels)
    _, pre_indices = kdtree.query(invalid_pixels, k=1)
    
    indices = valid_pixels[pre_indices]
    image[invalid_pixels[:, 0], invalid_pixels[:, 1]] = image[indices[:, 0], indices[:, 1]]
    
    return image
    
if __name__ == "__main__":
    image = np.array(([1, 2, 4, 0, 1], 
                      [3, 5, 2, 1, 2], 
                      [0, 4, 6, 3, 1]))
    
    image = nearest_kdtree(image)
    print(image)