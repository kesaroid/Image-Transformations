# Kesar TN
# University of Central Florida
# kesar@Knights.ucf.edu

# I shall only be using two libraries, namely
# PIL for reading the image, and
# Numpy for matrix manipulations
from PIL import Image
import numpy as np
from scipy.interpolate import interp2d

# The function rotation() when called rotates the image
def rotation(image):
    # The argument image is read using PIL and is converted to a numpy array
    img = Image.open(image)
    img = np.array(img)

    # We define a rotation matrix
    rotation_matrix = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]])

    # Create an empty canvas on which the image can rotate
    img_transformed = np.empty((232, 217, 3), dtype=np.uint8)

    # unwrap the each pixel of the image and store it as coordinates
    for i, row in enumerate(img):
        for j, _ in enumerate(row):
            image_data = img[i, j, :]
            coordinates = np.array([i, j, 1])
            # Matrix multiply rotation matrix with the coordinates
            i_out, j_out, __ = rotation_matrix @ coordinates
            img_transformed[i_out, j_out, :] = image_data

    # Display the rotated image 
    img_transformed = Image.fromarray(img_transformed, 'RGB')
    img_transformed.save('output1.png')
    img_transformed.show()


# The function scaling() when called scales the image
def scaling(image):
    # The argument image is read using PIL and is converted to a numpy array
    img = Image.open(image)
    img = np.array(img)

    # We define a scaling matrix
    scaling_matrix = np.array( [[2, 0, 0],
                                [0, 2, 0],
                                [0, 0, 1]])

    # Create an empty canvas on which the image can scale
    # img_transformed = np.empty((434, 464, 3), dtype=np.uint8)
    img_transformed = np.full((434, 464, 3), 255, dtype=np.uint8)

    # unwrap the each pixel of the image and store it as coordinates
    for i, row in enumerate(img):
        for j, _ in enumerate(row):
            image_data = img[i, j, :]
            coordinates = np.array([i, j, 1])
            # Matrix multiply scaled matrix with the coordinates
            i_out, j_out, __ = scaling_matrix @ coordinates
            img_transformed[i_out, j_out, :]  = image_data
            
    
    interpolated = np.empty((434, 464, 3), dtype=np.uint8)
    for i, row in enumerate(img_transformed):
        for j, _ in enumerate(row):
            interpolated[i, j, :] = nearest_neighbors(i, j, img, scaling_matrix)

    # Display the scaled image
    img_transformed = Image.fromarray(interpolated, 'RGB')
    img_transformed.save('output21.png')
    img_transformed.show()

# The function translation() when called translates the image
def translation(image):
    # The argument image is read using PIL and is converted to a numpy array
    img = Image.open(image)
    img = np.array(img)

    # Define a translation matrix
    translation_matrix = np.array( [[1, 0, 100],
                                    [0, 1, 50],
                                    [0, 0, 1]]) 

    # Create an empty canvas on which the image can translate
    img_transformed = np.empty((417, 432, 3), dtype=np.uint8)

    # unwrap the each pixel of the image and store it as coordinates
    for i, row in enumerate(img):
        for j, _ in enumerate(row):
            image_data = img[i, j, :]
            coordinates = np.array([i, j, 1])
            # Matrix multiply translation matrix with the coordinates
            i_out, j_out, __ = translation_matrix @ coordinates
            img_transformed[i_out, j_out, :] = image_data

    # Display the transformed image 
    img_transformed = Image.fromarray(img_transformed, 'RGB')
    img_transformed.save('output3.png')
    img_transformed.show()

def nearest_neighbors(i, j, image, scaling_matrix):
    scaling_matrix = np.linalg.inv(scaling_matrix)
    x_max, y_max = image.shape[0] - 1, image.shape[1] - 1
    x, y, _ = scaling_matrix @ np.array([i, j, 1])
    
    if np.floor(x) == x and np.floor(y) == y:
        x, y = int(x), int(y)
        return image[x, y]
    
    if np.abs(np.floor(x) - x) < np.abs(np.ceil(x) - x):
        x = int(np.floor(x))
    
    else:
        x = int(np.ceil(x))
    
    if np.abs(np.floor(y) - y) < np.abs(np.ceil(y) - y):
        y = int(np.floor(y))
    
    else:
        y = int(np.ceil(y))
    
    if x > x_max:
        x = x_max
    
    if y > y_max:
        y = y_max
    
    return image[x, y,]


rotation('image.png')
scaling('image.png')
translation('image.png')