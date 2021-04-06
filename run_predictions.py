import os
import numpy as np
import json
from PIL import Image

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    all_boxes = [] #initialize bounding box coordinate storage
    np_dot = [] #initialize store dot products
    img_dims = I.shape #saves the original image dimensions
    
    owd = os.getcwd() #saves original working directory

    os.chdir('../RedLights2011_Medium')
    train_img = Image.open('RL-010.jpg') #imports image with PIL functionality
    np_train_img = np.array(train_img) #converts PIL image to NumPy array
    os.chdir(owd)

    np_kernel_img = np_train_img[26:52,321:350,:]
    #manually sub-samples image to capture dimensions of kernel image showing a red light 
    # whole red light: [24:94,319:351,:]
    # just red light: [30:49,326:345,:]

    np_kernel_dims = np_kernel_img.shape #saves the dimensions of the kernel image

    # Flattening the kernel image and normalizing the resulting vector

    np_kernel_flat = np_kernel_img.ravel() #flattens the 3D NumPy array into a 1D array

    l_2 = (np.linalg.norm(np_kernel_flat, 2)) #calculates the l2 normalization value (as suggested in class), 2 is the order 
    #l2 means summing the square of all the vector elements equals 1

    if l_2 == 0:
        l_2 = 1 #prevents division by zero
        
    np_kernel_normed = np_kernel_flat / l_2 #normalizes the 1D array

    # Image Flattening and Normalization

    for r in range(img_dims[0]-np_kernel_dims[0]):
        for c in range(img_dims[1]-np_kernel_dims[1]):
            box_coords = [r, c,r+np_kernel_dims[0],c+np_kernel_dims[1]] #stores the bounding box coordinates in the specified format
            img_subsample = I[r:r+np_kernel_dims[0],c:c+np_kernel_dims[1],:] #subsamples a frame from the image

            img_flat = img_subsample.ravel() #flattens the 3D NumPy array into a 1D array
        
            img_l_2 = (np.linalg.norm(img_flat, 2)) #calculates the l2 normalization value, 2 is the order 
            
            if img_l_2 == 0:
                img_l_2 = 1 #prevents division by zero
            img_normed = img_flat / img_l_2 #normalizes the 1D array

            #Kernel Convolution & Thresholding

            np_dotprod = np.dot(img_normed, np_kernel_normed) #produces the dot product of image and kernel
            np_dot.append(np_dotprod) #stores the dot product
            all_boxes.append(box_coords) # stores the box coordinates
        
    dot_mean = np.mean(np_dot) #calculates the mean dot product
    dot_std = np.std(np_dot) #calculates the dot product standard deviation

    all_boxes = np.array(all_boxes) #converts list to NumPy array
    np_dot = np.array(np_dot) #converts list to NumPy array
    bounding_boxes = all_boxes[np_dot > 0.89]
    bounding_boxes = bounding_boxes.tolist() #converts NumPy array back to list for export
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = '../RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
