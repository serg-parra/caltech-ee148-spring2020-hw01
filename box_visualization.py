# Bounding Box Visualization

import copy

box_th = 5 #bounding boxes thickness (in pixels)
I_copy = copy.deepcopy(I) #duplicates the image analyzed

for i in range(len(bounding_boxes)):
    current = bounding_boxes[i] #selects one of the bounding boxes

    tl = current[0:2] #pulls out the top left corner
    br = current[2:4] #pulls out the bottom right corner
    
    og_box = I[tl[0]:br[0], tl[1]:br[1], :] #saves the targeted part of the original image
    
    I_copy[tl[0]-box_th:br[0]+box_th, tl[1]-box_th:br[1]+box_th, :] = 0 # creates a black box around the selected bounding box
    I_copy[tl[0]-box_th:br[0]+box_th, tl[1]-box_th:br[1]+box_th, 1] = 255 # creates a green box around the selected bounding box
    I_copy[tl[0]:br[0], tl[1]:br[1], :] = og_box #replaces the bounding box with the part of the original image
    
plt.imshow(I_copy)

plt.savefig('failure.png', dpi=300)