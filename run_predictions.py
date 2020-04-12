import os
import numpy as np
import json
from PIL import Image

THRESHOLD = 0.9

def detect_red_light(I, red_light):
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

    (n_rows,n_cols,n_channels) = np.shape(I)
    print(np.shape(I))
    (box_height,box_width,box_channels) = np.shape(red_light)
    red_light_flatten = np.float32(np.reshape(red_light, (np.size(red_light),)))
    red_light_flatten_norm = np.linalg.norm(red_light_flatten)
    print(np.shape(red_light))
    assert n_channels == box_channels

    r_inc = 1
    c_inc = 1

    # Only need to search top half of image
    for r in range(0,n_rows//2-box_height,r_inc):
        for c in range(0,n_cols-box_width,c_inc):
            I_box = I[r:r+box_height,c:c+box_width,:]
            assert I_box.shape == red_light.shape
            I_box_flatten = np.float32(np.reshape(I_box, np.size(I_box)))
            I_box_flatten_norm = np.linalg.norm(I_box_flatten)
            comp_val = np.dot(red_light_flatten, I_box_flatten) / (I_box_flatten_norm * red_light_flatten_norm)

            # if r == 153 and c == 314:
            #     print(I_box_flatten)
            #     print(I_box_flatten.shape)
            #     print(red_light_flatten)
            #     print(red_light_flatten.shape)
            #     print('truth', comp_val)
            #     print(np.inner(red_light_flatten, I_box_flatten))
            #     print((I_box_flatten_norm * red_light_flatten_norm))
            if comp_val > THRESHOLD:
                # print(comp_val)
                bounding_boxes.append([r, c, r+box_height, c+box_width])

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = '../preds/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))
print(len(file_names))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# # get file of red light
# red_light_fn = '../data/RedLight.jpg'
# red_light_img = Image.open(red_light_fn)
#
# # convert to numpy array:
# red_light = np.asarray(red_light_img)

# get red light image
red_light_lst = []
red_light_fn = 'RL-001.jpg'
red_light_img = Image.open(os.path.join(data_path, red_light_fn))
(left, top, right, bottom) = (314, 153, 324, 162)
(width, height) = (right-left, bottom-top)
red_light = red_light_img.crop((left, top, right, bottom))
red_light.save('red_light.jpg')
red_light_lst.append(red_light)

# red_light_img_small = red_light.resize((width//2, height//2))
# red_light_img_small.save('red_light_half.jpg')
# red_light_lst.append(red_light_img_small)

# red_light_img_twothirds = red_light.resize(((2*width)//3, (2*height)//3))
# red_light_img_twothirds.save('red_light_twothirds.jpg')
# red_light_lst.append(red_light_img_twothirds)

red_light_img15 = red_light.resize((int(1.5*width), int(1.5*height)))
red_light_img15.save('red_light15.jpg')
red_light_lst.append(red_light_img15)

red_light_img2 = red_light.resize((2*width, 2*height))
red_light_img2.save('red_light2.jpg')
red_light_lst.append(red_light_img2)

red_light_img25 = red_light.resize((int(2.5*width), int(2.5*height)))
red_light_img25.save('red_light25.jpg')
red_light_lst.append(red_light_img25)

red_light_img3 = red_light.resize((3*width, 3*height))
red_light_img3.save('red_light3.jpg')
red_light_lst.append(red_light_img3)
# [tl_row, tl_col, br_row, br_col] = [153, 314, 162, 324]
# red_light = np.asarray(red_light_img0)[tl_row:br_row+1,tl_col:br_col+1,:]
# red_light_img = Image.fromarray(red_light)
# red_light_img.save(os.path.join(preds_path,'hi1.jpg'))

preds = {}
# for i in range(len(file_names)):
for i in range(10):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I, red_light)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
