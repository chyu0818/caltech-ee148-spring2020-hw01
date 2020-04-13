import os
import sys
import numpy as np
import json
from PIL import Image, ImageDraw

THRESHOLD = 0.92
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = '../preds/hw01_preds'

def find_red_light_info(red_light, is_small, mult):
    red_light_arr = np.asarray(red_light)
    (height, width, n_chanels) = np.shape(red_light_arr)
    red_light_flat = np.reshape(red_light_arr, np.size(red_light_arr))
    #(left, top, right, bottom)
    # center = [26, 14, 26, 58]
    center = [4, 6, 4, 6]
    if is_small:
        # center = [15, 17, 15, 53]
        center = [4, 4, 4, 4]
    center = [int(np.ceil(mult*x)) for x in center]
    center[2] = width - center[0]
    center[3] = height - center[1]
    print(mult, center)
    return {'img': red_light_arr,
            'flat': red_light_flat,
            'norm': np.linalg.norm(red_light_flat),
            'center': center}

def find_target_red_light():
    red_light_lst = []
    fn = 'RL-001.jpg'
    img = Image.open(os.path.join(data_path, fn))

    # high
    # (left, top, right, bottom) = (314, 152, 324, 162) outlined square
    # no outline square (315, 154, 323, 162)
    # 335, 41
    # (left, top, right, bottom) = (320, 24, 350, 94)
    # (left, top, right, bottom) = (314, 152, 324, 162)
    (left, top, right, bottom) = (315, 154, 323, 162)
    (width, height) = (right-left, bottom-top)
    red_light = img.crop((left, top, right, bottom))
    red_light.save('red_light_high.jpg')
    red_light_lst.append(find_red_light_info(red_light, True, 1))
    print(red_light.size)

    # low
    # (left1, top1, right1, bottom1) = (417, 189, 429, 201) outlined square
    # no outline square (418, 190, 428, 200) width 10, height 30
    # 147, 28
    # (left1, top1, right1, bottom1) = (121, 14, 173, 86)
    # (left1, top1, right1, bottom1) = (417, 189, 429, 201)
    # (width1, height1) = (right1-left1, bottom1-top1)
    # red_light1 = img.crop((left1, top1, right1, bottom1))
    # red_light1.save('red_light_low.jpg')
    # red_light_lst.append(find_red_light_info(red_light1, False, 1))
    # print(red_light1.size)

    mults = [1.5, 2, 2.5, 3, 3.5]
    # mults = [0.2, 0.3, 0.5, 0.7, 1]
    for m in mults:
        red_light_img = red_light.resize((int(m*width), int(m*height)))
        red_light_img.save('red_light_high' + str(m) + '.jpg')
        red_light_lst.append(find_red_light_info(red_light_img, True, m))
        print(m, red_light_img.size)

        # red_light_img1 = red_light1.resize((int(m*width1), int(m*height1)))
        # red_light_img1.save('red_light_low' + str(m) + '.jpg')
        # red_light_lst.append(find_red_light_info(red_light_img1, False, m))
        # print(m, red_light_img1.size)
    return red_light_lst

def detect_red_light0(I, red_light):
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

            if comp_val > THRESHOLD:
                bounding_boxes.append([r, c, r+box_height, c+box_width])

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

def detect_red_light1(I, red_light_lst):
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
    # traffic light r above 200


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    (n_rows,n_cols,n_channels) = np.shape(I)

    light_inds = np.argwhere(I[:n_rows//2,:,0]>200)

    for [r, c] in light_inds:
        max_fit = 0.
        box = None
        for red_light in red_light_lst:
            (box_height, box_width, box_channels) = np.shape(red_light['img'])
            red_light_flatten = red_light['flat']
            red_light_flatten_norm = red_light['norm']
            [left, top, right, bottom] = red_light['center']
            # check even
            tl_row = int(r-top)
            br_row = int(r+bottom)
            tl_col = int(c-left)
            br_col = int(c+right)
            if tl_row >= 0 and br_row < n_rows and tl_col >= 0 and br_col < n_cols:
                I_box = I[tl_row:br_row,tl_col:br_col,:]
                assert np.shape(I_box) == np.shape(red_light['img'])
                I_box_flatten = np.float32(np.reshape(I_box, np.size(I_box)))
                I_box_flatten_norm = np.linalg.norm(I_box_flatten)
                comp_val = np.dot(red_light_flatten, I_box_flatten) / (I_box_flatten_norm * red_light_flatten_norm)

                if comp_val > THRESHOLD and comp_val > max_fit:
                    box = [tl_row, tl_col, br_row, br_col]
                    max_fit = comp_val
        if box != None:
            bounding_boxes.append(box)

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))
print(len(file_names))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# get red light image
red_light_lst = find_target_red_light()

preds = {}
for i in range(len(file_names)):
# for i in range(0,10):

    # read image using PIL:
    I0 = Image.open(os.path.join(data_path,file_names[i]))
    # I0 = Image.open(os.path.join(data_path,'RL-002.jpg'))

    # convert to numpy array:
    I = np.asarray(I0)
    preds[file_names[i]] = detect_red_light1(I, red_light_lst)
    # preds['RL-002.jpg'] = detect_red_light1(I, red_light_lst)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
