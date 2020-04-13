import os
import sys
import numpy as np
import json
from PIL import Image, ImageDraw

THRESHOLD_SQ = 0.91  # threshold for square traffic light kernel
THRESHOLD_REC = 0.8  # threshold for rectangle traffic light kernel

# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = '../preds/hw01_preds'

def find_red_light_info(red_light, mult):
    # convert to numpy array
    red_light_arr = np.asarray(red_light)
    (height, width, n_chanels) = np.shape(red_light_arr)

    # flatten
    red_light_flat = np.reshape(red_light_arr, np.size(red_light_arr))

    # (left, top, right, down) distance from center to each edge
    center = [4, 4, 4, 4]
    # adjust accordingly to resize
    center = [int(mult*x) for x in center]
    center[2] = width - center[0]
    center[3] = height - center[1]
    print('m', mult, center)
    return {'img': red_light_arr,
            'flat': red_light_flat,
            'norm': np.linalg.norm(red_light_flat),
            'center': center}

def find_target_red_light(fn):
    red_light_lst = [] # list of red lights of different sizes
    # Open file to get target red light.
    img = Image.open(os.path.join(data_path, fn))

    # Find square red light.
    (left, top, right, bottom) = (315, 154, 323, 162)
    (width, height) = (right-left, bottom-top)
    red_light = img.crop((left, top, right, bottom))
    red_light.save('red_light_high.jpg')

    # Find full traffic light (rectangle).
    (left_rec, top_rec, right_rec, bottom_rec) = (315, 154, 323, 172)
    (width_rec, height_rec) = (right_rec-left_rec, bottom_rec-top_rec)
    red_light_rec = img.crop((left_rec, top_rec, right_rec, bottom_rec))
    red_light_rec.save('red_light_high_full.jpg')

    red_light_lst.append({'sq':find_red_light_info(red_light, 1),
                          'rec':find_red_light_info(red_light_rec, 1)})
    print('m', 1, ',Sq:', red_light.size, 'Rec', red_light_rec.size)

    # Different sizes for red light.
    mults = [1.5, 2, 2.5, 3, 3.5]
    for m in mults:
        # Resize images (sq and rec) and save for reference.
        red_light_img = red_light.resize((int(m*width), int(m*height)))
        red_light_img.save('red_light_high' + str(m) + '.jpg')

        red_light_img_rec = red_light_rec.resize((int(m*width_rec), int(m*height_rec)))
        red_light_img_rec.save('red_light_high_rec' + str(m) + '.jpg')

        red_light_lst.append({'sq':find_red_light_info(red_light_img, m),
                              'rec':find_red_light_info(red_light_img_rec, m)})
        print('m', m, ',Sq:', red_light_img.size, 'Rec:', red_light_img_rec.size)
    return red_light_lst

def find_similarity(r, c, I, red_light, max_fit):
    (n_rows,n_cols,n_channels) = np.shape(I)

    # Extract red light info.
    (box_height, box_width, box_channels) = np.shape(red_light['img'])
    red_light_flatten = red_light['flat']
    red_light_flatten_norm = red_light['norm']
    [left, top, right, bottom] = red_light['center']

    # Find indices of bounding box.
    tl_row = int(r-top)
    br_row = int(r+bottom)
    tl_col = int(c-left)
    br_col = int(c+right)
    # Check that indices are within range.
    if tl_row >= 0 and br_row < n_rows and tl_col >= 0 and br_col < n_cols:
        # Define bounding box
        I_box = I[tl_row:br_row,tl_col:br_col,:]
        assert np.shape(I_box) == np.shape(red_light['img'])

        # Flatten image within boudning box and find norm.
        I_box_flatten = np.float32(np.reshape(I_box, np.size(I_box)))
        I_box_flatten_norm = np.linalg.norm(I_box_flatten)

        # Find inner product and divide by product of norms.
        comp_val = np.dot(red_light_flatten, I_box_flatten) / (I_box_flatten_norm * red_light_flatten_norm)

        # If value meets threshold, return value and bounding box.
        if comp_val > max_fit:
            return (comp_val, [tl_row, tl_col, br_row, br_col])
    return None

def remove_overlaps(boxes):
    # Iterate over all boxes.
    j = 0
    while j < len(boxes):
        ([tl_row, tl_col, br_row, br_col], fit) = boxes[j]
        # Iterate over all boxes other than boxes[j].
        k = 0
        while True:
            # Check whether boxes[k] overlaps with boxes[j].
            if k != j \
            and ((boxes[k][0][1] >= tl_col and boxes[k][0][1] <= br_col \
              and boxes[k][0][0] >= tl_row and boxes[k][0][0] <= br_row) \
              or (boxes[k][0][1] >= tl_col and boxes[k][0][1] <= br_col \
              and boxes[k][0][2] >= tl_row and boxes[k][0][2] <= br_row) \
              or (boxes[k][0][3] >= tl_col and boxes[k][0][3] <= br_col \
              and boxes[k][0][0] >= tl_row and boxes[k][0][0] <= br_row) \
              or (boxes[k][0][3] >= tl_col and boxes[k][0][3] <= br_col \
              and boxes[k][0][2] >= tl_row and boxes[k][0][2] <= br_row)):
                # Remove box with smaller value.
                if fit >= boxes[j][1]:
                    boxes.pop(k)
                    # Adjust indices.
                    if k < j:
                        j -= 1
                else:
                    boxes.pop(j)
                    break
            else:
                k += 1
            # Check whether we've reached the end of the list.
            if k >= len(boxes):
                j += 1
                break
    # Convert list of tuples (inds, val) to just list of inds.
    bounding_boxes = [inds for (inds, fit) in boxes]
    return bounding_boxes

def detect_red_light(I, red_light_lst):
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
    # This should be a list of tuples, each containing
    # (1) a list of indices of length 4 and (2) the fit of that bounding box.
    bounding_boxes_fits = []

    (n_rows,n_cols,n_channels) = np.shape(I)

    # Find pixels in the top half of image with red value greater than 200.
    # These pixels should include any bright lights.
    light_inds = np.argwhere(I[:n_rows//2,:,0]>200)

    # Iterate over the bright light pixels to find which ones are the centers
    # of red lights.
    for [r, c] in light_inds:
        box = None
        max_fit = THRESHOLD_SQ
        # Find which size of light best fits.
        for red_light in red_light_lst:
            # Use perceptron method to find whether fits square img of red light.
            sim = find_similarity(r, c, I, red_light['sq'], max_fit)

            # If there's a fit, confirm that it's a red light with full
            # outline of traffic light.
            if sim != None \
            and find_similarity(r, c, I, red_light['rec'], THRESHOLD_REC) != None:
                (max_fit, box) = sim
        if box != None:
            bounding_boxes_fits.append((box, max_fit))

    # Find local maxima of overlapping boxes.
    bounding_boxes = remove_overlaps(bounding_boxes_fits)
    for box in bounding_boxes:
        assert len(box) == 4

    return bounding_boxes

def main():
    print('Using data from', data_path)
    print('Saving predictions to', preds_path)
    os.makedirs(preds_path,exist_ok=True) # create directory if needed

    # get sorted list of files:
    file_names = sorted(os.listdir(data_path))

    # remove any non-JPEG files:
    file_names = [f for f in file_names if '.jpg' in f]

    # get red light image
    fn = 'RL-001.jpg'
    print('\nFinding target red lights from', fn)
    red_light_lst = find_target_red_light(fn)

    preds = {}
    # for i in range(len(file_names)):
    for i in range(3):

        # read image using PIL:
        I0 = Image.open(os.path.join(data_path,file_names[i]))

        # convert to numpy array:
        I = np.asarray(I0)
        preds[file_names[i]] = detect_red_light(I, red_light_lst)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds.json'),'w') as f:
        json.dump(preds,f)
    return 0

if __name__ == '__main__':
    main()
