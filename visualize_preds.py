import json
import numpy as np
from PIL import Image
import os

def draw_boxes(I, bounding_boxes):
    outline_rgb = [36, 248, 229]
    (n_rows,n_cols,n_channels) = np.shape(I)
    I_new = np.copy(I)

    # iterate through all boxes
    for [tl_row,tl_col,br_row,br_col] in bounding_boxes:
        I_new[tl_row:br_row+1,(tl_col,br_col),:] = outline_rgb
        I_new[(tl_row,br_row),tl_col+1:br_col,:] = outline_rgb
    return I_new

def main():
    # set the path to the downloaded data:
    data_path = '../data/RedLights2011_Medium'

    # set a path for saving predictions:
    preds_path = '../preds/hw01_preds'

    # get sorted list of files:
    file_names = sorted(os.listdir(data_path))

    # remove any non-JPEG files:
    file_names = [f for f in file_names if '.jpg' in f]

    # get predictions
    with open(os.path.join(preds_path,'preds.json')) as f:
        bounding_boxes = json.load(f)

    # for i in range(len(file_names)):
    for i in range(10):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names[i]))

        # convert to numpy array:
        I = np.asarray(I)

        I_new = draw_boxes(I, bounding_boxes[file_names[i]])
        I_img = Image.fromarray(I_new)
        I_img.save(os.path.join(preds_path,file_names[i]))
    return

if __name__ == '__main__':
    main()
