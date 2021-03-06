import json
import numpy as np
from PIL import Image, ImageDraw
import os

def draw_boxes(I, bounding_boxes):
    # iterate through all boxes
    for [tl_row, tl_col, br_row, br_col] in bounding_boxes:
        draw = ImageDraw.Draw(I)
        draw.rectangle([tl_col, tl_row, br_col, br_row], outline=(36, 248, 229))
        del draw
    return I

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
        
    ex_good_4 = [328, 92, 194, 49, 10]
    ex_bad_5 = [0, 113, 148, 166, 42]
    # Uncomment for all files
    # for i in range(len(file_names)):
    for i in (ex_good_4 + ex_bad_5):
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names[i]))
        # draw box and save
        I = draw_boxes(I, bounding_boxes[file_names[i]])
        I.save(os.path.join(preds_path,file_names[i]))
    return 0

if __name__ == '__main__':
    main()
