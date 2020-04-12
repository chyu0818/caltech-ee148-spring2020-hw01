import os
import numpy as np
import json
from PIL import Image

# full: [153, 315, 172, 323]
# small: [153, 314, 162, 324]
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# get red light image
red_light_fn0 = 'RL-001.jpg'
red_light_img0 = Image.open(os.path.join(data_path, red_light_fn0))
# [tl_row, tl_col, br_row, br_col] = [153, 314, 162, 324]
# red_light = np.asarray(red_light_img0)[tl_row:br_row+1,tl_col:br_col+1,:]
# red_light_img = Image.fromarray(red_light)
# red_light_img.save('../data/RedLight.jpg')

(left, top, right, bottom) = (314, 153, 324, 162)
red_light_img = red_light_img0.crop((left, top, right, bottom))
red_light_img.save('red_light.jpg')

size_small = ((right - left) // 2, (bottom - top) // 2)
red_light_img_small = red_light_img.resize(size_small)

size_big = ((right - left) * 2, (bottom - top) * 2)
red_light_img_big = red_light_img.resize(size_big)
