import os
import numpy as np
import json
from PIL import Image, ImageDraw

# full: [153, 315, 172, 323]
# small: [153, 314, 162, 324]
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# get red light image
red_light_fn0 = 'RL-010.jpg'
red_light_img0 = Image.open(os.path.join(data_path, red_light_fn0))
# [tl_row, tl_col, br_row, br_col] = [153, 314, 162, 324]
# red_light = np.asarray(red_light_img0)[tl_row:br_row+1,tl_col:br_col+1,:]
# red_light_img = Image.fromarray(red_light)
# red_light_img.save('../data/RedLight.jpg')

# outlined rectangle (318, 24, 352, 95)
# no outlined rectangle(320, 24, 350, 94)
(left, top, right, bottom) = (320, 24, 350, 94)
red_light_img = red_light_img0.crop((left, top, right, bottom))
red_light_img.save('red_light.jpg')

# low light (416, 188, 430, 202)
# outlined rectangle big light  (119, 11, 175, 89)
# no outlined recatangle big light (121, 14, 173, 86)
(left, top, right, bottom) = (121, 14, 173, 86)
red_light_img = red_light_img0.crop((left, top, right, bottom))
red_light_img.save('red_light1.jpg')

# [tl_col, tl_row, br_col, br_row] = [419, 191, 429, 201]
# draw = ImageDraw.Draw(red_light_img0)
# draw.rectangle([tl_col, tl_row, br_col, br_row], outline=(36, 248, 229))
# del draw
# red_light_img0.save('red_light1.jpg')
