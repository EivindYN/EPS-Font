"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

import os
import copy

from skimage import img_as_uint, io as ioo
from skimage.filters import threshold_otsu
from skimage.morphology import area_closing, remove_small_holes, flood_fill, convex_hull_object, closing
from skimage.morphology import skeletonize_3d, binary_closing
from skimage.morphology import square
import os.path
from os import path
import torchvision.transforms as transforms
from util.DTS.util import *

def load_image(img_file):
    image = Image.open(img_file)
    image = image.convert("L")
    img = np.array(image)
    return img

def load_image_rgb(img_file):
    image = Image.open(img_file)
    image = image.convert('RGB')
    img = np.array(image)
    return img

def load_image_tensor(img_file):
    image = Image.open(img_file)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5), std = (0.5))])
    img = transform(image)
    img = np.array(img).reshape(1,1,64,64)
    return img

ColorDict = {0: [-1,-1,-1], 0.1: [0,0,1], 0.2: [0.5,0,0.5], 0.3:[0,1,0], 0.4:[0,1,1], 0.5:[1,0,0], 0.6:[1,1,0], 0.7:[0.7,0.7,0.7], 0.8: [1, 0, 1], 0.9:[0.3,0.3,0.3], 1:[1,1,1]}

def image_to_tensor(img):
    use_color_dict = img.shape[-1] != 3
    img = Image.fromarray((img * 255).astype(np.uint8)) # * 255 if skeleton
    #img = img.convert("L")
    img = img.convert('RGB')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5), std = (0.5))])
    img = transform(img)
    img = np.array(img).reshape(1,3,64,64)
    if use_color_dict:
        for y in range(64):
            for x in range(64):
                img[0,0:3,x,y] = ColorDict[neg1_1_to_0_1_round(img[0,0,x,y])]
    return img


def make_path(name):
    psplit = name.replace("\\","/").split("/")
    if len(psplit) == 0:
        return
    ppath = "/".join(psplit[:-1])
    if not path.exists(ppath):
        make_path(ppath)
        os.mkdir(ppath)

def save_image(img_array, name):
    im = Image.fromarray(img_array * 255) # * 255 if skeleton
    im = im.convert("L")
    make_path(name)
    im.save(name)

def save_image_debug(img_array, name):       
    n = 64
    image = np.array([[np.array([1,1,1]) for _ in range(n)] for _ in range(n)])
    for y in range(64):
        for x in range(64):
            image[y][x] = (np.array(ColorDict[round(img_array[y][x], 1)]) + 1) * 255/2
    im = Image.fromarray(image.astype(np.uint8)) # * 255 if skeleton     
    make_path(name)
    im.save(name)

def save_image_rgb(img_array, name):
    im = Image.fromarray((img_array * 255).astype(np.uint8)) # * 255 if skeleton
    #im = im.convert("rgb")
    make_path(name)
    im.save(name)

def get_binary_no_thresh(img):
    binary = img > 254
    return binary

def get_binary(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary

def get_binary_area(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    #binary = closing(binary) #Removes small parts, like in a_special4 down-right and the line there too
    #binary = closing(binary, square(4)) #Removes small parts, like in a_special4 down-right and the line there too
    binary = np.invert(binary)
    
    binary = closing(binary) #Removes small parts, like in a_special4 down-right and the line there too
    binary = closing(binary, square(2)) #Removes small parts, like in a_special4 down-right and the line there too
    #binary = area_closing(binary) #Removes small dots, like in a_special4 down-right
    binary = np.invert(binary)
    #binary = closing(binary) #Removes small parts, like in a_special4 down-right and the line there too
    #binary = area_closing(binary) #Removes small dots, like in a_special4 down-right
    return binary

def skeleton(img):
    img = img / 255
    image = get_binary(invert(img))

    # Skeletonize (otsu + skeletonize_3d)
    skeleton = skeletonize_3d(image)
    skeleton = binary_closing(skeleton)

    skeleton = get_binary(invert(skeleton))
    return skeleton

def invert(img):
    imgnew = copy.deepcopy(img)
    for x in range(len(img)):
        for y in range(len(img[0])):
            imgnew[x][y] = 1.0 - img[x][y]
    return imgnew

def layer_depth_2_info(img):
    layer_depth_img, depth_count = layer_depth(img)
    adjusted_img = copy.deepcopy(img)
    for y in range(64):
        for x in range(64):
            adjusted_img[y][x] = 0 if layer_depth_img[(x,y)] == 2 else 1
    return adjusted_img, depth_count[2] == 0

def layer_depth_invert(img):
    layer_depth_img, _ = layer_depth(img)
    adjusted_img = copy.deepcopy(img)
    for y in range(64):
        for x in range(64):
            if layer_depth_img[(x,y)] == 0:
                adjusted_img[y][x] = 1
            else:
                adjusted_img[y][x] = layer_depth_img[(x,y)] % 2
    return adjusted_img

def layer_depth(img):
    next_queue = [(0,0), (63,0), (0, 63), (63, 63)]
    for pos in next_queue:
        x,y = pos
        if not img[y][x]:
            raise Exception("not white corner")
    queue = []
    depth_pos = {}
    depth_count = defaultdict(int)
    depth_counter = -1
    while next_queue:
        queue = copy.copy(next_queue)
        next_queue = []
        depth_counter += 1
        while queue:
            pos = queue.pop()
            x,y = pos
            for adj_pos in adjacent(pos):
                if out_of_bound(adj_pos) or adj_pos in depth_pos:
                    continue
                x2, y2 = adj_pos
                if img[y2][x2] == img[y][x]:
                    queue.append(adj_pos)
                    depth_pos[adj_pos] = depth_counter
                    depth_count[depth_counter] += 1
                else:
                    next_queue.append(adj_pos)
                    depth_pos[adj_pos] = depth_counter + 1
                    depth_count[depth_counter + 1] += 1
    return depth_pos, depth_count