import multiprocessing as mul
import os

import torch.utils.data as data

from PIL import Image
import os
import os.path
from util.DTS.imgproc import *
from util.DTS.image import Image
import numpy as np
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]
from enum import Enum
import util.util
ImgMode = Enum('ImgMode', 'NORMAL OUTLINE INVERT')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def make_skeleton_images():
    for language in ["source"]:#"source", "english", "chinese"]:        
        optdataroot = "./datasets/font"
        optphase = "train"
        optmax_dataset_size = float("inf")
        dataroot = os.path.join(optdataroot, optphase, language)  # get the image directory
        paths = sorted(make_dataset(dataroot, optmax_dataset_size))  # get image paths
        for img in paths:
            #skeleton_image = get_binary_area(load_image(img))
            #if language == "source":
            #    raise Exception("binary_area shouldn't be used")
            output_path = img.replace("datasets/font", "datasets/font_offset_color")
            if os.path.exists(output_path):
                continue
            skeleton_image = skeleton(get_binary(load_image(img)))
            save_image(skeleton_image, output_path)


def areaAllFonts():
    for language in ["english"]:#"source", "english", "chinese"]:
        optdataroot = "./datasets/font"
        optphase = "train"
        optmax_dataset_size = float("inf")
        dataroot = os.path.join(optdataroot, optphase, language)  # get the image directory
        paths = sorted(make_dataset(dataroot, optmax_dataset_size))  # get image paths
        for img in paths:
            save_image(get_binary_area(load_image(img)[:64,:64]), img.replace("datasets/font", "datasets/font_area"))

def get_source_skeleton_path(path):
    character = path.split(os.sep)[-1]
    source = "./datasets/font_offset_color/train/source" #FIX font instead of font_same_cut
    return os.path.join(source, character)

def posneg_to_pos0(num):
    return round(((num + 1) * 5))/10

def pos0_to_posneg(num):
    return num*2 - 1

def resizeAllFonts():
    for language in ["english"]:#"source", "english", "chinese"]:
        optdataroot = "./datasets/font"
        optphase = "train"
        optmax_dataset_size = float("inf")
        dataroot = os.path.join(optdataroot, optphase, language)  # get the image directory
        paths = sorted(make_dataset(dataroot, optmax_dataset_size))  # get image paths
        for img_path in paths:
            print(img_path)
            source_skeleton_path = get_source_skeleton_path(img_path)
            source_skeleton_img = get_binary(load_image(source_skeleton_path))
            character_img = get_binary_area(load_image(img_path))
            image = Image(source_skeleton_img, character_img)
            image.setup()
            info = image.auto_morph()
            save_image(image.make_image(image.pixels, False), img_path.replace("datasets/font_same_cut", "datasets/font_resize_cut"))
            #save_image(character_img, img_path.replace("datasets/font_test_cut", "datasets/font_resize_cut"))
            print(pos0_to_posneg(np.array(info)))

def offset_skeleton(img_path):
    debug = False
    if debug:
        name = img_path.replace("datasets/font", "datasets/font_offset_debug")
    else:
        name = img_path.replace("datasets/font", "datasets/font_offset")
    if os.path.exists(name):
        return
    print(name)
    source_skeleton_path = get_source_skeleton_path(img_path)
    source_skeleton_img = get_binary(load_image(source_skeleton_path))
    goal_img = get_binary(load_image(img_path))
    image = Image(name, source_skeleton_img, goal_img)
    image.setup()
    image.auto_morph()
    image.make_rubber()
    while not image.done:
        image.update()
    if debug:
        save_image_debug(image.make_image(image.pixels, True), name)
    else:
        image.stroke = 0
        save_image(image.make_image(image.pixels, False), name)

def offset_both(img_path):
    optdataroot = "datasets/font"
    name = img_path.replace(optdataroot, "datasets/font_offset_check")
    name2 = img_path.replace(optdataroot, "datasets/font_offset_color")
    name_debug = img_path.replace(optdataroot, "datasets/font_offset_debug")
    if os.path.exists(name2):
        return
    print(name2)
    source_skeleton_path = get_source_skeleton_path(img_path)
    source_skeleton_img = get_binary(load_image(source_skeleton_path))
    goal_img = get_binary(load_image(img_path))
    image = Image(name, source_skeleton_img, goal_img)
    image.setup()
    image.auto_morph()
    image.make_rubber()
    while not image.done:
        image.update()
    image_debug = image.make_image(image.pixels, True)
    image.stroke = 0
    #image_offset = image.make_image(image.pixels, False)
    image_color = image.make_offset_image()
    save_image_debug(image_debug, name_debug)
    save_image_rgb(image_color, name2) #Could be bugged because np.array([])*255 behaves wierdly
    #color_to_skel(name2) #Bug arashi transform M+
    #save_image(image_offset, name)

def offset_color_skeleton(img_path):
    name = img_path.replace("datasets/font", "datasets/font_offset_color")
    if os.path.exists(name):
        return
    print(name)
    source_skeleton_path = get_source_skeleton_path(img_path)
    source_skeleton_img = get_binary(load_image(source_skeleton_path))
    goal_img = get_binary(load_image(img_path))
    image = Image(name, source_skeleton_img, goal_img)
    image.setup()
    image.auto_morph()
    image.make_rubber()
    while not image.done:
        image.update()
    image.stroke = 0
    save_image_rgb(image.make_offset_image(), name) #Could be bugged because np.array([])*255 behaves wierdly
    #save_image(character_img, img_path.replace("datasets/font_test_cut", "datasets/font_resize_cut"))
    #print(pos0_to_posneg(np.array(info)))

def warmup(input):
    return input

def color_to_skel(img_path):
    #optdataroot = "datasets/font_help"
    #from_path = img_path.replace(optdataroot, "datasets/font_offset_color_test")
    #if not os.path.exists(from_path):
    #    return
    from_path = img_path
    name = from_path.replace("offset_color", "offset")
    if os.path.exists(name):
        return
    print(name)
    source_skeleton_path = get_source_skeleton_path(img_path)
    source_skeleton_img = get_binary(load_image(source_skeleton_path))
    image = Image(name, source_skeleton_img, source_skeleton_img)
    image.setup()
    goal_img = load_image_rgb(from_path)
    image.color_to_skel(goal_img)
    image.stroke = 0
    image_skel = image.make_image(image.pixels, False)
    save_image(image_skel, name)

def make_font_images():
    use_pool = True
    pool_amount = 10

    if use_pool:
        print("Setting up pool...")
        if pool_amount <= 0:
            pool_amount = mul.cpu_count()
        pool = mul.Pool(processes=pool_amount)
        pool.map(warmup, range(pool_amount))
        print("Done with pool")

    for language in ["english", "chinese"]:#"source", "english", "chinese"]:
        optdataroot = "./datasets/font"
        optphase = "train"
        #optphase = "test_unknown_style"
        optmax_dataset_size = float("inf")
        dataroot = os.path.join(optdataroot, optphase, language)  # get the image directory
        paths = sorted(make_dataset(dataroot, optmax_dataset_size))  # get image paths
        #func = offset_skeleton #check if Debug = True
        #func = offset_color_skeleton
        func = offset_both
        #func = color_to_skel
        if use_pool:
            pool.map(func, iter(paths))
        else:
            list(map(func, iter(paths)))

def fitness(source, goal):
    image = Image("name", source, goal)
    image.setup()
    image.auto_morph()
    image.make_rubber()
    while not image.done:
        image.update()
    _, fitness = image.image_and_fitness(image.pixels, debug = False)
    fitness_empty = image.fitness_empty()
    diff = fitness_empty - fitness
    #print(fitness_empty, fitness)
    return diff/(fitness + 10)

def coefficient(list):
    return np.std(list)/np.average(list)


def outline_check_font(parent_path, common_info_dict, optphase):
    f1 = 0
    f2 = 0
    f3 = 0
    for filename in os.listdir(parent_path):
        if filename not in ["o.png", "x.png"]:
            continue
        f = os.path.join(parent_path, filename)
        img = get_binary(load_image(f))
        remove_common(img,f,common_info_dict, optphase)
        img_help = get_binary_no_thresh(load_image(f))
        remove_common(img_help,f,common_info_dict, optphase)
        img_depth_2, _ = layer_depth_2_info(img_help)
        img_invert = layer_depth_invert(img_help)
        if (64**2 - sum(sum(img))) > (64**2 - sum(sum(img_depth_2)))*8: #Empty or very small, dont pick outline
            f2 -= 1000000
        if (64**2 - sum(sum(img))) > (64**2 - sum(sum(img_invert)))*8: #Empty or very small
            return ImgMode.NORMAL
        source_path = get_source_skeleton_path(f)
        source_img = get_binary(load_image(source_path))

        f1 += fitness(source_img, img)
        f2 += fitness(source_img, img_depth_2)
        f3 += fitness(source_img, img_invert)
    best_fitness = max(f1,f2,f3)
    if f1 == best_fitness:
        return ImgMode.NORMAL
    if f2 == best_fitness:
        return ImgMode.OUTLINE
    if f3 == best_fitness:
        return ImgMode.INVERT

def fix_outlines():
    outline_info_dict = {} # true => font has a outline
    common_info_dict = {} # common black pixels
    for language in ["english", "chinese"]:#"source", "english", "chinese"]:
        optdataroot = "datasets/font"
        optendlocation = "datasets/font"
        optphase = "train"
        optphase = "test_unknown_style"
        optmax_dataset_size = float("inf")
        dataroot = os.path.join("./" + optdataroot, optphase, language)  # get the image directory
        paths = sorted(make_dataset(dataroot, optmax_dataset_size))  # get image paths
        base = get_parent_folder(get_parent_folder(paths[0]))
        for img_path in paths:
            name = img_path.replace(optdataroot, optendlocation)
            if os.path.exists(name):
                continue
            parent_path = get_parent_folder(img_path.replace(optphase + os.sep + "chinese", optphase + os.sep + "english"))
            if parent_path not in outline_info_dict:
                outline_info_dict[parent_path] = outline_check_font(parent_path, common_info_dict, optphase)
                #print(parent_path)
            character_img = get_binary(load_image(img_path))
            if outline_info_dict[parent_path] == ImgMode.OUTLINE:
                character_img_help = get_binary_no_thresh(load_image(img_path))
                remove_common(character_img, img_path, common_info_dict, optphase)
                character_img, _ = layer_depth_2_info(character_img_help)
            elif outline_info_dict[parent_path] == ImgMode.INVERT:
                character_img_help = get_binary_no_thresh(load_image(img_path))
                remove_common(character_img, img_path, common_info_dict, optphase)
                character_img = layer_depth_invert(character_img_help)
            else:
                remove_common(character_img, img_path, common_info_dict, optphase)
            save_image(character_img, name)

# Some fonts have a common background for all the characters, this background will ruin the skeleton deformation
def remove_common(img, img_path, common_info_dict, optphase):
    parent_path = get_parent_folder(img_path.replace(optphase + os.sep + "chinese", optphase + os.sep + "english"))
    if parent_path in common_info_dict and common_info_dict[parent_path] is None:
        n = 64
        common = get_binary_no_thresh(load_image(img_path))
        for filename in os.listdir(parent_path):
            f = os.path.join(parent_path, filename)
            image = get_binary_no_thresh(load_image(f))
            common += image
        common_info_dict[parent_path] = None
        n = 64
        for y in range(n):
            if common_info_dict[parent_path] is not None:
                break
            for x in range(n):
                if common[y][x] == 0:
                    common_info_dict[parent_path] = common
                    break
        count = 0
        for y in range(n):
            for x in range(n):
                if common[y][x] == 0:
                    count += 1
        if count < 10: # accidental hits most likely, therefore set to None
            common_info_dict[parent_path] = None
        elif count > 0:
            print(parent_path, "| Common:", count)
    if parent_path in common_info_dict and common_info_dict[parent_path] is not None:
        n = 64
        common = common_info_dict[parent_path]
        for y in range(n):
            for x in range(n):
                if common[y][x] == 0:
                    img[y][x] = 1

def main():
    make_skeleton_images()
    make_font_images()




if __name__ == '__main__':
    main()