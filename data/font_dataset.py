import os
from matplotlib import style
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

class FontDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.style_language = opt.direction.split("2")[0]
        self.content_language = opt.direction.split("2")[1]
        BaseDataset.__init__(self, opt)
        self.dataroot = os.path.join(opt.dataroot, opt.phase, self.content_language)  # get the image directory
        self.paths = sorted(make_dataset(self.dataroot, opt.max_dataset_size))  # get image paths
        self.style_channel = opt.style_channel
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.5), std = (0.5))])
        self.img_size = opt.load_size

        self.stages = opt.stage
        self.base_parent_path = opt.dataroot.replace("/", os.sep)
        self.style_parent_path = self.base_parent_path
        self.base_content_parent_path = self.base_parent_path
        self.content_parent_path = "." + os.sep + "datasets" + os.sep + "font_offset_color"
        if opt.stage == "0":
            self.gt_parent_path = self.content_parent_path
        elif opt.stage == "1":
            self.gt_parent_path = self.base_parent_path

    def __getitem__(self, index):
        # get content path and corresbonding stlye paths
        gt_path = self.paths[index].replace("/", os.sep)
        gt_path = gt_path.replace(self.base_parent_path, self.gt_parent_path)
        style_paths = self.get_style_paths(gt_path)
        content_path = self.get_content_path(gt_path)
        # load and transform images
        gt_image = self.load_image(gt_path)
        content_image = self.load_image(content_path)
        style_image = torch.from_numpy(np.array([np.array(self.load_image(style_path)) for style_path in style_paths]))
        base_content_path = self.get_base_content_path(gt_path)
        base_content_image = self.load_image(base_content_path)
        return {'gt_images':gt_image, 'style_images':style_image,
                'style_image_paths':style_paths, 'image_paths':gt_path, 'base_content_images': base_content_image, 'content_images': content_image}
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
    
    def load_image(self, path):
        image = Image.open(path)
        image = self.transform(image)
        return image
        
    def get_style_paths(self, gt_path):
        style_path = gt_path.replace(self.gt_parent_path, self.style_parent_path)
        parts = style_path.split(os.sep)
        english_font_path = os.path.join(parts[0], parts[1], parts[2], parts[3], self.style_language, parts[5])
        english_paths = [os.path.join(english_font_path, letter) for letter in random.sample(os.listdir(english_font_path), self.style_channel)] #random.randint(1,self.style_channel)
        return english_paths
    
    def get_base_content_path(self, gt_path):
        base_content_path = gt_path.replace(self.gt_parent_path, self.base_content_parent_path)
        parts = base_content_path.split(os.sep)
        return os.path.join(parts[0], parts[1], parts[2], parts[3], 'source', parts[-1])

    def get_content_path(self, gt_path):
        content_path = gt_path.replace(self.gt_parent_path, self.content_parent_path)
        parts = content_path.split(os.sep)
        return os.path.join(parts[0], parts[1], parts[2], parts[3], 'source', parts[-1])