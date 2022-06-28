import os
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random

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
        dataset, self.styles, self.chars = make_dataset(self.dataroot, opt.max_dataset_size)
        self.paths = sorted(dataset)  # get image paths
        self.style_channel = opt.style_channel
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.5), std = (0.5))])
        self.img_size = opt.load_size
        
    def __getitem__(self, index):
        # get content path and corresbonding stlye paths
        gt_path = self.paths[index].replace("/", os.sep)
        parts = gt_path.split(os.sep)
        style_paths = self.get_style_paths(parts)
        content_path = self.get_content_path(parts)
        alpha = 0
        # load and transform images
        content_image = self.load_image(content_path)
        gt_image = self.load_image(gt_path)
        style_image = torch.cat([self.load_image(style_path) for style_path in style_paths], 0)
        style_index = self.styles.index(parts[-2])
        return {'gt_images':gt_image, 'content_images':content_image, 'style_images':style_image,
                'style_image_paths':style_paths, 'image_paths':gt_path, 'style_indexes': style_index} #, 'style_index': style_index, 'char_index': char_index}
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
    
    def load_image(self, path):
        image = Image.open(path)
        image = self.transform(image)
        return image
        
    def get_style_paths(self, parts):
        english_font_path = os.path.join(parts[0], parts[1], parts[2], parts[3], self.style_language, parts[5])
        english_paths = [os.path.join(english_font_path, letter) for letter in random.sample(os.listdir(english_font_path), self.style_channel)] #random.randint(1,self.style_channel)
        return english_paths
    
    def get_different_style_path(self, parts, letter):
        english_font_path_parent = os.path.join(parts[0], parts[1], parts[2], parts[3], self.style_language)
        english_font_path = os.path.join(english_font_path_parent, random.sample(os.listdir(english_font_path_parent), 1)[0]) #random.randint(1,self.style_channel)
        english_path = os.path.join(english_font_path, letter) #random.randint(1,self.style_channel)
        while not os.path.exists(english_path):
            english_font_path = os.path.join(english_font_path_parent, random.sample(os.listdir(english_font_path_parent), 1)[0])
            english_path = os.path.join(english_font_path, letter)
        return english_path
    
    def get_content_path(self, parts):
        return os.path.join(parts[0], parts[1], parts[2], parts[3], 'source', parts[-1])