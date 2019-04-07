import torch.utils.data as data
from PIL import Image
import json
import os
import os.path
import sys
from random import randint
import torch

def make_dataset(dir):
    path = []
    dir = os.path.expanduser(dir)
    seg_subdir = dir + '/deeplab'
    img_subdir = dir + '/leftImg'
    cities = [city for city in os.listdir(seg_subdir)]
    city_dict = {}
    for city in cities:
        c = os.path.join(seg_subdir,city)
        num_snippet = int(os.listdir(c) / 30)
        city_dict[city] = num_snippet
    
    suffix = []
    suffix_num = []
    for i in range(30-4): # [1,3,5], [2,4,6]...[25,27,29]
        suffix_num.append([str(i).zfill(6), str(i+2).zfill(6), str(i+4).zfill(6)])
    
    for city in cities:
        num_snippet = city_dict[city]
        for n in range(num_snippet):
            s = city + '_' + str(n).zfill(6)+'_'
            for k in range(len(suffix_num)):
                suffix.append([s+suffix_num[k][i] for i in range(3)])
    
    for ss in suffix:
        seg_p = [s+'_leftImg8bit.png' for s in ss]
        img_p = [s+'_leftImgseg.png' for s in ss]
        path.append(([os.path.join(seg_subdir,p) for p in seg_p], [os.path.join(img_subdir,p) for p in img_p]))
            
    return path


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root = /data/agong
        root/train/leftImg/aachen/aachen_000116_000000_leftImg8bit.png
        root/train/deeplab/aachen/aachen_000115_000029_leftImgseg.png
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        samples = make_dataset(root)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[2][1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        seg_paths, img_paths = self.samples[index]
        seg = [pil_loader_seg(p) for p in seg_paths]
        img = [pil_loader_RGB(p) for p in img_paths]
        return img[0], seg[0], img[2], seg[2], img[1]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader_RGB(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        im = Image.open(f)
        im.thumbnail(256, Image.ANTIALIAS)
        return im.convert('RGB')

def pil_loader_seg(path):
    with open(path, 'rb') as f:
        im = Image.open(f)
        im.resize((256,256), Image.NEAREST)
        return im.convert('RGB')

def pil_loader_8bit(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    return Image.open(path)

def default_loader(path):
    return pil_loader_8bit(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root = /data/agong/
        root/leftImg/train/aachen/aachen_000116_000000_leftImg8bit.png
        root/deeplab/train/aachen/aachen_000115_000029_leftImgseg.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
