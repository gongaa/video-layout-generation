import torch.utils.data as data
from PIL import Image
import cv2
import json
import os
import os.path
import sys
from random import randint
import torch
import torchvision.transforms as transforms
from itertools import groupby
from operator import itemgetter

def make_dataset(dir):
    path = []
    dir = os.path.expanduser(dir)
    seg_subdir = dir + '/deeplab256_label'
    # seg_subdir = dir + '/deeplab'
    img_subdir = dir + '/leftImg256'
    cities = [city for city in os.listdir(seg_subdir)]
    for city in cities:
        seg_city_subdir = os.path.join(seg_subdir, city)
        img_city_subdir = os.path.join(img_subdir, city)
        ff = [f for f in os.listdir(seg_city_subdir) if f.endswith('.png')]
        idx_snippet = set([int( f.split('_')[1]) for f in ff])
        for idx in idx_snippet:
            fs = [int(f.split('_')[2]) for f in ff if f.startswith(city+'_'+str(idx).zfill(6))]
            fs.sort()
            ranges = []
            for k,g in groupby(enumerate(fs),lambda x:x[0]-x[1]):
                ranges.append(list(map(itemgetter(1),g)))
            suffix = []
            for r in ranges:
                for i in range(r[0], r[-1]-6):
                    suffix.append([str(i).zfill(6), str(i+3).zfill(6), str(i+6).zfill(6)])
            prefix = os.path.join(city, city+'_'+str(idx).zfill(6)+'_')
            suffix_dir = []
            for s in suffix:
                suffix_dir.append([prefix+s[i] for i in range(3)])
            for ss in suffix_dir:
                seg_p = [s+'_gtFine_myseg_id.png' for s in ss]
                # seg_p = [s+'_leftImgseg.png' for s in ss]
                img_p = [s+'_leftImg8bit.png' for s in ss]
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
        # self.targets = [s[1][1] for s in samples]

        self.transform = transform
        # self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        seg_paths, img_paths = self.samples[index]
        seg = [cv2_loader_seg(p) for p in seg_paths]
        img = [cv2_loader_RGB(p) for p in img_paths]
        if self.transform is not None:
            seg = [self.transform(s) for s in seg]
            img = [self.transform(i) for i in img]
        to_normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        return to_normalize(img[0]), seg[0], to_normalize(img[1]), seg[1], to_normalize(img[2]) # also normalize GT image

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

def cv2_loader_RGB(path):
    with open(path, 'r') as f:
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)    # BGR to RGB, otherwise zombie on tensorboard
        # im = cv2.resize(im, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        return im

def cv2_loader_seg(path):
    with open(path, 'r') as f:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # im = cv2.imread(path)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        return im



def pil_loader_RGB(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        im = Image.open(f)
        im.resize((256,256), Image.ANTIALIAS)
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
