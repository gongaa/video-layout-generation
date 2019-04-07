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
    seg_subdir = dir + '/segmentations'
    json_subdir = dir + '/jsons'
    seg_files = [os.path.join(seg_subdir,filename) for filename in os.listdir(seg_subdir) if filename.endswith('.png')]
    json_files = [os.path.join(json_subdir,filename) for filename in os.listdir(json_subdir) if filename.endswith('.json')]
    seg_files.sort(key=lambda r: int((r.split('/')[-1]).split('.')[0]))
    json_files.sort(key=lambda r: int((r.split('/')[-1]).split('.')[0]))
    for seg_path, json_path in zip(seg_files, json_files):
        path.append((seg_path, json_path))

    return path


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root = ~/../../data/agong
        root/train/images
        root/train/jsons with same id
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
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, json_path = self.samples[index]
        seg = self.loader(path)
        # do something to json_path
        if self.transform is not None:
            seg = self.transform(seg)
        # if self.target_transform is not None:
        mask, onehot = add_mask_onehot(json_path)
        return seg, mask, onehot

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
        img = Image.open(f)
        return img.convert('RGB')

def pil_loader_8bit(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    return Image.open(path)

def default_loader(path):
    return pil_loader_8bit(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root = ~/../../data/agong/train or val or test
        root/images
        root/jsons
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


img_h = 1024
img_w = 2048
num_cls = 29
min_area = 20000
max_area = 1024*2048/4
min_total_missing_area = 2000

label2index = {
    'unlabeled':0,
    'ego vehicle':0,
    'rectification border':0,
    'out of roi':0,
    'static':0,
    'dynamic':1,
    'ground':2,
    'road':3,
    'sidewalk':4,
    'parking':5,
    'rail track':6,
    'building':7,
    'wall':8,
    'fence':9,
    'guard rail':10,
    'bridge':11,
    'tunnel':12,
    'pole':13,
    'polegroup':13,
    'traffic light':14,
    'traffic sign':15,
    'vegetation':16,
    'terrain':17,
    'sky':18,
    'person':19,
    'persongroup':19,
    'rider':20,
    'ridergroup':20,
    'car':21,
    'cargroup':21,
    'truck':22,
    'truckgroup':22,
    'bus':23,
    'caravan':24,
    'trailer':25,
    'train':26,
    'motorcycle':27,
    'motorcyclegroup':27,
    'bicycle':28,
    'bicyclegroup':28,
    'license plate':0
}

def add_mask_onehot(file):
    with open(file, 'r') as f:
        text = json.load(f)
    mask = torch.zeros((img_h,img_w))
    objects_list = text['objects']
    # sort object according to their idx
    objects_list = sorted(objects_list, key=lambda k: k['idx'])
    num_obj = len(objects_list)
    while True:
        rand_idx = randint(0, num_obj-1)
        # check whether area fullfill requirement
        x_1, x_2, y_1, y_2 = objects_list[rand_idx]['x_min'], objects_list[rand_idx]['x_max'], objects_list[rand_idx]['y_min'], objects_list[rand_idx]['y_max']
        bbox_area = (x_2-x_1)*(y_2-y_1)
        if bbox_area > max_area or bbox_area < min_area: 
            continue
        # generate random crop
#         print(rand_idx)
        w = x_2-x_1
        h = y_2-y_1
#         print(x_1,x_2,y_1,y_2)
        rand_w = randint(w//2, w)
        rand_h = randint(h//2, h)
        rand_x1 = randint(x_1, x_2-rand_w)
        rand_y1 = randint(y_1, y_2-rand_h)
        rand_x2 = rand_x1 + rand_w
        rand_y2 = rand_y1 + rand_h
        mask[rand_y1:rand_y2, rand_x1:rand_x2] = 1
#         print("axis",rand_x1,rand_x2,rand_y1,rand_y2)
        break

    one_hot = torch.zeros(num_cls)
    # generate one-hot vector
    for idx in objects_list[rand_idx]['intersection_id']:
        # check whether its bbox is fully contained in cropped window
        # check whether its area is bigger than the threshold
        x_1,x_2,y_1,y_2 = objects_list[idx]['x_min'],objects_list[idx]['x_max'],objects_list[idx]['y_min'],objects_list[idx]['y_max']
        if x_1>rand_x1 and x_2<rand_x2 and y_1>rand_y1 and y_2<rand_y2 and (x_2-x_1)*(y_2-y_1)>min_total_missing_area:
            lin_idx = label2index[objects_list[idx]['label']] # label that lin set
            one_hot[lin_idx] = one_hot[lin_idx] + 1 
        
    return mask.byte(), one_hot
