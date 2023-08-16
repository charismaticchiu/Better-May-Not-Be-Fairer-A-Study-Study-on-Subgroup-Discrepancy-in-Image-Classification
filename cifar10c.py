import numpy as np
import os
import PIL
import torch
import torchvision
from typing import Any, Callable, Optional, Tuple
from collections import defaultdict
from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets
import random
# from utils import load_txt

corruptions = ['natural',
                'gaussian_noise',
                'shot_noise',
                'speckle_noise',
                'impulse_noise',
                'defocus_blur',
                'gaussian_blur',
                'motion_blur',
                'zoom_blur',
                'snow',
                'fog',
                'brightness',
                'contrast',
                'elastic_transform',
                'pixelate',
                'jpeg_compression',
                'spatter',
                'saturate',
                'frost']

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str, label_file: str,severity: int,
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.data = self.data[(severity-1)*10000:severity*10000]
        self.targets = self.targets[(severity-1)*10000:severity*10000]
        
        #### custom
        self.subgroup = []
        self.n_classes = 10
        label_count = defaultdict(int)
        img_id2dataset_id = defaultdict(dict)
        dataset_id2img_id = {}
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.ds_to_img_subgroup = np.load(f'{label_file}.npy', allow_pickle=True).item()  # a dict stores e.g. 'bird': {'0003.png': 1}, 'frog' : {'0001.png':0}, etc
        # root should be 'cifar_test'
        for i, (label) in enumerate(self.targets):
            label_count[label] += 1
            img_id = label_count[label]
            # img_id2dataset_id[label][img_id] = i
            # dataset_id2img_id[i] = (label,img_id)
            cl_name = self.classes[label]
            img_name = '%04d.png' % img_id
            # import pdb
            # pdb.set_trace()
            self.subgroup.extend([self.ds_to_img_subgroup[cl_name][img_name]])
        assert len(self.subgroup) == len(self.targets)
       
    def __getitem__(self, index):
        img, targets, subgroup = self.data[index], self.targets[index], self.subgroup[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets, subgroup
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        # random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)