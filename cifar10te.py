from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
# from .vision import VisionDataset

class CIFAR10te(VisionDataset):

    def __init__(
            self,
            root: str,
            label_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:

        super(CIFAR10te, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        # {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.ds_to_img_subgroup = np.load(os.path.join(root, f'{label_file}.npy'), allow_pickle=True).item()  # a dict stores e.g. 'bird': {'003.png': 1}, 'frog' : {'001.png':0}, etc
        # root should be 'cifar_test'

        self.data: Any = []
        self.targets = []
        self.subgroup = []
        self.n_classes = 10
        import pdb
        # pdb.set_trace()
        for class_name in self.classes:
            print(f'processing {class_name} test set')
            for img_id in self.ds_to_img_subgroup[class_name].keys() :  # os.listdir(os.path.join(self.root, class_name)):
                file_path = os.path.join(self.root, 'test', class_name, img_id)
                img = Image.open(file_path) # they should be 32,32,3 already
                np_img = np.asarray(img) # note that right here they are uint8
                assert np_img.shape == (32,32,3), 'numpy img shape is not (32,32,3)'
                self.data.append(np_img.reshape(1, 32, 32, 3))
                self.targets.append(self.class_to_idx[class_name])
                self.subgroup.append(self.ds_to_img_subgroup[class_name][img_id]) 
  
        self.data = np.vstack(self.data) # .reshape(-1, 32, 32, 3)

        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, subgroup = self.data[index], self.targets[index], self.subgroup[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, subgroup


    def __len__(self) -> int:
        return len(self.data)


