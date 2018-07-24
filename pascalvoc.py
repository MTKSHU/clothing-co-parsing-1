import numpy as np
import os
import scipy.io

from PIL import Image
import torch,pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from transforms import OneHotEncode

def split_train_test_set(train_img_root,train_mask_root,split_ratio =0.9):
    img_root, mask_root = train_img_root,train_mask_root
    imgs = [x.split('.')[0] for x in os.listdir(img_root) if x.find('jpg')>0]
    masks = [x.split('.')[0] for x in os.listdir(mask_root) if x.find('mat')>0]
    common_files = list(set(imgs)&set(masks))
    common_files = np.sort(common_files)
    train_index = np.random.choice(range(len(common_files)), int(split_ratio*len(common_files)),replace=False)
    train_imgs = common_files[train_index]
    train_imgs.sort()
    test_imgs = [x for x in common_files if x not in train_imgs]
    pd.Series(train_imgs).to_csv('train.csv',index=False)
    pd.Series(test_imgs).to_csv('test.csv',index=False)




class PascalVOC(Dataset):

    def __init__(self, data_root, photo_root ='photos',mask_root='annotations/pixel-level',img_transform = Compose([]),\
     label_transform=Compose([]), co_transform=Compose([]),\
      train_phase=True,split=1,labeled=True,seed=0):
        np.random.seed(seed)
        self.n_class = 59
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, photo_root)
        self.labels_root = os.path.join(self.data_root, mask_root)
        self.img_list = pd.read_csv(os.path.join(self.data_root,'train.csv'),dtype=str).values.ravel() if train_phase else pd.read_csv(os.path.join(self.data_root,'test.csv')).values.ravel()
        # self.img_list = [str(x[0]) for x in self.img_list]

        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.train_phase = train_phase

    def __getitem__(self, index):
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.jpg'), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(os.path.join(self.labels_root,filename+'.mat'), 'rb') as f:
            label = scipy.io.loadmat(f)['groundtruth']
            label = Image.fromarray(label)

        image, label = self.co_transform((image,label))
        image = self.img_transform(image)
        label = self.label_transform(label)
        ohlabel = OneHotEncode()(label)

        return image, label, ohlabel

    def __len__(self):
        return len(self.img_list)

if __name__=="__main__":
    from torchvision.transforms import ToTensor, Compose

    from transforms  import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn, ZeroPadding, OneHotEncode

    imgtr = [ToTensor()]
    labtr = [ToTensorLabel()]
    cotr = [RandomSizedCrop((321,321))]

    a = PascalVOC(data_root='',img_transform=Compose(imgtr), label_transform=Compose(labtr),co_transform=Compose(cotr))
    a = iter(a)
    b,c,d = next(a)

