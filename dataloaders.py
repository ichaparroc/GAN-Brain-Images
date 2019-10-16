import os
import glob
import torch
import errno
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms, datasets
from skimage import io
from PIL import Image

class dataset_basel_segmentation(torch.utils.data.Dataset):
    def __init__(self, path, subset, type_label):
        self.root_dir1 = path+'basel/'+subset+'/ims/2/'
        self.root_dir2 = path+'basel/'+subset+'/gts/2/'
        self.type_label = type_label
        self.filelist1 = glob.glob(self.root_dir1+'*.png')
        self.transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([.5], [.5])])
        
    def __len__(self):
        return len(self.filelist1)
    
    def __getitem__(self, idx):
        x = io.imread(self.filelist1[idx])
        cut = self.filelist1[idx].rfind("/")
        cut += 1
        filename = self.filelist1[idx][cut:]
        pair = self.root_dir2 + filename
        y = io.imread(pair)
        x = np.uint8(x)
        y = np.uint8(y)

        if(self.type_label == 'label'):
            y *= 255 #original only 0's and 1's
        elif(self.type_label == 'label_input'):
            y *= x
        else:
            raise Exception("Invalid type_label, type_targets: label, label_input")
        x = Image.fromarray(x)
        y = Image.fromarray(y)
        x = self.transform(x)
        y = self.transform(y)
        return x, y

class dataset_basel_synthesis(torch.utils.data.Dataset):
    def __init__(self, path, subset):
        self.root_dir1 = path+'basel/'+subset+'/shp/2/'
        self.root_dir2 = path+'basel/'+subset+'/gts/2/'
        self.root_dir3 = path+'basel/'+subset+'/ims/2/'
        self.filelist1 = glob.glob(self.root_dir1+'*.png')
        self.transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([.5], [.5])])
        
    def __len__(self):
        return len(self.filelist1)
    
    def __getitem__(self, idx):
        x1 = io.imread(self.filelist1[idx])
        cut = self.filelist1[idx].rfind("/")
        filename = self.filelist1[idx][cut+1:]
        pair2 = self.root_dir2 + filename
        pair3 = self.root_dir3 + filename
        x2 = io.imread(pair2)
        y = io.imread(pair3)
        x1 = np.uint8(x1)
        x2 = np.uint8(x2)
        y = np.uint8(y)
        x2 *= 255
        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        y = Image.fromarray(y)
        cut = filename.rfind("_")
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        y = self.transform(y)
        return x1, x2, y

class dataset_isles2018_segmentation(torch.utils.data.Dataset):
    def __init__(self, path, subset):
        self.root_dir1 = path+'isles2018/'+subset+'/CBF/'
        self.root_dir2 = path+'isles2018/'+subset+'/CBV/'
        self.root_dir3 = path+'isles2018/'+subset+'/CT/'
        self.root_dir4 = path+'isles2018/'+subset+'/MTT/'
        self.root_dir5 = path+'isles2018/'+subset+'/Tmax/'
        self.subset = subset
        if(subset == 'train'):
            self.root_dir6 = path+'isles2018/'+subset+'/OT/'
        elif(subset != 'test'):
            raise Exception("Invalid subset, subsets: train, test")
        self.filelist1 = glob.glob(self.root_dir1+'*.png')
        self.transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([.5], [.5])])
        
    def __len__(self):
        return len(self.filelist1)
    
    def __getitem__(self, idx):
        x1 = io.imread(self.filelist1[idx])
        cut = self.filelist1[idx].rfind("/")
        filename = self.filelist1[idx][cut+1:]
        pair2 = self.root_dir2 + filename
        pair3 = self.root_dir3 + filename
        pair4 = self.root_dir4 + filename
        pair5 = self.root_dir5 + filename
        x2 = io.imread(pair2)
        x3 = io.imread(pair3)
        x4 = io.imread(pair4)
        x5 = io.imread(pair5)

        x1 = np.uint8(x1)
        x2 = np.uint8(x2)
        x3 = np.uint8(x3)
        x4 = np.uint8(x4)
        x5 = np.uint8(x5)

        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        x3 = Image.fromarray(x3)
        x4 = Image.fromarray(x4)
        x5 = Image.fromarray(x5)

        x1 = self.transform(x1)
        x2 = self.transform(x2)
        x3 = self.transform(x3)
        x4 = self.transform(x4)
        x5 = self.transform(x5)

        if(self.subset == 'train'):
            pair6 = self.root_dir6 + filename
            y = io.imread(pair6)
            y = np.uint8(y)
            y = Image.fromarray(y)
            y = self.transform(y)
            return x1, x2, x3, x4, x5, y
        elif(self.subset == 'test'):
            return x1, x2, x3, x4, x5
        else:
            raise Exception("Invalid subset, subsets: train, test")

class dataset_brats2015_segmentation(torch.utils.data.Dataset):
    def __init__(self, path, subset):
        self.root_dir1 = path+'brats2015/'+subset+'/Flair/'
        self.root_dir2 = path+'brats2015/'+subset+'/T1/'
        self.root_dir3 = path+'brats2015/'+subset+'/T1c/'
        self.root_dir4 = path+'brats2015/'+subset+'/T2/'
        self.subset = subset
        if(subset == 'train'):
            self.root_dir5 = path+'brats2015/'+subset+'/OT/'
        elif(subset != 'test'):
            raise Exception("Invalid subset, subsets: train, test")
        self.filelist1 = glob.glob(self.root_dir1+'*.png')
        self.transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([.5], [.5])])
        
    def __len__(self):
        return len(self.filelist1)
    
    def __getitem__(self, idx):
        x1 = io.imread(self.filelist1[idx])
        cut = self.filelist1[idx].rfind("/")
        filename = self.filelist1[idx][cut+1:]
        pair2 = self.root_dir2 + filename
        pair3 = self.root_dir3 + filename
        pair4 = self.root_dir4 + filename
        x2 = io.imread(pair2)
        x3 = io.imread(pair3)
        x4 = io.imread(pair4)

        x1 = np.uint8(x1)
        x2 = np.uint8(x2)
        x3 = np.uint8(x3)
        x4 = np.uint8(x4)

        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        x3 = Image.fromarray(x3)
        x4 = Image.fromarray(x4)

        x1 = self.transform(x1)
        x2 = self.transform(x2)
        x3 = self.transform(x3)
        x4 = self.transform(x4)

        if(self.subset == 'train'):
            pair5 = self.root_dir5 + filename
            y = io.imread(pair5)
            y = np.uint8(y)
            y = Image.fromarray(y)
            y = self.transform(y)
            return x1, x2, x3, x4, y
        elif(self.subset == 'test'):
            return x1, x2, x3, x4
        else:
            raise Exception("Invalid subset, subsets: train, test")