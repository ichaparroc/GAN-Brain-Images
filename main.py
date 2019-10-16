import os
import glob
import torch
import errno
import random
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
from IPython import display
from matplotlib import pyplot as plt
from matplotlib import interactive
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from torchsummary import summary
from torch.autograd import Variable
from torchvision import transforms, datasets
from skimage import io
from PIL import Image
import argparse

plt.ion()

from dataloaders import *
from parameter import *

def sample_images(dataset):
	datas = []
	datas = next(iter(dataset))
	for data in datas:
		plt.figure(figsize=(config.sample_size, config.sample_size), dpi = data.shape[2])
		print(data.shape[2])
		plt.axis("off")
		plt.imshow(np.transpose(vutils.make_grid(data[:sample_size*sample_size], nrow = sample_size, normalize = True),(1,2,0)))
		plt.show()
	
	input("press intro")

config = get_parameters()
print(config)

####### Set Reproductibility

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
print("Random Seed: ", manualSeed)

####### Set Value Settings

sample_size = 3

##TODO TODO TODO
#number of channels TODO

####### Dataloaders

if(config.dataset == 'basel'):

	if(config.task == 'segmentation'):

		train_dataset = dataset_basel_segmentation(path = config.data_path, subset ='train', type_label = config.type_label)
		dataset_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size)

		valid_dataset = dataset_basel_segmentation(path = config.data_path, subset ='valid', type_label = config.type_label)
		dataset_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size = config.batch_size)

		sample_images(dataset_loader_train)
		sample_images(dataset_loader_valid)

	elif(config.task == 'synthesis'):

		train_dataset = dataset_basel_synthesis(path = config.data_path, subset ='train')
		dataset_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size)

		valid_dataset = dataset_basel_synthesis(path = config.data_path, subset ='valid')
		dataset_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size = config.batch_size)

		sample_images(dataset_loader_train)
		sample_images(dataset_loader_valid)

elif(config.dataset == 'isles2018'):

	if(config.task == 'segmentation'):

		train_dataset = dataset_isles2018_segmentation(path = config.data_path, subset ='train',)
		dataset_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size)

		test_dataset = dataset_isles2018_segmentation(path = config.data_path, subset ='test')
		dataset_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size)

		sample_images(dataset_loader_train)
		sample_images(dataset_loader_test)

	elif(config.task == 'synthesis'):

		print("not implemented") #TODO

elif(config.dataset == 'brats2015'):

	if(config.task == 'segmentation'):

		train_dataset = dataset_brats2015_segmentation(path = config.data_path, subset ='train',)
		dataset_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size)

		test_dataset = dataset_brats2015_segmentation(path = config.data_path, subset ='test')
		dataset_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size)

		sample_images(dataset_loader_train)
		sample_images(dataset_loader_test)

