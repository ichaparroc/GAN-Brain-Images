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
from nets import *

def sample_images(dataset):
	datas = []
	datas = next(iter(dataset))
	for data in datas:
		plt.figure(figsize=(config.sample_size, config.sample_size), dpi = data.shape[2])
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

		train_dataset = dataset_isles2018_segmentation(path = config.data_path, subset ='train')
		dataset_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size)

		test_dataset = dataset_isles2018_segmentation(path = config.data_path, subset ='test')
		dataset_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size)

		sample_images(dataset_loader_train)
		sample_images(dataset_loader_test)

	elif(config.task == 'synthesis'):

		print("not implemented") #TODO

elif(config.dataset == 'brats2019'):

	if(config.task == 'segmentation'):

		train_dataset = dataset_brats2019_segmentation(path = config.data_path, subset ='TRAIN',)
		dataset_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size)

		test_dataset = dataset_brats2019_segmentation(path = config.data_path, subset ='VALID')
		dataset_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size)

		sample_images(dataset_loader_train)
		sample_images(dataset_loader_test)

	elif(config.task == 'synthesis'):

		train_dataset = dataset_brats2019_synthesis(path = config.data_path, subset ='TRAIN',)
		dataset_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size)

		test_dataset = dataset_brats2019_synthesis(path = config.data_path, subset ='VALID')
		dataset_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size)

		sample_images(dataset_loader_train)
		sample_images(dataset_loader_test)		


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

G = Generator()
D = Discriminator()
if torch.cuda.is_available():
  G.cuda()
  D.cuda()

G.apply(weights_init)
D.apply(weights_init)

"""
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
"""

G.train()
D.train()

# def Adam optimizer
G_optimizer = Adam(G.parameters(), lr=lr, betas=(config.beta1, config.beta2)) #betas is widely used for adamoptim
D_optimizer = Adam(D.parameters(), lr=lr, betas=(config.beta1, config.beta2))

def train_D(input, target): #eq.(1) 

	if(config.task == 'segmentation'):

	    D.zero_grad()
	        
	    output = G.forward(input).detach()
	        
	    result = D.forward(output)
	        
	    target_D = D.forward(target)
	    
	    D_loss = -torch.mean(torch.abs(result - target_D))
	    D_loss.backward()
	    
	    # update params
	    D_optimizer.step()
	    
	    return D_loss

def train_G(input, target): #eq.(4)

	if(config.task == 'segmentation'):

	    G.zero_grad()

	    output = G.forward(input)

	    result = D.forward(output)

	    target_G = D.forward(target)

	    #loss_dice = dice_loss(output,target)
	    G_loss = torch.mean(torch.abs(result - target_G))
	    #G_loss_dice = torch.mean(torch.abs(result - target_G)) + loss_dice
	    G_loss.backward()
	    #G_loss_dice.backward()
	    G_optimizer.step()

	    return G_loss

def test_images(x, y, epoch, n_batch):
    
    # forward prop
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        
    fake_y = G.forward(x)
  
    images = torch.cat((x, y, fake_y), 0).data.cpu()
  
    logger.save_torch_images(images, epoch, n_batch, plot_horizontal=True)


def print_cfm(conf_matrix):
    
    ##consufion matrix:
    FN = conf_matrix[0]
    FP = conf_matrix[1]
    TN = conf_matrix[2]
    TP = conf_matrix[3]
    
    try:
        err = (FP+FN)/(TP+TN+FN+FP)
    except:
        err = 0
            
    try:
        acc = (TP+TN)/(TP+TN+FN+FP)
    except:
        acc = 0

    try:
        sen = (TP)/(TP+FN)
    except:
        sen = 0
            
    try:
        spe = (TN)/(TN+FP)
    except: 
        spe = 0
            
    try:
        pre = (TP)/(TP+FP)
    except:
        pre = 0
        
    try:
        dice = (2*TP)/(2*TP + FN+ FP)
    except:
        dice = 0
    
    return [err, acc, sen, spe, pre, dice]

def test_segmentation(dataset):
    with torch.no_grad():
 
        all_preds = torch.tensor([])
        all_reals = torch.tensor([])
        for batch, (real_x_test, real_y_test) in enumerate(dataset):
            if torch.cuda.is_available():
                real_y_test = real_y_test.cuda()
                real_x_test = real_x_test.cuda()
                all_preds = all_preds.cuda()
                all_reals = all_reals.cuda()
            preds = G(real_x_test)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_reals = torch.cat((all_reals, real_y_test), dim=0)

        ones = torch.ones(1)
        zeros = torch.zeros(1)
        
        if torch.cuda.is_available():
            ones = ones.cuda()
            zeros = zeros.cuda()
           
        #pixel calculation
        all_preds_px = all_preds.reshape(1, -1)
        all_preds_px = all_preds.squeeze()
        all_reals_px = all_reals.reshape(1, -1)
        all_reals_px = all_reals.squeeze()

        all_preds_px = (all_preds_px + 1) / 2
        all_reals_px = (all_reals_px + 1) / 2
        
        all_preds_px = torch.where(all_preds_px>0.5,ones,zeros)
        all_reals_px = torch.where(all_reals_px>0.5,ones,zeros)
                
        all_preds_px = all_preds_px.type(torch.cuda.LongTensor).type(torch.cuda.FloatTensor)
        all_reals_px = all_reals_px.type(torch.cuda.LongTensor).type(torch.cuda.FloatTensor)
              
        confusion_vector = all_preds_px / all_reals_px
        
        TP = torch.sum(confusion_vector == 1).item()
        FP = torch.sum(confusion_vector == float('inf')).item()
        TN = torch.sum(torch.isnan(confusion_vector)).item()
        FN = torch.sum(confusion_vector == 0).item()
        
        cfm = []
        cfm.append(FN)
        cfm.append(FP)
        cfm.append(TN)
        cfm.append(TP)
        
        #print
        metrics_px = print_cfm(cfm)
        
        return cfm, metrics_px

logger = Logger(model_name='SegAN-w-SA-masked_2', data_name='Brain_Stroke')

def train(batch_size):
    G_loss_log = []
    D_loss_log = []
    
    cfm_train = []
    cfm_test = []
    
    metrics_px_train = []
    metrics_px_test = []

    test_x, test_y = next(iter(dataset_loader_test))
    test_x_1, test_y_1 = next(iter(dataset_loader_test_1))
  
    for epoch in range(num_epochs):
        
        start_time = time()
        
        # logger recording the losses for each minibatch in this epoch
        G_batch_loss = []
        D_batch_loss = []
        
        for batch, (x, y) in enumerate(dataset_loader_train):  # enumerate through minibatches
            
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            
            # train discriminator
            D_loss = train_D(x, y)
            
            # train generator
            G_loss = train_G(x, y)

            # add batch losses to logger
            D_batch_loss.append(D_loss)
            G_batch_loss.append(G_loss)

            #random plot images
            number = random.randint(0,x.size()[0]-3)      
            x_train = x[number:number+3]
            y_train = y[number:number+3]
            
            number = random.randint(0,test_x.size()[0]-3)  
            x_test = test_x[number:number+3]
            y_test = test_y[number:number+3]
            
            number = random.randint(0,test_x_1.size()[0]-3)  
            x_test_1 = test_x_1[number:number+3]
            y_test_1 = test_y_1[number:number+3]
            
        torch.cuda.empty_cache()
        
        #compute average loss of this epoch
        G_loss_log.append(sum(G_batch_loss) / len(G_batch_loss))
        D_loss_log.append(sum(D_batch_loss) / len(D_batch_loss))
        
        clear_output()
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch, num_epochs, batch, num_batches))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(D_loss_log[epoch], G_loss_log[epoch]))
        
        test_images(x_train, y_train, epoch, num_epochs)
        test_images(x_test, y_test, epoch, num_epochs)
        test_images(x_test_1, y_test_1, epoch, num_epochs)
        
        torch.cuda.empty_cache()

        tmp_cfm_train, tmp_metrics_px_train = test_segmentation(dataset_loader_train)
        tmp_cfm_test, tmp_metrics_px_test = test_segmentation(dataset_loader_test)
        
        torch.cuda.empty_cache()
        
        cfm_train.append(tmp_cfm_train)
        cfm_test.append(tmp_cfm_test)
        metrics_px_train.append(tmp_metrics_px_train)
        metrics_px_test.append(tmp_metrics_px_test)
        
        FN = tmp_cfm_train[0]
        FP = tmp_cfm_train[1]
        TN = tmp_cfm_train[2]
        TP = tmp_cfm_train[3]
        binary = np.array([[TN, FP], [FN, TP]])
        fig, ax = plot_confusion_matrix(conf_mat=binary, show_absolute=True, show_normed=True, colorbar=True)
        fig.savefig('cmt-train.png')
        plt.show()
        plt.close()
        
        FN = tmp_cfm_test[0]
        FP = tmp_cfm_test[1]
        TN = tmp_cfm_test[2]
        TP = tmp_cfm_test[3]
        binary = np.array([[TN, FP], [FN, TP]])
        fig, ax = plot_confusion_matrix(conf_mat=binary, show_absolute=True, show_normed=True, colorbar=True)
        fig.savefig('cmt-test.png')
        plt.show()
        plt.close()
        
        if (epoch+1)%10 == 0:
            logger.save_models(G, D, epoch)
        
        if epoch != 0:            
            
            fig = plt.figure()
            plt.plot(D_loss_log, label="D Losses")
            plt.plot(G_loss_log, label="G Losses")
            plt.legend(frameon=False)
            plt.show()
            fig.savefig('losses.png')
            plt.close()
            
            fig = plt.figure()
            plt.plot([row[0] for row in metrics_px_train], label="Error Rate Train")
            plt.plot([row[0] for row in metrics_px_test], label="Error Rate Test")
            plt.legend(frameon=False)
            plt.show()
            fig.savefig('error-rate.png')
            plt.close()
            
            fig = plt.figure()
            plt.plot([row[1] for row in metrics_px_train], label="Accuracy Train")
            plt.plot([row[1] for row in metrics_px_test], label="Accuracy Test")
            plt.legend(frameon=False)
            plt.show()
            fig.savefig('accuracy.png')
            plt.close()
            
            fig = plt.figure()
            plt.plot([row[2] for row in metrics_px_train], label="Sensibility Train")
            plt.plot([row[2] for row in metrics_px_test], label="Sensibility Test")
            plt.legend(frameon=False)
            plt.show()
            fig.savefig('sensibility.png')
            plt.close()
            
            fig = plt.figure()
            plt.plot([row[3] for row in metrics_px_train], label="Specificity Train")
            plt.plot([row[3] for row in metrics_px_test], label="Specificity Test")
            plt.legend(frameon=False)
            plt.show()
            fig.savefig('specificity.png')
            plt.close()
            
            fig = plt.figure()
            plt.plot([row[4] for row in metrics_px_train], label="Precision Train")
            plt.plot([row[4] for row in metrics_px_test], label="Precision Test")
            plt.legend(frameon=False)
            plt.show()
            fig.savefig('precision.png')
            plt.close()
            
            fig = plt.figure()
            plt.plot([row[5] for row in metrics_px_train], label="Dice Train")
            plt.plot([row[5] for row in metrics_px_test], label="Dice Test")
            plt.legend(frameon=False)
            plt.show()
            fig.savefig('dice.png')
            plt.close()
            
            G_loss_log_ = np.asarray(G_loss_log)
            np.savetxt("G_loss_log.csv", G_loss_log_, delimiter=",")
            D_loss_log_ = np.asarray(D_loss_log)
            np.savetxt("D_loss_log.csv", D_loss_log_, delimiter=",")
            cfm_train_ = np.asarray(cfm_train)
            np.savetxt("cfm_train.csv", cfm_train_, delimiter=",")
            cfm_test_ = np.asarray(cfm_test)
            np.savetxt("cfm_test.csv", cfm_test_, delimiter=",")
            metrics_px_train_ = np.asarray(metrics_px_train)
            np.savetxt("metrics_px_train.csv", metrics_px_train_, delimiter=",")
            metrics_px_test_ = np.asarray(metrics_px_test)
            np.savetxt("metrics_px_test.csv", metrics_px_test_, delimiter=",")

        
        elapsed_time = time() - start_time
        print('Total time used: %d seconds' % elapsed_time)

    return G_loss_log, D_loss_log, cfm_train, cfm_test, metrics_px_train, metrics_px_test


# record time used
from time import time
start_time = time()
G_loss_log, D_loss_log, cfm_train, cfm_test, metrics_px_train, metrics_px_test = train(batch_size)
elapsed_time = time() - start_time
print('Total time used: %d seconds' % elapsed_time)