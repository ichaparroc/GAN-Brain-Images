import os
import glob
import torch
import errno
import random
import torch.nn as nn
from torch.nn.utils import spectral_norm

def block(
	in_channels, out_channels, kernel_size = 4, stride = 2,
	padding = 1, transposed = False, normalize = False, activation = "None", dropout = False):
	
	block = nn.Sequential()
	if transposed:
		block.add_module("ConvTranspose2d", nn.ConvTranspose2d(
			in_channels = in_channels, out_channels = out_channels, 
			kernel_size = kernel_size, stride = stride, padding = padding, bias = False))
	else:
		block.add_module("Conv2D", nn.Conv2d(
			in_channels = in_channels, out_channels = out_channels, 
			kernel_size = kernel_size, stride = stride, padding = padding, bias = False))
	if normalize:
		block.add_module("BatchNorm2d", nn.BatchNorm2d(out_channels))
	if activation == "ReLU":
		block.add_module("ReLU", nn.ReLU())
	elif activation == "LeakyReLU":
		block.add_module("LeakyReLU", nn.LeakyReLU(0.2, inplace = True))
	elif activation == "Tanh":
		block.add_module("Tanh", nn.Tanh())
	elif activation == "Sigmoid":
		block.add_module("Sigmoid", nn.Sigmoid())
	elif activation != "None":
		raise Exception("Invalid activation fuction, activation functions: ReLU, LeakyReLU, Tanh, Sigmoid")
	if dropout:
		block.add_module("Dropout", nn.Dropout2d(0.5, inplace=True))
	return block

class self_attn(nn.Module):
		def __init__(self, in_channels):
				super(self_attn, self).__init__()
				self.in_channels = in_channels
				
				self.snconv1x1_theta = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1))
				#self.snconv1x1_theta_bn = nn.BatchNorm2d(in_channels//8)
				
				self.snconv1x1_phi = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1))
				#self.snconv1x1_phi_bn = nn.BatchNorm2d(in_channels//8)
				
				self.snconv1x1_g = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1))
				#self.snconv1x1_g_bn = nn.BatchNorm2d(in_channels//2)
				
				self.snconv1x1_attn = spectral_norm(nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1))
				#self.snconv1x1_attn_bn = nn.BatchNorm2d(in_channels)
				
				self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
				self.softmax  = nn.Softmax(dim=-1)
				self.sigma = nn.Parameter(torch.zeros(1))

		def forward(self, x): #x : B x C x W x H
				_, ch, h, w = x.size()
				# Theta path
				theta = self.snconv1x1_theta(x)
				#theta = self.snconv1x1_theta_bn(theta)
				theta = theta.view(-1, ch//8, h*w)
				# Phi path
				phi = self.snconv1x1_phi(x)
				#phi = self.snconv1x1_phi_bn(phi)
				phi = self.maxpool(phi)
				phi = phi.view(-1, ch//8, h*w//4)
				# Attn map
				attn = torch.bmm(theta.permute(0, 2, 1), phi)
				attn = self.softmax(attn)
				# g path
				g = self.snconv1x1_g(x)
				#g = self.snconv1x1_g_bn(g)
				g = self.maxpool(g)
				g = g.view(-1, ch//2, h*w//4)
				# Attn_g
				attn_g = torch.bmm(g, attn.permute(0, 2, 1))
				attn_g = attn_g.view(-1, ch//2, h, w)
				attn_g = self.snconv1x1_attn(attn_g)
				#attn_g = self.snconv1x1_attn_bn(attn_g)
				# Out
				out = x + self.sigma*attn_g
				return out