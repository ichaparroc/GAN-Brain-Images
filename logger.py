import os
import glob
import torch
import errno
import numpy as np
import torchvision.utils as vutils
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

class Logger:

  def __init__(self, model_name, data_name):
    self.model_name = model_name
    self.data_name = data_name

    self.data_subdir = '{}_{}'.format(model_name, data_name)

  def save_torch_images(self, images, epoch, n_batch, plot_horizontal=True):

    horizontal_grid = vutils.make_grid(images, nrow=4, normalize=True, scale_each=False)
    out_dir = './data/images/{}'.format(self.data_subdir)
    Logger._make_dir(out_dir)

    # Plot and save horizontal
    fig = plt.figure(figsize=(4,4),dpi=128)
    plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
    plt.axis('off')
    display.display(plt.gcf())
    plt.close()

    out_dir = './data/images/{}'.format(self.data_subdir)
    Logger._make_dir(out_dir)
    fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir, self.data_subdir, epoch, n_batch))

  def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

    # var_class = torch.autograd.variable.Variable
    if isinstance(d_error, torch.autograd.Variable):
      d_error = d_error.data.cpu().numpy()
    if isinstance(g_error, torch.autograd.Variable):
      g_error = g_error.data.cpu().numpy()
    if isinstance(d_pred_real, torch.autograd.Variable):
      d_pred_real = d_pred_real.data
    if isinstance(d_pred_fake, torch.autograd.Variable):
      d_pred_fake = d_pred_fake.data

    print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, n_batch, num_batches))
    print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
    print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

  def save_models(self, generator, discriminator, epoch):
    out_dir = './data/models/{}'.format(self.data_subdir)
    Logger._make_dir(out_dir)
    torch.save(generator.state_dict(),
           '{}/G_epoch_{}'.format(out_dir, epoch))
    torch.save(discriminator.state_dict(),
           '{}/D_epoch_{}'.format(out_dir, epoch))

  @staticmethod
  def _step(epoch, n_batch, num_batches):
    return epoch * num_batches + n_batch

  @staticmethod
  def _make_dir(directory):
    try:
      os.makedirs(directory)
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise