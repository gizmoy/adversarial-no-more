'''
The Cutout Method
'''
import numpy as np
import torch

def cutout(x, length = 10):
  '''
  Cutout Method to add noise to training

  Args:
    :param x: input tensor
    :param length: length of the cutout square
  '''

  dims = x.shape
  size = dims[0]
  height = dims[1]
  width = dims[2]
  channel = dims[3]

  cutouts = torch.zeros(size, height, width, channel)

  index = 0
  for img in x:
    cutouts[index, :, :, :] = __cutout(img, length = length).reshape(1, height, width, channel)
    index += 1

  return cutouts


def __clip(num, minimum, maximum):
  if num <= minimum:
    return minimum
  elif num >= maximum:
    return maximum
  else:
    return num


def __cutout(img, length = 10):
  '''
  Apply to a single image

  reference: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
  '''

  dims = img.shape
  height = dims[0]
  width = dims[1]
  channel = dims[2]

  mask = torch.ones(height, width)
  
  y = np.random.randint(height)
  x = np.random.randint(width)

  y1 = __clip(y + length // 2, minimum = 0, maximum = height)
  y2 = __clip(y - length // 2, minimum = 0, maximum = height)
  x1 = __clip(x + length // 2, minimum = 0, maximum = width)
  x2 = __clip(x - length // 2, minimum = 0, maximum = width)

  mask[y1: y2, x1: x2] = 0

  mask = torch.stack([mask] * channel, dim = 2).to(img.device)

  img = img * mask

  return img
