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
    cutouts[index, :, :, :] = cutout_single_image(img, length = length).reshape(1, height, width, channel)
    index += 1

  return cutouts


def clip(num, minimum, maximum):
  if num <= minimum:
    return minimum
  elif num >= maximum:
    return maximum
  else:
    return num


def cutout_single_image(img, length = 10):
  '''
  Apply to a single image

  reference: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
  '''

  img_copy = img * 1 # copy by value instead of reference

  dims = img.shape
  height = dims[0]
  width = dims[1]
  channel = dims[2]

  y = np.random.randint(height)
  x = np.random.randint(width)

  y1 = clip(y + length // 2, minimum = 0, maximum = height)
  y2 = clip(y - length // 2, minimum = 0, maximum = height)
  x1 = clip(x + length // 2, minimum = 0, maximum = width)
  x2 = clip(x - length // 2, minimum = 0, maximum = width)

  mask = torch.ones(y1 - y2, x1 - x2, channel) * int(torch.max(img))

  img_copy[y2: y1, x2: x1, :] = mask

  return img_copy
