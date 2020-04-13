'''
Random Noise
'''
import torch

def random_noise(clean_data):
  ''' 
  Generate Random Noise as Images

  Args:
    :param clean_data: input tensor
  '''

  maximum = int(torch.max(clean_data))
  dims = tuple(clean_data.size())
  return torch.FloatTensor(dims[0], dims[1], dims[2], dims[3]).uniform_(0, maximum)
