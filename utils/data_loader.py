import os, random, h5py
from random import shuffle

import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F



class H5pyDataset(data.Dataset):
	def __init__(self, root, exp_name, mode, out_ch, data_mode):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.mode = mode
		self.out_ch = out_ch

		self.exp_name = exp_name
		self.data_mode = data_mode

		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		fp = h5py.File(image_path, "r")
		if len(fp['data'].shape) == 2:
			n_channel = 1
		else:
			n_channel = fp['data'].shape[2]

		image = np.array(fp['data'])
		gt    = np.array(fp['annot'])
		fp.close()
		
		image = Image.fromarray(image.astype(np.uint8))
		image = T.ToTensor()(image)
		image = T.Normalize((.5,)*n_channel, (.5,)*n_channel)(image)

		if self.data_mode=='location':
			image = image.view(*image.shape[1:])
			idx_i = torch.linspace(0, 1, image.size(1)).repeat(image.size(0))
			idx_j = torch.linspace(0, 1, image.size(0)).repeat(image.size(1)).transpose(1,0)
			print(image.shape, idx_i.shape, idx_j.shape)
			image = torch.stack([image, idx_i, idx_j], dim=0)


		if self.out_ch == 2:
			gt = gt>0
		expand = []
		for i in range(self.out_ch):
			expand.append(gt==i)
		gt = np.stack(expand)
		gt = torch.Tensor(gt)

		return image, gt

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)



def get_loader(config, mode='train'):
	"""Builds and returns Dataloader."""
	
	dataset = H5pyDataset(
		exp_name = config.name,
		root = os.path.join(config.data_root_path, mode),
		mode = mode,
		out_ch = config.output_ch,
		data_mode = config.data_mode)
	
	data_loader = data.DataLoader(
		dataset=dataset,
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=config.num_workers)

	return data_loader
