import os
import numpy as np
from PIL import Image

import torch
import torchvision


def store_raw_image(tensor, folder, filename):
    os.makedirs(folder, exist_ok=True)
    im0 = torchvision.transforms.ToPILImage()(tensor)
    im0.save(os.path.join(folder, filename))

def store_classification_image(tensor, folder, filename):
    os.makedirs(folder, exist_ok=True)
    tensor = torch.argmax(tensor, dim=0)
    
    store_raw_image(tensor/20)



'''
gt0 = torchvision.transforms.ToPILImage()(GT[0, ...])
im0 = torchvision.transforms.ToPILImage()(images[0, ...])
os.makedirs("/content/drive/image_log/train", exist_ok=True)
gt0.save("/content/drive/image_log/train/{}_gt_or.jpg".format(i))
im0.save("/content/drive/image_log/train/{}_im.jpg".format(i))
'''