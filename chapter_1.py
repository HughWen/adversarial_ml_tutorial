# %%
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# read the image, resize to 224 and convert to PyTorch Tensor
pig_img = Image.open('pig.jpg')
preprocess = transforms.Compose([
    transforms.Resize(244),
    transforms.ToTensor(),
])
pig_tensor = preprocess(pig_img)[None, :, :, :]

# plot image (note that numpy using HWC whereas Pytorch uses CHS, so we need to convert)
plt.imshow(pig_tensor[0].numpy().transpose(1, 2, 0))
plt.show()

# %%
import torch
import torch.nn as nn
from torchvision.models import resnet50


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


# values are standard normalization for ImageNet image
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Load the pre-trained ResNet50, and put into evaluation mode (necessary to e.g. turn off batchnorm)
model = resnet50(pretrained=True)
model.eval()

# form prediction
pred = model(norm(pig_tensor))

# %%
import json


with open("imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
print(imagenet_classes[pred.max(dim=1)[1].item()])


# %%
