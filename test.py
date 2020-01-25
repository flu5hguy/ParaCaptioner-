import torch 
import torchvision
model = torchvision.models.resnet50(pretrained=False) 
modules = list(model.children())
for iter, module in enumerate(modules):
    print("###### {} #####".format(iter)) 
    print(module)