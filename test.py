import torch
import torchvision

model_fastercnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False) 
modules_2 = list(model_fastercnn.children())
for iter, module in enumerate(modules_2):
    print("###### {} ######".format(iter))
    print(module) 

print(model_fastercnn.roi_heads.box_predictor)

# model = torchvision.models.resnet50(pretrained=False) 
# modules = list(model.children())
# for iter, module in enumerate(modules):
#     print("###### {} #####".format(iter)) 
#     print(module)

# print(type(model))
# print(model.fc.in_features) 