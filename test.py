import torch
import torchvision

# model_fastercnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
# modules_2 = list(model_fastercnn.children())
# for iter, module in enumerate(modules_2):
#     print("###### {} ######".format(iter))
#     print(module) 

# print(model_fastercnn.roi_heads.box_predictor)

# model = torchvision.models.resnet50(pretrained=False) 
# modules = list(model.children())
# for iter, module in enumerate(modules):
#     print("###### {} #####".format(iter)) 
#     print(module)

# print(type(model))
# print(model.fc.in_features) 

# class Base():
#     def __init__(self):
#         self.name = "base"
#     def print_name(self):
#         print(self.name)
    
# class SubBase(Base):
#     def __init__(self):
#         self.name = "sub"
#     def print_name(self): 
#         super().print_name()
#         print(self.name) 

# base = Base() 
# base.print_name() 

# subbase = SubBase() 
# subbase.print_name() 

num_label = 10
batch_size = 5
truth = torch.randint(0,2, (batch_size, num_label)).float()
print(truth)
input_ = torch.randint(0,2, (batch_size, num_label)).float()

loss = torch.nn.BCELoss()

out_loss = loss(truth, input_)
print(out_loss)