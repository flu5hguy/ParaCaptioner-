import torch
import torch.nn as nn 
import torchvision.models.detection as detection 
from data_loader import data_sampler
from PIL import Image
import cv2 as cv
import numpy as np 
from torchvision import * 


# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         faster_rcnn = detection.FasterRCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


def drawer(img, label, bbox, score): 
    # First move from cuda to cpu then convert to numpy 
    bbox_np = bbox.cpu().numpy() 
    label = label.cpu().numpy() 

    color = (0,255,0)
    # Extracting the points (note that the opencv accepts points as tuple)
    pts1 = tuple((bbox_np[:2]).astype(np.int))
    pts2 = tuple((bbox_np[-2:]).astype(np.int))
    img = cv.rectangle(img, pts1, pts2, color, 2)
    img = cv.putText(img, str(label), pts1, cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, color, 2, cv.LINE_AA)

    cv.imshow("Test", img) 
    cv.waitKey(0) 


model = detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
modules = list(model.children())
for iter, module in enumerate(modules): 

    print("\n\n#### {} #### ".format(iter))
    print(module)

# no Gradient needed for memory efficiency
with torch.no_grad() :  
    # putting the model to evaluation mode
    model.eval()

    img_numpy, img_path = data_sampler()
    img_pil = Image.open(img_path).convert('RGB') 

    # Defining the preprocessor (Convert to Tensor and range between 0 to 1)
    pre_processor = transforms.Compose([transforms.ToTensor()])
    img_trans = pre_processor(img_numpy).to(device)

    imgs = list() 
    imgs.append(img_trans)

    # Predictions
    preds = model(imgs) 

    print(preds) 

    for pred in preds: 
        for bbox, label, score in zip(pred['boxes'], pred['labels'], pred['scores']) : 
            print("#### ")
            print(label) 
            print(bbox) 
            print(score)
            drawer(img_numpy, label, bbox, score)