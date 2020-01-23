import numpy as np 
import json 
import os
import matplotlib.pyplot as plt
import cv2 as cv

# file names
caps_fname = "captions.json"
catsinfo_fname = "categories_info.json"
imgsinfo_fname = "images_info.json"
label_fname = "labels.json"


# confirm existance of data directory 
root_data_path = "./data" 
imgs_path = os.path.join(root_data_path, "images/val2017")  
path_chk = os.path.exists(root_data_path) 
assert(path_chk) 

try:
    with open(os.path.join(root_data_path, caps_fname)) as file:
        data_caps = json.load(file)
    with open(os.path.join(root_data_path, catsinfo_fname)) as file:
        data_catsinfo = json.load(file)
    with open(os.path.join(root_data_path, imgsinfo_fname)) as file:
        data_imgs = json.load(file) 
    with open(os.path.join(root_data_path, label_fname)) as file:
        data_label = json.load(file) 

except: 
    print("Something went wrong loading the files")

print("Lengths of Captions is: {}\nLength of Image Info is: {}\nLenght of labels is: {}\nLenght of Category info is: {}"
.format(len(data_caps), len(data_imgs), len(data_label), len(data_catsinfo)))

# print some samples from data sets
# image wise data drawer. 
rand_idx = np.random.randint(0, len(data_imgs), dtype=int)
rand_imgid = data_imgs[rand_idx]['id']
rand_filename = data_imgs[rand_idx]['file_name']


label_matches = [label['category_id'] for idx, label in enumerate(data_label) if label['image_id'] == rand_imgid ]
print(label_matches) 
caps_matches = [cap['caption'] for idx, cap in enumerate(data_caps) if cap['image_id'] == rand_imgid ]
print(caps_matches)


img = cv.imread(os.path.join(imgs_path, rand_filename), cv.IMREAD_COLOR)
cv.imshow("Sample", img)
if cv.waitKey(0) == ord('e'):
    cv.destroyAllWindows() 


print("Done and Done")