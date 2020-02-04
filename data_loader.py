import numpy as np 
import json 
import os
import matplotlib.pyplot as plt
import cv2 as cv
# Regular Expressions as Preprocessing texts 
from collections import Counter
import re
import sys
import pickle

# Will be used in generating the dictionary, to neglect words which their frequency is less that @acceptable_word_freq 
word_counter = Counter() 
acceptable_word_freq = 5
# The order is important since later on we generate the empty tensor with torch.zeros which means index 0 should be pad 
add_word = ['PAD', 'START', 'END', 'UNKOWN']
# The File Which the dictionary will be generated
dict_path = "dictionary.pkl"

# file names
# caps_fname = "captions.json"
# catsinfo_fname = "categories_info.json"
# imgsinfo_fname = "images_info.json"
caps_fname = "additional_captions.json"
imgsinfo_fname = "additional_images_info.json"
label_fname = "additional_labels.json"

catsinfo_fname = "categories_info.json"

# confirm existance of data directory 
root_data_path = "./data" 
#imgs_path = os.path.join(root_data_path, "images/val2017")  
imgs_path = os.path.join(root_data_path, "MoreImages")  
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
    print("Something went wrong during loading of files")

# getting the total number of labels, later will be required for the multi-hot codding. 
print("Initializing Label Processing ...")
labels = set() 
for iter, label in enumerate(data_label):
    labels.add(label['category_id'])
    if iter % 10000 == 0:
        print("Labels [{}/{}]:".format(iter, len(data_label)))

#total_label_numbers = len(labels)
total_label_numbers = 91
print("Label Processing Finished.")

def gen_dictionary(data_caps, len_caps):
    
    print("\nInitiating the Dictionary Generation Process ...")

    max_len = 0 
    # Iterate over Captions to build the dictionary of words. 
    word2idx = dict() 
    idx2word = dict() 
    print("Processing Captions ...")
    for iter, data_cap in enumerate(data_caps): 
        cap =  str(data_cap['caption']) 
        cap_words = re.findall(r'\w+', cap.lower())
        len_words = len(cap_words) 
        if len_words > max_len: 
            max_len = len_words
        word_counter.update(cap_words)

        if iter % 5000 == 0:
            print("{} Out of {} is Processed".format(iter, len_caps))

    words =[word for word, cnt in word_counter.items() if cnt >= acceptable_word_freq]
    words = add_word + words 
    for idx, word in enumerate(words):
        word2idx[word] = idx
        idx2word[idx] = word 
    
    dictionary = [word2idx, idx2word] 
    with open(dict_path, 'wb') as file:
        pickle.dump(dictionary, file)
    
    print("Maximum Number of Words in a Caption is {}".format(max_len))
    print("Dictionary Generation Finished.") ; 
    

# This function will show a sample from provided dataset.
def data_sampler(): 
    # length of files
    len_imgs = len(data_imgs) 
    len_caps = len(data_caps) 
    len_labels = len(data_label)

    print("Lengths of Captions is: {}\nLength of Image Info is: {}\nLenght of labels is: {}\nLenght of Category info is: {}\n\n"
    .format(len_caps, len_imgs, len_labels, len(data_catsinfo), len(data_label)))
    
    # generating dictionary
    gen_dictionary(data_caps, len_caps)

    # print some samples from data sets
    # image wise data drawer. 
    rand_idx = np.random.randint(0, len(data_imgs), dtype=int)
    rand_imgid = data_imgs[rand_idx]['id']
    rand_filename = data_imgs[rand_idx]['file_name']


    label_matches = [label['category_id'] for idx, label in enumerate(data_label) if label['image_id'] == rand_imgid ]
    print(label_matches) 
    caps_matches = [cap['caption'] for idx, cap in enumerate(data_caps) if cap['image_id'] == rand_imgid ]
    print(caps_matches)

    img_path = os.path.join(imgs_path, rand_filename) 
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    cv.imshow("Sample", img)
    if cv.waitKey(0) == ord('e'):
        cv.destroyAllWindows() 


    print("Done and Done")
    return img, img_path


if __name__ == '__main__': 

    data_sampler()

    with open(dict_path, 'rb') as file: 
        data = pickle.load(file)

