import torch
import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from models import EncoderCNN, DecoderRNN, LabelClassifier
from PIL import Image
from data_loader import dict_path, total_label_numbers
from model_train import (feature_gen_path, caption_gen_path, num_layers,
     model_extension, NUM_EPOCHS, path_trained_model, lstm_output_size,
     word_embedding_size, input_resnet_size, label_classifier_size, label_gen_path)
import cv2 as cv
import json

test_dir_path = "./test_res"
# Building the Test Directory
if not os.path.exists(test_dir_path):
    os.makedirs(test_dir_path)

dict_path = "dictionary.pkl"

# file names
caps_fname = "captions.json"
catsinfo_fname = "categories_info.json"
imgsinfo_fname = "images_info.json"
label_fname = "labels.json"

# confirm existance of data directory -
root_data_path = "./data" 


################# Saved Files Path #######################
label_save_path = "./labels.txt"
captions_save_path = "./captions.txt"

############# To be replace with "labeling_images" ###########
imgs_path = os.path.join(root_data_path, "images/val2017")  
path_chk = os.path.exists(root_data_path) 
assert(path_chk) 

img_path_captioner = os.path.join(root_data_path, "images/val2017")

try:
    with open(os.path.join(root_data_path, caps_fname)) as file:
        data_caps = json.load(file)
    with open(os.path.join(root_data_path, catsinfo_fname)) as file:
        data_catsinfo = json.load(file)
    with open(os.path.join(root_data_path, imgsinfo_fname)) as file:
        data_imgs = json.load(file) 
    with open(os.path.join(root_data_path, label_fname)) as file:
        data_labels = json.load(file) 

except: 
     print("Something went wrong during loading of files, Terminating ...")
     exit

# # Prepare an image
# image_path = "./data/images/val2017/000000001584.jpg"
# #image_path = "./data/my_test_images/4.jpg"
# image = Image.open(image_path) 
# image_tensor = transform(image).unsqueeze(0).to(device)


# Labeling Threshold
labeler_threshold = 0.5
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing(resizeing and normalization)
transform = transforms.Compose([
    transforms.Resize((224,224), Image.LANCZOS),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])


# Function for Testing the Labler.
def test_labler():
    # Accuracy Example-Based [2]
    acc = 0 
    # Precision Example-Based
    prc = 0 

    output_file = open(label_save_path, 'w')
    # Some examples don't have any labels (all set to zero) this counter is for 
    # neglecting those in Example-Based Accuracy Calculation
    omit_exams = 0 
    omit_exams_prec = 0 
    with torch.no_grad():
        model = LabelClassifier(label_classifier_size, total_label_numbers).eval().to(device)
        
        model_path = os.path.join(path_trained_model, label_gen_path + model_extension)
        # Loading model weights from pkl files 
        model.load_state_dict(torch.load(model_path))

        # Iterate over test Images provided by data_imgs.
        for iter, data_img in enumerate(data_imgs):
            # Loading the Image File
            inp_img = Image.open(os.path.join(imgs_path, data_img['file_name']))
            # Since the model assumes a batch number ... 
            image_tensor = transform(inp_img).unsqueeze(0).to(device) 

            # Finding the matched label for current image id.
            match_labels = [label['category_id'] for label in data_labels if label['image_id'] == data_img['id']]

            # Map the matches to multi-hot vector.
            ground_truth = np.zeros(total_label_numbers, dtype=int)
            for match_label in match_labels:
                ground_truth[match_label] = 1

            output = model(image_tensor)
            # Thresholding the output, Note that we should take the variable from GPU
            output = (output>labeler_threshold).int()[0].cpu().numpy()
            #print("Predicted Labels:",output)
            #print("Ground Truth:", ground_truth)

            # Printing the labels to output files
            # Retreiving the Label Number, See the Encoding section
            ret_labels = np.where(output != 0)[0]
            output_file.write(np.array2string(ret_labels) + "\n")

            ################# Calculating the Precision and Accuracy(Example-based) Methods ############
            # intersection and union
            inter = 0 
            union = 0 
            # Iterate over predicted and groundturth labels
            for i in range(total_label_numbers):
                if output[i] == ground_truth[i] and output[i] == 1:
                    inter += 1
                    union += 1  
                elif output[i] != ground_truth[i]: 
                    union += 1

            try:
                acc += (inter/union) 
            except: 
                print("This sample Cause me a problem, led to all zeros predicted and ground truth.")
                omit_exams += 1 
            
            output_sum = np.sum(output)
            if output_sum == 0:
                print("Zero Some Output Not Included in Calculations ... ")
                omit_exams_prec += 1 
            else:
                prc += (inter/output_sum)
                
            if iter == 300:
                print("Accuracy(Example-based see [2]) :", acc/(iter+1 - omit_exams))
                print("Precision(Example-based):", prc/(iter+1-omit_exams_prec) )
                break 



def test_captioner():   
    # Load vocabulary wrapper
    with open(dict_path, 'rb') as file:
        dictionary = pickle.load(file)

    # Build models
    encoder = EncoderCNN(word_embedding_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(word_embedding_size, lstm_output_size, len(dictionary[0]), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder_model_path = os.path.join(path_trained_model ,feature_gen_path + str(NUM_EPOCHS)+ model_extension) 
    decoder_model_path = os.path.join(path_trained_model ,caption_gen_path + str(NUM_EPOCHS)+ model_extension)
    encoder.load_state_dict(torch.load(encoder_model_path))
    print("Feature Extractor Model Loaded Successfully")
    decoder.load_state_dict(torch.load(decoder_model_path))
    print("Caption Generator Loaded Successfully") 

    # Open Caption Saver File
    output_file = open(captions_save_path, 'w')
    
    for iter, data_img in enumerate(data_imgs): 
        img_path = os.path.join(imgs_path, data_img['file_name'])
        ### Change ###
        inp_img = Image.open(img_path)
        # Since the model assumes a batch number ... 
        image_tensor = transform(inp_img).unsqueeze(0).to(device) 

        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()       
        
        idx2word = dictionary[1]
        # Convert word_ids to words
        sampled_caption = []
        for idx in sampled_ids:
            word = idx2word[idx] 
            sampled_caption.append(word)
            if word == 'END':
                break
        sentence = ' '.join(sampled_caption)
        
        # Writing the Caption to File
        output_file.write(sentence + "\n")

        # Print out the image and the generated caption
        print ("Caption: ", sentence)
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        window_name = "Sample Image with Caption as Overlay" 
        cv.imshow(window_name, image)
        cv.displayOverlay(window_name, sentence)
        cv.waitKey(0)

        if iter == 10:
            return  
    
if __name__ == '__main__':

    # Testing the Captioner
    test_captioner()

    # Testing the Labeler
    #test_labler()