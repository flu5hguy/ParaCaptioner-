import torch
import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from models import EncoderCNN, DecoderRNN
from PIL import Image
from data_loader import dict_path
from model_train import (feature_gen_path, caption_gen_path, num_layers,
     model_extension, NUM_EPOCHS, path_trained_model, lstm_output_size,
     word_embedding_size, input_resnet_size)
import cv2 as cv

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def test():
    # Image preprocessing(resizeing and normalization)
    transform = transforms.Compose([
        transforms.Resize((224,224), Image.LANCZOS),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
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

    # Prepare an image
    #image_path = "./data/images/val2017/000000001584.jpg"
    image_path = "./data/my_test_images/4.jpg"
    image = Image.open(image_path) 
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    idx2word = dictionary[1]
    # Convert word_ids to words
    sampled_caption = []
    for idx in sampled_ids:
        word = idx2word[idx] 
        sampled_caption.append(word)
        if word == 'END':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    window_name = "Sample Image with Caption as Overlay" 
    cv.imshow(window_name, image)
    cv.displayOverlay(window_name, sentence)
    cv.waitKey(0)
    
if __name__ == '__main__':
    test()