import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from dataset import get_loader, data_loader_labeler
from models import EncoderCNN, DecoderRNN, LabelClassifier
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from data_loader import * 
from PIL import Image

BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_EPOCHS_LABLER = 10
LEARN_RATE = 0.001
num_layers = 1
lstm_output_size = 512 
word_embedding_size = 256
input_resnet_size = 256
label_classifier_size = 256
path_trained_model = os.path.join(root_data_path, "trained_model")

# Path where we save the trained models.
caption_gen_path = "captioner"
feature_gen_path = "feature-extractor-" 
label_gen_path = "label-generator-"
model_extension = ".ckpt"


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function which Train the Label Generator
def train_labelclassifier():
    print("Training the LabelClassifier ...")
    # Create model directory
    if not os.path.exists(path_trained_model):
        os.makedirs(path_trained_model)
    
    # Image preprocessing, first resize the input image then do normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.Resize((input_resnet_size,input_resnet_size), interpolation=Image.ANTIALIAS),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    data_loader = data_loader_labeler(
        imgs_path, 
        data_label, 
        data_imgs, 
        total_label_numbers, 
        transform,
        BATCH_SIZE,
        True,
        2)

    # Build Model
    model = LabelClassifier(label_classifier_size, total_label_numbers).to(device)

    # Model Criterion
    # BinaryCrossEntropy with Sigmoid
    criterion = nn.BCELoss()
    params = list(model.linear.parameters()) + list(model.linear_label.parameters()) \
           + list(model.bn.parameters()) + list(model.resnet.avgpool.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARN_RATE)

    # Begin Training
    total_steps = len(data_loader)
    for epoch in range(NUM_EPOCHS_LABLER):
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
    
            model.zero_grad() 
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                        .format(epoch, NUM_EPOCHS, i, total_steps, loss.item()))

    torch.save(model.state_dict(), os.path.join(path_trained_model, '{}.ckpt'.format(label_gen_path)))
    
# Function for Training the Captioner
def train_captioner():
    print("Training The Capitoner ... ") 
    # Create model directory
    if not os.path.exists(path_trained_model):
        os.makedirs(path_trained_model)
    
    # Image preprocessing, first resize the input image then do normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.Resize((input_resnet_size,input_resnet_size), interpolation=Image.ANTIALIAS),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Loading pickle dictionary
    with open(dict_path, 'rb') as file:
        dictionary = pickle.load(file)
    
    # Build data loader
    data_loader = get_loader(imgs_path, data_caps, dictionary, 
                             transform, BATCH_SIZE,
                             shuffle=True, num_workers=2) 

    # Build the models
    encoder = EncoderCNN(word_embedding_size).to(device)
    decoder = DecoderRNN(word_embedding_size, lstm_output_size, len(dictionary[0]), num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARN_RATE)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(NUM_EPOCHS):
        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, NUM_EPOCHS, i, total_step, loss.item(), np.exp(loss.item()))) 
                
        # Sace model after each epoch ... 
        torch.save(decoder.state_dict(), os.path.join(
            path_trained_model, 'captioner{}.ckpt'.format(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(
            path_trained_model, 'feature-extractor-{}.ckpt'.format(epoch+1)))


### Main Call 
if __name__ == '__main__':
    # Train the Captioner Network
    train_captioner()

    # Train the Label Classifier
    #train_labelclassifier() 
