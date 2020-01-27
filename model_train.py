
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from dataset import get_loader 
from models import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from data_loader import * 
from PIL import Image

BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARN_RATE = 0.001
num_layers = 1
lstm_output_size = 512 
word_embedding_size = 256
input_resnet_size = 256
path_trained_model = os.path.join(root_data_path, "trained_model")

# Path where we save the trained models.
caption_gen_path = "captioner"
feature_gen_path = "feature-extractor-" 
model_extension = ".ckpt"


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
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
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, NUM_EPOCHS, i, total_step, loss.item(), np.exp(loss.item()))) 
                
        # Sace model after each epoch ... 
        torch.save(decoder.state_dict(), os.path.join(
            path_trained_model, 'captioner{}.ckpt'.format(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(
            path_trained_model, 'feature-extractor-{}.ckpt'.format(epoch+1)))

if __name__ == '__main__':
    train() 

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
#     parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
#     parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
#     parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
#     parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
#     parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
#     parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
#     # Model parameters
#     parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
#     parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
#     parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
#     parser.add_argument('--num_epochs', type=int, default=5)
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--num_workers', type=int, default=2)
#     parser.add_argument('--learning_rate', type=float, default=0.001)
#     args = parser.parse_args()
#     print(args)
#     main(args)