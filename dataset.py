import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import json
import re

# the length of file names in coco image, the id will be prefixed by 
# leading zeros to reach length of $name_len
name_len = 12
imgs_extension = ".jpg"

# The Default colate function is sufficient...
class CoCoDatasetLabeler(data.Dataset):
    def __init__(self, root, data_label, data_images, num_labels, transform=None):
        self.root = root 
        self.data_label = data_label 
        self.data_images = data_images
        self.num_labels = num_labels
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.data_images[index]['file_name']
        img_id = self.data_images[index]['id']

        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Finidng labels corresponding to image id
        labels = [label['category_id'] for iter, label in enumerate(self.data_label) if label['image_id'] == img_id]

        label_multi_hot = torch.zeros(self.num_labels)
        for idx in labels:
            label_multi_hot[idx] = 1.
        
        return image, label_multi_hot

        
    def __len__(self):
        return len(self.data_images) 

def data_loader_labeler(
    root, 
    data_label, 
    data_images, 
    num_labels, 
    transform, 
    batch_size,
    shuffle,
    num_workers):
    coco_label = CoCoDatasetLabeler(root=root, data_label=data_label, data_images=data_images, num_labels=num_labels, transform=transform)

    data_loader = torch.utils.data.DataLoader(
         dataset=coco_label,
         batch_size=batch_size, 
         shuffle=shuffle, 
         num_workers=num_workers,
         pin_memory=True)

    return data_loader 


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, data_cap, dictionary, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.word2idx = dictionary[0]
        self.idx2word = dictionary[1]
        self.transform = transform
        self.data_cap = data_cap

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption = self.data_cap[index]['caption']
        img_id = self.data_cap[index]['image_id']
        # Following is for getting the imgname from it's id
        path = str(img_id).zfill(name_len) + imgs_extension

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word idxs.
        words = re.findall(r'\w+', str(caption).lower())

        caption = []
        caption.append(self.word2idx['START'])
        caption.extend([self.word2idx[word] for word in words if word in self.word2idx])
        caption.append(self.word2idx['END'])
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        # The input to the network is pair of image and caption
        # since every caption in coco dataset have multiple caption
        # Thus we put base of indexing as number of captions
        return len(self.data_cap) 


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, data_cap, dictionary, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       data_cap= data_cap,
                       dictionary= dictionary,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader