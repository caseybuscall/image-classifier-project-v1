#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import time

def get_input():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image' , type=str, default='*.jpg', help='Image for Inference')
    parser.add_argument('--save' , type=str, default='checkpoint.pth', help='path to saved model')
    parser.add_argument('--cat_to_name' , type=str, default='cat_to_name.json', help='file for mapping')
    parser.add_argument('--top_k' , type=int, default='5', help='number of best probabilities')
    parser.add_argument('--gpu' , type=bool, default=False, help='use GPU if available')
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    
    # Loading the image
    im = Image.open(image)
    
    # Preprocess the image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize])
    
    im = Image.open(image)
    im_tensor = preprocess(im)
    
    return im, im_tensor

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Put the model in evaluation mode
    model.eval()
    
    image = process_image(image_path)
    
    image = image.unsqueeze(0)
    
    # turn off gradient calculation
    with torch.no_grad():
        output = model(image)
        top_prob, top_labels = torch.topk(output, topk)
    
    # top pobabilitiess
    top_prob = top_prob.exp()

    class_to_idx_inv = {model.class_to_idx[key]: int(key) for key, value in model.class_to_idx.items()}
    mapped_classes = []

    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])

    return top_prob.numpy()[0], mapped_classes
    
def main():
    args = get_input()
        
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
num_categories = max(map(int, cat_to_name.keys()))
        
model = load_checkpoint(args.save)
print(model.classifier)
print("checkpoint loaded")
    
image = args.image
    
# Calculating the topk probabilities, top 5
top_prob, top_classes = predict(img, model, args.top_k)
    
# List to store the predicted labels
labels = []
for classes in top_classes:
     labels.append(cat_to_name[str(class_idx)] + ' (class ' + str(class_idx) + ')')
    
if __name__ == "__main__":
    main()