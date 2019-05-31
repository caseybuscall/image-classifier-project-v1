#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir' , type=str, default='*flowers', help='Directory of data')
    parser.add_argument('--save-dir' , type=str, default='', help='Directory for saving the checkpoint')
    parser.add_argument('--arch' , type=str, default = 'densenet121', help='the model architecture for training')
    parser.add_argument('--learning_rate' , type=float, default = 0.0001, help='learning rate')
    parser.add_argument('--epochs' , type=int, default = 5, help='epochs')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if its available')
    
    return parser.parse_args()

def main():
    args = get_input_args()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Batch size
    batch_size = 64

    # Transforms for the training, validation, and test sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)    
    
    # Loading the pretrained model
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freezing the parameters so it doesn't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Creating classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, 500)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.3)),
                            ('fc2', nn.Linear(500, num_categories+1)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    model.classifier = classifier
    
    # Optimizer and loss function
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Training the model
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 25
    for epoch in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1    
        
            #optimizer.zero_grad()
            model = model.to(device)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            model.train()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    #for inputs, labels in validationloader:
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
                print(f"Epoch{epoch+1}/{epochs}.. "
                        f"Training loss: {running_loss/print_every:.4f}.. "
                        f"Validation loss: {test_loss/len(validationloader):.4f}.. "
                        f"Validation accuracy: { (accuracy/len(validationloader))*100 :.4f}%")
        running_loss = 0
        model.train() 
    
    # Saving the checkpoint 
    model.class_to_idx = train_data.class_to_idx
    
    # Checkpoint
    checkpoint = {'architecture' : args.arch,
              'classifier' : classifier,
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'state_dict' : model.state_dict()
             } 

    save_path = args.save_dir + '/checkpoint.pth'
    torch.save(checkpoint, save_path)
    print('Checkpoint Saved')

if __name__ == '__main__':
    main()