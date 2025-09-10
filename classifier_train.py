import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data_preprocessing import ClassificationDataset, crop_and_save_detections
from .species_classifier import SpeciesClassifier
import glob
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


#----------------------------------------------------------------
# TODO:
# (TRACK EVERYTHING ON wandb)
# 1. create training set for classifier on sectors taken from detected animals
#    to detect specific species ✅
# 2. build torch agumentations for this classification dataset
# 3. train classifier on these pictures and test it on test set
# 4. put everything together into a main function
# (EXTRA): build frontend for this
#----------------------------------------------------------------

# TODO: data augmentations (maybe proportional to the class distribution or define stratified sampler)
transform = transforms()
train_sampler = None

training_set = ClassificationDataset("class_dataset", transform=transform, split="train")
test_set = ClassificationDataset("class_dataset", split="test")

# TODO: split training set into train and val


# TODO: create dat
train_loader = DataLoader(training_set, batch_size=16, shuffle=True, sampler=train_sampler)
# val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=5, shuffle=False)

# training loop
def train(model, train_loader, val_loader, criterion, optimizer, epochs=100, device="cuda"):
    """Training loop for classifier model

    Args:
        model (torch.nn.Module): the model to be trained
        train_loader (torch.utils.data.DataLoader): DataLoader with the training images and labels
        val_loader (torch.utils.data.DataLoader): DataLoader with the validation images and labels
        criterion (torch.nn.Loss): the loss function
        optimizer (torch.optim.Optimizer): the optimizer for gradient descent
        epochs (int, optional): number of epochs to run the training loop over whole dataset. Defaults to 100.

    Returns:
        torch.nn.Module: trained model.
    """
    model.to(device)
    best_val_loss = 99999

    for ep in range(epochs):
        # losses initialization
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_f1 = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        
        model.train()

        for i, data in tqdm(enumerate(train_loader), desc=f"Epoch {ep+1}/{epochs} - Training"):
            # get images and labels from input data list
            imgs, labels = data[0].to(device), data[1].to(device)

            # clear the gradient accumulation of optimizer
            optimizer.zero_grad()

            # forward pass
            outputs = model(imgs)
            preds = np.argmax(outputs.detach.cpu(), 0)
            
            # backpropagation & gradient descent
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update loss
            epoch_loss += loss.item()
            # calculate metrics
            epoch_acc += accuracy_score(labels.detach().cpu(), preds) 
            epoch_f1 += f1_score(labels.detach().cpu(), preds)        

            
        # divide by number of batches
        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)
        epoch_f1 /= len(train_loader)

        # validation phase
        model.eval()
        with torch.no_grad():
            for j, val_data in tqdm(enumerate(val_loader), desc=f"Epoch {ep+1}/{epochs} - Validation"):
                val_imgs, val_labels = val_data[0].to(device), val_data[1].to(device)
                
                # forward pass
                val_outputs = model(val_imgs)
                val_preds = np.argmax(val_outputs.detach().cpu(), 0)

                # get loss and metrics
                val_batch_loss = criterion(val_outputs, val_labels)
                val_loss += val_batch_loss.item()
                val_acc += accuracy_score(val_labels.detach().cpu(), val_preds) 
                val_f1 += f1_score(val_labels.detach().cpu(), val_preds)      

            val_loss /= len(val_loader)
            val_acc /= len(train_loader)
            val_f1 /= len(train_loader)

        print(f"Epoch {ep+1}/{epochs}: Train Loss: {epoch_loss:.4f} --- Val Loss: {val_loss:.4f}")
        print(f"Metrics: Train Acc: {epoch_acc:.4f} --- Train F1: {epoch_f1:.4f} --- Val Acc: {val_acc:.4f} --- Val F1: {val_f1:.4f}\n")

    # if validation loss decreases from best one, save the model weights
    if val_loss + 1e-3 < best_val_loss:
        best_val_loss = val_loss
        
        os.makedirs("./best_models", exists=True)
        
        torch.save(model.state_dict(), "./best_models/best_megadetector.pt")
        print("Val loss decreased: new best model saved at ./best_models/best_megadetector.pt\n")

    return model

# testing loop
def test(model, test_loader, criterion, device="cuda"):

    model.to(device)

    # validation phase
    model.eval()
    with torch.no_grad():
        for i, test_data in tqdm(enumerate(train_loader), desc=f"Testing..."):
            test_imgs, test_labels = test_data[0].to(device), test_data[1].to(device)
                
            # forward pass
            test_outputs = model(test_imgs)
            test_preds = np.argmax(test_outputs.detach().cpu(), 0)

            # get loss and metrics
            test_batch_loss = criterion(test_outputs, test_labels)
            test_loss += test_batch_loss.item()
            test_acc += accuracy_score(test_labels.detach().cpu(), test_preds) 
            test_f1 += f1_score(test_labels.detach().cpu(), test_preds)      

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_f1 /= len(test_loader)

        print(f"Test Loss: {test_loss:.4f} || Metrics: Test Acc: {test_acc:.4f} --- Test F1: {test_f1:.4f}\n")