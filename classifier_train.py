import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_preprocessing import ClassificationDataset, crop_and_save_detections
from species_classifier import SpeciesClassifier
import glob
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter





def create_stratified_split(dataset, train_ratio=0.9, random_seed=42):
    """Create a stratified train/validation split ensuring each class has samples in both splits.
    
    Args:
        dataset: ClassificationDataset instance
        train_ratio: proportion of data for training
        random_seed: random seed for reproducibility
        
    Returns:
        tuple: (train_indices, val_indices)
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Get labels and create a mapping from class to indices
    labels = dataset.labels['class_id'].tolist()
    class_to_indices = {}
    
    for idx, label in enumerate(labels):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for class_id, indices in class_to_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        # Calculate split point, ensuring at least 1 sample in training
        n_samples = len(indices)
        n_train = max(1, int(n_samples * train_ratio))
        
        # If we have only 1 sample, put it in training
        if n_samples == 1:
            train_indices.extend(indices.tolist())
        else:
            train_indices.extend(indices[:n_train].tolist())
            val_indices.extend(indices[n_train:].tolist())
    
    print(f"Stratified split - Classes: {len(class_to_indices)}")
    for class_id, indices in class_to_indices.items():
        n_train = len([i for i in train_indices if labels[i] == class_id])
        n_val = len([i for i in val_indices if labels[i] == class_id])
        print(f"  Class {class_id}: {n_train} train, {n_val} val (total: {len(indices)})")
    
    return train_indices, val_indices


def create_weighted_sampler(dataset, indices=None):
    """Create a WeightedRandomSampler that gives higher probability to low-frequency classes.
    
    Args:
        dataset: ClassificationDataset instance
        indices: list of indices to consider (for subset sampling)
        
    Returns:
        WeightedRandomSampler: Sampler that balances class frequencies
    """
    # Get labels, either all or subset based on indices
    if indices is None:
        labels = dataset.labels['class_id'].tolist()
    else:
        all_labels = dataset.labels['class_id'].tolist()
        labels = [all_labels[i] for i in indices]
    
    # Count frequency of each class
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate weights: inverse frequency
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # Create sample weights: each sample gets the weight of its class
    sample_weights = [class_weights[label] for label in labels]
    
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights: {dict(class_weights)}")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


#------------------------------DATASETS------------------------------


# Training transforms with augmentations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomAutocontrast(p=0.3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test transforms without augmentations
test_transform = transforms.Compose([
    # Use same normalization as training
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# use separate transforms for train and test
training_set = ClassificationDataset("class_dataset", transform=train_transform, split="train", fixed_size=(224, 224))
test_set = ClassificationDataset("class_dataset", transform=test_transform, split="test", fixed_size=(224, 224))

# create stratified train/validation split from training set
train_indices, val_indices = create_stratified_split(training_set, train_ratio=0.8)

# create subset datasets using the stratified indices
from torch.utils.data import Subset
train_dataset = Subset(training_set, train_indices)
val_dataset = Subset(training_set, val_indices)

print(f"Dataset sizes: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_set)}")


#--------------------------------------------------------------------



class MulticlassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (torch.tensor, optional): alpha weight vector, to help with imbalanced datasets. Defaults to None.
            gamma (float, optional): gamma focusing parameter, to weigh more difficult samples. Defaults to 2.
            reduction (str, optional): type of reduction to apply to the loss. Defaults to "mean".
        """
        super(MulticlassFocalLoss, self).__init__()
        self.alpha = alpha # vector of weights to control class imbalancedness
        self.gamma = gamma # controls amount of focus given to hard-to-classify examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss for multiclass classification.
        
        Args:
            inputs (torch.tensor): raw logits tensor from model output [batch_size, num_classes]
            targets (torch.tensor): ground truth class indices tensor [batch_size]
        
        Returns:
            torch.tensor: Computed focal loss value
        """
        
        # get softmax probabilities for each class for each sample from raw logits 
        p = F.softmax(inputs, dim=1)
        # gather the model's probabilities for the ground truth classes for each sample:
        # - targets.unsqueeze(1) adds one dimension (since for gather, input and index must have same dims)
        # - p.gather(a, t) extracts from the tensor p the elements across the axis a, secified by the indices
        #   in tensor t
        # - we squeeze(1) the result across the rows to get back to a single dimension vector
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # compute standard cross-entropy loss (-log(p_t)) for each sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # apply alpha weighting if provided, to handle class imbalance
        if self.alpha is not None:
            # vector of corresponding gt classes alpha weights to be used
            # obtained from indexing of the original tensor with the target class indices
            alpha_t = self.alpha[targets]
        else:
            # no weighting: all classes are weighted equally
            alpha_t = torch.Tensor([1.0]).repeat(targets.size(dim=0))
            
        # focal loss formula: -alpha_t * (1 - p_t)^gamma * log(p_t)
        # (ce_loss already contains the log(p_t) term from cross entropy)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        # no reduction
        else:
            return focal_loss



# training loop
def train(model, train_loader, val_loader, criterion, optimizer, epochs=100, device="cuda", wandb_run = None, patience=20):
    """Training loop for classifier model

    Args:
        model (torch.nn.Module): the model to be trained
        train_loader (torch.utils.data.DataLoader): DataLoader with the training images and labels
        val_loader (torch.utils.data.DataLoader): DataLoader with the validation images and labels
        criterion (torch.nn.Loss): the loss function
        optimizer (torch.optim.Optimizer): the optimizer for gradient descent
        epochs (int, optional): number of epochs to run the training loop over whole dataset. Defaults to 100.
        wandb_run (wandb.run, optional): wandb run to track the training
        patience (int, optional): number of epochs to wait without improvement on validation, before early stopping.

    Returns:
        torch.nn.Module: trained model.
    """

    if wandb_run:
        wandb_run.watch(model, criterion, log="all", log_freq=5)

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for ep in range(epochs):
        # ...existing training code...
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_f1 = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        
        model.train()

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {ep+1}/{epochs} - Training"):
            imgs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy()) 
            epoch_f1 += f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')        

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)
        epoch_f1 /= len(train_loader)

        # validation phase
        model.eval()
        with torch.no_grad():
            for j, val_data in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {ep+1}/{epochs} - Validation"):
                val_imgs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_imgs)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_batch_loss = criterion(val_outputs, val_labels)
                val_loss += val_batch_loss.item()
                val_acc += accuracy_score(val_labels.detach().cpu().numpy(), val_preds.detach().cpu().numpy()) 
                val_f1 += f1_score(val_labels.detach().cpu().numpy(), val_preds.detach().cpu().numpy(), average='weighted')      

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_f1 /= len(val_loader)

        if wandb_run:
            wandb_run.log({"epoch": ep+1, "train_loss": epoch_loss, "val_loss": val_loss,
                        "train_acc": epoch_acc, "train_f1": epoch_f1,
                        "val_acc": val_acc, "val_f1": val_f1}, step=ep)

        print(f"Epoch {ep+1}/{epochs}: Train Loss: {epoch_loss:.4f} --- Val Loss: {val_loss:.4f}")
        print(f"Metrics: Train Acc: {epoch_acc:.4f} --- Train F1: {epoch_f1:.4f} --- Val Acc: {val_acc:.4f} --- Val F1: {val_f1:.4f}\n")

        # model save and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            os.makedirs("./weights", exist_ok=True)
            torch.save(model.state_dict(), "./weights/best_classifier.pt")
            print("Val loss decreased: new best model saved at ./weights/best_classifier.pt\n")
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {ep+1} epochs")
                # load best model
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break

    return model

# testing loop
def test(model, test_loader, criterion, device="cuda", wandb_run=None):
    model.to(device)
    
    test_loss = 0.0
    test_acc = 0.0
    test_f1 = 0.0
    
    # WANDB LOGGING PART
    # For logging individual results
    predictions_table = []
    
    # Get class names mapping from the dataset
    class_id_to_name = {}
    if hasattr(test_loader.dataset, 'dataset'):  # Handle Subset wrapper
        dataset = test_loader.dataset.dataset
    else:
        dataset = test_loader.dataset
        
    if hasattr(dataset, 'labels'):
        # Create mapping from class_id to class_label
        labels_df = dataset.labels
        class_id_to_name = dict(zip(labels_df['class_id'], labels_df['class_label']))
    
    class_names = class_id_to_name if class_id_to_name else None

    # validation phase
    model.eval()
    with torch.no_grad():
        for i, test_data in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing..."):
            test_imgs, test_labels = test_data[0].to(device), test_data[1].to(device)
                
            # forward pass
            test_outputs = model(test_imgs)
            test_preds = torch.argmax(test_outputs, dim=1)
            test_probs = F.softmax(test_outputs, dim=1).detach().cpu().numpy()

            # get loss and metrics
            test_batch_loss = criterion(test_outputs, test_labels)
            test_loss += test_batch_loss.item()
            test_acc += accuracy_score(test_labels.detach().cpu().numpy(), test_preds.detach().cpu().numpy()) 
            test_f1 += f1_score(test_labels.detach().cpu().numpy(), test_preds.detach().cpu().numpy(), average='weighted')
            
            # Log individual predictions for wandb table
            if wandb_run:
                for j in range(test_imgs.shape[0]):
                    # Convert image tensor to wandb Image format
                    img_array = test_imgs[j].detach().cpu().numpy()
                    # Denormalize image from ImageNet normalization for display
                    # Reverse ImageNet normalization: (normalized * std) + mean
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_array = img_array * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1)
                    img_array = np.clip(img_array, 0, 1)  # Ensure values are in [0,1]
                    img_array = np.transpose(img_array, (1, 2, 0))  # CHW to HWC
                    
                    true_label = test_labels[j].item()
                    pred_label = test_preds[j].item()
                    confidence = test_probs[j].max()
                    correct = true_label == pred_label
                    
                    # Get class names if available
                    true_class = class_names[true_label] if class_names and true_label in class_names else str(true_label)
                    pred_class = class_names[pred_label] if class_names and pred_label in class_names else str(pred_label)
                    
                    predictions_table.append([
                        wandb.Image(img_array),
                        true_class,
                        pred_class,
                        confidence,
                        correct
                    ])

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_f1 /= len(test_loader)

        if wandb_run:
            # Create and log the predictions table
            columns = ["Image", "True Label", "Predicted Label", "Confidence", "Correct"]
            test_table = wandb.Table(columns=columns, data=predictions_table)
            
            # log on wandb
            wandb_run.log({
                "loss": test_loss,  
                "test_acc": test_acc, 
                "test_f1": test_f1,
                "test_predictions": test_table
            })

        print(f"Test Loss: {test_loss:.4f} || Metrics: Test Acc: {test_acc:.4f} --- Test F1: {test_f1:.4f}\n")



# login on wandb
wandb.login(key=os.environ.get("WANDB_API_KEY"))

config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 16}

with wandb.init(project="wildlife-detection", config=config) as run:
    # get hyperparameters from config
    
    #model = SpeciesClassifier()
    # get pretrained resnet
    model = torch.hub.load('pytorch/vision:v0.10.0', "resnet34", pretrained=True)
    num_classes = len(training_set.labels['class_id'].unique())  # Get actual number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    # calculate class weights for focal loss (inverse frequency)
    train_labels = [training_set.labels.iloc[i]['class_id'] for i in train_indices]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # create alpha weights (inverse frequency, normalized)
    alpha_weights = torch.zeros(num_classes)
    for class_id in range(num_classes):
        if class_id in class_counts:
            alpha_weights[class_id] = total_samples / (num_classes * class_counts[class_id])
    
    print(f"Alpha weights for focal loss: {alpha_weights}")
    


    # Optionally create weighted sampler for balanced training
    # Uncomment the line below if you want to use weighted sampling
    # train_sampler = create_weighted_sampler(training_set, train_indices)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=run.config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=run.config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False)

    # use focal loss instead of crossentropy for better handling of class imbalance
    criterion = MulticlassFocalLoss(alpha=alpha_weights.cuda() if torch.cuda.is_available() else alpha_weights, 
                                   gamma=2.0, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=run.config["learning_rate"])

    # train model
    model = train(model, train_loader, val_loader, criterion, optimizer, run.config["epochs"], wandb_run = run)

    # test model
    test(model, test_loader, criterion, wandb_run=run)