import os
from collections import Counter
from textwrap import indent
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import wandb
from src.models.custom_classifier import CustomClassifier
from utils.data_preprocessing import ClassificationDataset

# ------------------------------DATASETS------------------------------

# training data augmentations
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAutocontrast(p=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# test transforms (only normalization, same as training)
test_transform = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

# use separate transforms for train and test
train_set = ClassificationDataset(
    "eu_mammals_classification",
    transform=train_transform,
    split="train",
    fixed_size=(224, 224),
)
# here we use "val" split as the test images were under the val directory
test_set = ClassificationDataset(
    "eu_mammals_classification",
    transform=test_transform,
    split="val",
    fixed_size=(224, 224),
)

# split the training set into train and validation sets (85/15 split)
train_indices, val_indices = train_test_split(
    np.arange(len(train_set)),
    test_size=0.15,
    stratify=train_set.targets if hasattr(train_set, "targets") else None,
    random_state=33,
)


val_set = Subset(train_set, val_indices)
train_set = Subset(train_set, train_indices)

print(
    f"Dataset sizes: Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}"
)

# --------------------------------------------------------------------


# custom multicalss focal loss function
class MulticlassFocalLoss(nn.Module):
    r"""Compute focal loss for multiclass classification.

    The focal loss is defined as
    .. math::
        FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)

    where:
        - :math:`\alpha_t` is a weighting parameter for imbalance in classes
        - :math:`gamma` is a scaling parameter to focus more on hard-to-classify samples
    """

    def __init__(
        self,
        alpha: torch.tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha (torch.tensor, optional): Alpha weight vector, to help with imbalanced datasets. Defaults to None.
            gamma (float, optional): Gamma focusing parameter, to weigh more difficult samples. Defaults to 2.
            reduction (str, optional): Type of reduction to apply to the loss. Defaults to "mean".
        """
        super(MulticlassFocalLoss, self).__init__()
        self.alpha = alpha  # vector of weights to control class imbalancedness
        self.gamma = (
            gamma  # controls amount of focus given to hard-to-classify examples
        )
        self.reduction = reduction

    def forward(self, inputs: torch.tensor, targets: torch.tensor):
        """
        Args:
            inputs (torch.tensor): Raw logits tensor from model output [batch_size, num_classes]
            targets (torch.tensor): Ground truth class indices tensor [batch_size]

        Returns:
            torch.tensor: Computed focal loss value
        """

        # get softmax probabilities for each class for each sample from raw logits
        p = F.softmax(inputs, dim=1)
        # gather the model's probabilities for the ground truth classes for each sample:
        # - targets.unsqueeze(1) adds one dimension (for torch.gather, input and index must have same number of dims)
        # - p.gather(a, t) extracts from the tensor p the elements across the axis a, secified by the indices in tensor t
        # - we squeeze(1) the result across the rows to get back to a single dimension vector
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (batch_size, )

        # compute standard cross-entropy loss (-log(p_t)) for each sample
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # apply alpha weighting if provided, to handle class imbalance
        if self.alpha is not None:
            # vector of corresponding gt classes alpha weights to be used
            # obtained from indexing of the original tensor with the target class indices
            alpha_t = self.alpha[targets]
        else:
            # no weighting: all classes are weighted equally
            alpha_t = torch.Tensor([1.0]).repeat(targets.size(dim=0))

        # focal loss calculation
        # (ce_loss already contains the log(p_t) term from cross entropy)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        # no reduction
        else:
            return focal_loss


# training loop
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    device: str = "cuda",
    wandb_run: wandb.Run = None,
    patience: int = 50,
) -> Tuple[nn.Module, dict]:
    """Training loop for classifier model.

    Args:
        model (nn.Module): Neural Network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader with the training images and labels.
        val_loader (torch.utils.data.DataLoader): DataLoader with the validation images and labels.
        criterion (nn.Module): Loss function used to optimize the network.
        optimizer (torch.optim.Optimizer): the optimizer for gradient descent.
        epochs (int, optional): Number of epochs to run the training loop over whole dataset.
                                Defaults to 100.
        device (str, optional): Computing device to use for the model training (e.g., CPU or GPU).
                                Defaults to "cuda".
        wandb_run (wandb.Run, optional): Wandb run to track the training on wandb.
        patience (int, optional): Number of epochs to wait without improvement on validation loss, before early stopping.
                                Defaults to 50.
    Returns:
        Tuple[nn.Module, dict]: Tuple containing trained model and best model state dict.
    """

    if wandb_run:
        wandb_run.watch(model, criterion, log="all", log_freq=5)

    model.to(device)
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for ep in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_f1 = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0

        model.train()

        for i, data in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {ep + 1}/{epochs} - Training",
        ):
            imgs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy_score(
                labels.detach().cpu().numpy(), preds.detach().cpu().numpy()
            )
            epoch_f1 += f1_score(
                labels.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                average="weighted",
            )

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)
        epoch_f1 /= len(train_loader)

        # validation phase
        model.eval()
        with torch.no_grad():
            for j, val_data in tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Epoch {ep + 1}/{epochs} - Validation",
            ):
                val_imgs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_imgs)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_batch_loss = criterion(val_outputs, val_labels)
                val_loss += val_batch_loss.item()
                val_acc += accuracy_score(
                    val_labels.detach().cpu().numpy(), val_preds.detach().cpu().numpy()
                )
                val_f1 += f1_score(
                    val_labels.detach().cpu().numpy(),
                    val_preds.detach().cpu().numpy(),
                    average="weighted",
                )

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_f1 /= len(val_loader)

        # wandb logging
        # ---------------------------------------
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": ep + 1,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "train_acc": epoch_acc,
                    "train_f1": epoch_f1,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                },
                step=ep,
            )
        # ---------------------------------------

        print(
            f"Epoch {ep + 1}/{epochs}: Train Loss: {epoch_loss:.4f} --- Val Loss: {val_loss:.4f}"
        )
        print(
            f"Metrics: Train Acc: {epoch_acc:.4f} --- Train F1: {epoch_f1:.4f} --- Val Acc: {val_acc:.4f} --- Val F1: {val_f1:.4f}\n"
        )

        # model save and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()

            os.makedirs("./weights", exist_ok=True)
            torch.save(model.state_dict(), "./weights/best_classifier.pt")
            print(
                "Val loss decreased: new best model saved at ./weights/best_classifier.pt\n"
            )
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {ep + 1} epochs")
                # load best model
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break

    return model, best_model_state


# testing loop
def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device="cuda",
    wandb_run=None,
) -> tuple:
    """Training loop for classifier model.

    Args:
        model (nn.Module): Ttrained neural Network model to be tested.
        test_loader (torch.utils.data.DataLoader): DataLoader with the testing images and labels.
        criterion (nn.Module): Loss function to be computed on the test set.
        device (str, optional): Computing device to use for the model training (e.g., CPU or GPU).
                                Defaults to "cuda".
        wandb_run (wandb.Run, optional): Wandb run to track the training on wandb.

    Returns:
        tuple: Loss, accuracy and F1 score on the test set.
    """
    model.to(device)

    test_loss = 0.0
    test_acc = 0.0
    test_f1 = 0.0

    # for logging individual results on wandb
    # ---------------------------------------------------------------------------------
    predictions_table = []

    # get class names mapping from the dataset
    class_id_to_name = {}
    if hasattr(test_loader.dataset, "dataset"):  # handle Subset wrapper
        dataset = test_loader.dataset.dataset
    else:
        dataset = test_loader.dataset

    if hasattr(dataset, "labels"):
        # create mapping from class_id to class_label
        labels_df = dataset.labels
        class_id_to_name = dict(zip(labels_df["class_id"], labels_df["class_label"]))

    class_names = class_id_to_name if class_id_to_name else None
    # ---------------------------------------------------------------------------------

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, test_data in tqdm(
            enumerate(test_loader), total=len(test_loader), desc="Testing..."
        ):
            test_imgs, test_labels = test_data[0].to(device), test_data[1].to(device)

            # forward pass
            test_outputs = model(test_imgs)
            test_preds = torch.argmax(test_outputs, dim=1)
            test_probs = F.softmax(test_outputs, dim=1).detach().cpu().numpy()

            # collect predictions and labels for confusion matrix
            all_preds.extend(test_preds.detach().cpu().numpy())
            all_labels.extend(test_labels.detach().cpu().numpy())

            # get loss and metrics
            test_batch_loss = criterion(test_outputs, test_labels)
            test_loss += test_batch_loss.item()
            test_acc += accuracy_score(
                test_labels.detach().cpu().numpy(), test_preds.detach().cpu().numpy()
            )
            test_f1 += f1_score(
                test_labels.detach().cpu().numpy(),
                test_preds.detach().cpu().numpy(),
                average="weighted",
            )

            # log individual predictions for wandb table
            # --------------------------------------------------------------------

            if wandb_run:
                for j in range(test_imgs.shape[0]):
                    # convert image tensor to wandb Image format
                    img_array = test_imgs[j].detach().cpu().numpy()
                    # denormalize image from ImageNet normalization for display
                    # reverse normalization: (normalized * std) + mean
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_array = img_array * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1)
                    img_array = np.clip(img_array, 0, 1)  # ensure values are in [0,1]
                    img_array = np.transpose(
                        img_array, (1, 2, 0)
                    )  # (C, W, H) to (H, W, C) shape

                    true_label = test_labels[j].item()
                    pred_label = test_preds[j].item()
                    confidence = test_probs[j].max()
                    correct = true_label == pred_label

                    # get class names if available
                    true_class = (
                        class_names[true_label]
                        if class_names and true_label in class_names
                        else str(true_label)
                    )
                    pred_class = (
                        class_names[pred_label]
                        if class_names and pred_label in class_names
                        else str(pred_label)
                    )

                    predictions_table.append(
                        [
                            wandb.Image(img_array),
                            true_class,
                            pred_class,
                            confidence,
                            correct,
                        ]
                    )
        # --------------------------------------------------------------------

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_f1 /= len(test_loader)

        # generate confusion matrix for wandb
        # --------------------------------------------------------------------
        if wandb_run:
            # compute confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            # create class labels for the confusion matrix
            if class_names:
                class_labels = [class_names.get(i, str(i)) for i in range(len(cm))]
            else:
                class_labels = [str(i) for i in range(len(cm))]

            # create a matplotlib figure for the confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
            )
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()

            # log confusion matrix to wandb
            wandb_run.log({"confusion_matrix": wandb.Image(plt)})
            plt.close()  # close the figure to free memory
        # --------------------------------------------------------------------

        # create and log the predictions table
        # --------------------------------------------------------------------

        if wandb_run:
            columns = [
                "Image",
                "True Label",
                "Predicted Label",
                "Confidence",
                "Correct",
            ]
            test_table = wandb.Table(columns=columns, data=predictions_table)

            # log on wandb
            wandb_run.log(
                {
                    "loss": test_loss,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "test_predictions": test_table,
                }
            )
        # --------------------------------------------------------------------
        print("\n")
        print("=" * 15, "Results on test set:", "=" * 15)
        print("\n")

        print(
            indent(
                f"Test Loss: {test_loss:4f}\nTest Accuracy: {test_acc:4f}\nTest F1-Score: {test_f1:4f}",
                ">",
            )
        )

    return test_loss, test_acc, test_f1


def main(config: dict, model: str = "custom") -> None:
    """Main function to run model training and testing, with logging on wandb.

    Args:
        config (dict): dictionary with training configurations (used also for wandb logging).
        model (str, optional): Type of neural network to train. Must be in ["custom", "pretrained"].

    Raises:
        ValueError: If the chosen model is not among the available ones.

    Examples:
        >>> configs_custom = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 16,
        "focal_loss_gamma": 2.0,
        "init_con_features": 32,
        "dropout_rate": 0.2,
        "class_loss_weighting": True}
        >>> main(configs_custom, "custom")
        >>> configs_resnet = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 16,
        "focal_loss_gamma": 2.0,
        "freeze": True
        "class_loss_weighting": True}
        >>> main(configs_resnet, "pretrained")
    """

    if model not in ["custom", "pretrained"]:
        raise ValueError(
            "Invalid model type. Please choose one of ['custom', 'pretrained']."
        )

    # login on wandb
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    with wandb.init(project="wildlife-detection", config=config) as run:
        # get original dataset labels (have to use .dataset since it is a Subset now)
        num_classes = len(train_set.dataset.labels["class_id"].unique())

        if model == "custom":
            model = CustomClassifier(
                init_features=run.config["init_con_features"],
                dropout_rate=run.config["dropout_rate"],
            )
        elif model == "pretrained":
            # load resnet34
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet34", pretrained=True
            )
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            if run.config["freeze"]:
                # freeze initial 2 of the 4 main resnet34 "layer" blocks
                for param in model.layer1.parameters():
                    param.requires_grad = False
                for param in model.layer2.parameters():
                    param.requires_grad = False

        # calculate class weights for focal loss (inverse frequency)
        train_labels = [
            train_set.dataset.labels.iloc[i]["class_id"] for i in train_indices
        ]
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)

        if run.config["class_loss_weighting"]:
            # create alpha weights (inverse frequency, normalized)
            alpha_weights = torch.zeros(num_classes)
            for class_id in range(num_classes):
                if class_id in class_counts:
                    alpha_weights[class_id] = total_samples / (
                        num_classes * class_counts[class_id]
                    )
        else:
            # no weighting: all classes weigh the same
            alpha_weights = torch.ones(num_classes)

        print(f"Alpha weights for focal loss: {alpha_weights}")

        # create dataloaders
        train_loader = DataLoader(
            train_set, batch_size=run.config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=run.config["batch_size"], shuffle=False
        )
        test_loader = DataLoader(test_set, batch_size=5, shuffle=False)

        # use focal loss instead of crossentropy for better handling of class imbalance
        criterion = MulticlassFocalLoss(
            alpha=alpha_weights.cuda() if torch.cuda.is_available() else alpha_weights,
            gamma=run.config["focal_loss_gamma"],
            reduction="mean",
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=run.config["learning_rate"]
        )

        # train model
        model, best_model_state = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            run.config["epochs"],
            wandb_run=run,
        )

        # load model with best performance on validation, for testing
        model.load_state_dict(best_model_state)

        # test model
        test(model, test_loader, criterion, wandb_run=run)


if __name__ == "__main__":
    configs = {
        "learning_rate": 0.001,
        "epochs": 300,
        "batch_size": 16,
        "focal_loss_gamma": 2,
        "init_con_features": 32,
        "dropout_rate": 0.3,
        "class_loss_weighting": True,
    }

    main(configs, "custom")
