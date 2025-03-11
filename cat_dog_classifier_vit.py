import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.amp import GradScaler, autocast
from torchvision.models import ViT_B_16_Weights

from PIL import Image
from tqdm import tqdm

# Create a GradScaler for mixed precision training to optimize memory usage and speed up training
scaler = GradScaler("cuda")

# Define data transformations for both training and validation datasets
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and standard deviation
])

val_transforms = transforms.Compose([
    transforms.Resize(224),  # Resize image to 224x224
    transforms.CenterCrop(224),  # Crop center of the image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and standard deviation
])

# Set train and validation images path
train_dir = './images/train'
validation_dir = './images/validate'

# Locate and load train and validation images from folders, and apply transformations
train_dataset = datasets.ImageFolder(root = train_dir, transform = train_transforms)
val_dataset = datasets.ImageFolder(root = validation_dir, transform = val_transforms)

batch_size = 512
num_workers = 6

# Create datasets via DataLoader with optimizations for performance:
# `batch_size`: Defines the number of samples per batch.
# `shuffle`: Randomizes the training data order for better generalization.
# `num_workers`: Number of subprocesses used to load the data in parallel.
# `prefetch_factor`: Number of batches loaded in advance to avoid waiting on I/O.
# `pin_memory`: Enables faster transfer of data to GPU by keeping it in page-locked memory.
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, prefetch_factor = 2, pin_memory = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, prefetch_factor = 2, pin_memory = True)

# Define Visual Transformer Model
class ViTModel(nn.Module):

    def __init__(self, num_classes=1):

        super(ViTModel, self).__init__()

        # Load a pre-trained Vision Transformer model (ViT-B/16) with ImageNet-1k weights from torchvision
        self.vit = models.vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_V1)

        
        # Access the head (classification layer) correctly
        # The original head is in a Sequential container with a Linear layer at the end
        # Access the first Linear layer
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, num_classes)

        # Freeze all pre-trained layers (except the head)
        for param in self.vit.parameters():
            
            param.requires_grad = False 

        # Unfreeze only the final classification layer (head)
        for param in self.vit.heads.parameters():
            
            param.requires_grad = True  
    
    
    def forward(self, x):

        return self.vit(x)


def train_one_epoch(model, train_loader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Use tqdm for the training loop to show progress
    with tqdm(train_loader, unit = "batch") as tepoch:

        for images, labels in tepoch:

            # Move data to the same device as the model (GPU or CPU)
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # Enable automatic mixed precision (AMP) for faster computations and reduced memory usage on CUDA-enabled devices
            with autocast("cuda"):

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())

            # Use GradScaler to scale loss for better numerical stability during backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track loss and accuracy
            running_loss += loss.item()
            predicted = torch.sigmoid(outputs).squeeze().round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar with current loss and accuracy
            tepoch.set_postfix(loss = running_loss / (tepoch.n + 1), accuracy = correct / total)
    

    return (running_loss / len(train_loader)), (correct / total)


def validate(model, val_loader, criterion, device):

    model.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation during validation to save memory and computation
    with torch.no_grad():  

        for images, labels in val_loader:
            
            # Move data to the same device as the model (GPU or CPU)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())

            val_loss += loss.item()
            predicted = outputs.squeeze().round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    

    return (val_loss / len(val_loader)), (correct / total)


def main():

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize model, loss function, and optimizer
    model = ViTModel(num_classes = 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    num_epochs = 1
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validate the model
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch summary (losses and accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'cat_dog_classifier_vit.pth')

    # Inference on a test image
    img_path = './images/cat_test.jpg'
    img = Image.open(img_path).convert('RGB')

    # Add batch dimension
    img = val_transforms(img).unsqueeze(0)

    # Move image to GPU or CPU
    img = img.to(device)

    # Switch model to evaluation mode
    model.eval()

    # Classify test image as belonging to Cat or Dog class
    with torch.no_grad():

        output = model(img)
        prediction = output.squeeze().item()
        
        if prediction < 0.5:

            print("It's a Cat!")

        else:

            print("It's a Dog!")


if __name__ == '__main__':

    main()
