import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from PIL import Image
from tqdm import tqdm

# Create a GradScaler for mixed precision training to optimize memory usage and speed up training
scaler = GradScaler("cuda")


def main():

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:" + str(device))

    # Define data transformations for both training and validation datasets
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomResizedCrop(150),  # Randomly crop and resize to 150x150
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize with mean and standard deviation
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(150),   # Resize image to 150x150
        transforms.CenterCrop(150),  # Crop center of the image to 150x150
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize with mean and standard deviation
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


    # Define CNN Model
    class CNNModel(nn.Module):

        def __init__(self):

            super(CNNModel, self).__init__()

            self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1) # 3 input channels (RGB), 32 output channels
            self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
            
            # Fully connected layers after flattening the output from convolutional layers
            self.fc1 = nn.Linear(128 * 18 * 18, 512)  # Flattened size (based on input image size)
            self.fc2 = nn.Linear(512, 1)  # Output a single value for binary classification
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.sigmoid = nn.Sigmoid() # Sigmoid to get output between 0 and 1 (for binary classification)
        

        def forward(self, x):

            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.sigmoid(x)
            
            return x

    # Initialize model, loss function, and optimizer
    model = CNNModel().to(device)  
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    num_epochs = 1

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Train the Model with GPU and Progress Bar
    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for the training loop to show progress
        with tqdm(train_loader, unit="batch") as tepoch:

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
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1), accuracy=correct / total)
        

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Switch model to evaluation mode
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
        

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), 'cat_dog_classifier.pth')

    # Inference on a test image
    img_path = './images/cat_test.jpg'
    img = Image.open(img_path).convert('RGB')

    # Add batch dimension
    img = val_transforms(img).unsqueeze(0) 

    # Move image to GPU or CPU
    img = img.to(device)  

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
