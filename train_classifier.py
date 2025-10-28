import torch, os, argparse
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from vqvae import FlatVQVAE
import distributed as dist

# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Initialize CUDA
torch.cuda.init()

class ReconstructedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def decode_quantizes(model, quantizes):
    quantizes = torch.tensor(quantizes).float().to(device)
    with torch.no_grad():
        images = model.decode(quantizes)
    return images

ckpt_vqvae = "/home/abghamtm/work/masking_comparison/checkpoint/vqvae/model_epoch80_flat_vqvae80x80_144x456codebook.pth"
torch.cuda.set_device(1) 
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

quantizes = np.load('/home/abghamtm/work/masking_comparison/checkpoint/vqvae/quantized_epoch80_flat_vqvae80x80_144x456codebook.npy')
labels = np.load('/home/abghamtm/work/masking_comparison/checkpoint/vqvae/labels_epoch80_flat_vqvae80x80_144x456codebook.npy')
labels = torch.from_numpy(labels)

model_vqvae = FlatVQVAE().to(device)
model_vqvae.load_state_dict(torch.load(ckpt_vqvae, map_location=device))
model_vqvae = model_vqvae.to(device)
model_vqvae.eval()
reconstructed_images = decode_quantizes(model_vqvae, quantizes)

dataset = ReconstructedDataset(reconstructed_images, labels)

# Split the indices in a stratified manner
train_indices, test_indices = train_test_split(
    np.arange(len(labels)),
    test_size=0.2,
    stratify=labels,
    random_state=42
)
# Create subsets of the dataset
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

batchsize_modified=1
train_loader = DataLoader(train_dataset, batch_size=batchsize_modified, shuffle=True, num_workers=0)#256
for inputs, labels in train_loader:
    print(inputs.shape, labels.shape)
    break

test_loader = DataLoader(test_dataset, batch_size=batchsize_modified, shuffle=True, num_workers=0)

weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 30  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader):
        inputs = preprocess(inputs)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    torch.save(model.state_dict(), f"/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/weights_epoch{str(epoch + 1).zfill(2)}.pth")

# Define classifier and load saved model(weights)
classifier = resnet50(pretrained=False)
models_list = os.listdir("/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/")
models_list.sort()
for model_name in models_list:
    print(model_name)
    classifier.load_state_dict(torch.load(os.path.join("/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/",model_name)))
    classifier.to(device)
    classifier.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = preprocess(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'loss: {total_loss:.2f}%')

