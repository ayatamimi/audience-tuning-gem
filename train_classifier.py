import torch, os, argparse
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
#from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from vqvae import FlatVQVAE
import distributed as dist
import math
import neptune.new as neptune


run = neptune.init_run(
    project="UTKFaces",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
    capture_stdout=False,
    capture_stderr=False,
    #with_id="distil",
    source_files=["train_classifier.py"]
)



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
    
#pure CPU fallback (slow but safe)
# =============================================================================
# def decode_quantizes(model, quantizes):
#     quantizes = torch.tensor(quantizes).float().to(device)
#     with torch.no_grad():
#         images = model.decode(quantizes)
#     return images
# 
# =============================================================================

import torch

def decode_quantizes(model, quantizes, device="cuda:0", batch_size=16, use_autocast=True):
    """
    Decode large `quantizes` without CUDA OOM by streaming mini-batches to the GPU.

    Args:
        model: VQ-VAE (or similar) with .decode() method
        quantizes: numpy array or torch.Tensor, shape [N, ...]
        device: "cuda:0", "cuda:1", or "cpu"
        batch_size: per-GPU batch size to fit in memory
        use_autocast: use mixed precision on CUDA to cut memory

    Returns:
        images: torch.Tensor on CPU, concatenated over all batches
    """
    model.eval()
    # Keep source on CPU; only slice-batches go to GPU
    q_cpu = torch.as_tensor(quantizes, device="cpu", dtype=torch.float32)
    # Pin for faster H2D copies (no effect if already CUDA/CPU not pinned)
    if q_cpu.device.type == "cpu":
        try:
            q_cpu = q_cpu.pin_memory()
        except RuntimeError:
            pass  # pinning may fail on some platforms

    outs = []
    N = q_cpu.shape[0]

    # Choose autocast only for CUDA
    use_amp = use_autocast and ("cuda" in str(device) and torch.cuda.is_available())

    with torch.no_grad():
        for i in range(0, N, batch_size):
            q = q_cpu[i:i+batch_size]
            q = q.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.decode(q)
            else:
                out = model.decode(q)

            # move each batch result back to CPU to free VRAM
            outs.append(out.float().cpu())

            # clean up per-batch GPU tensors
            del q, out
            if "cuda" in str(device):
                torch.cuda.empty_cache()

    return torch.cat(outs, dim=0)



ckpt_vqvae = "./checkpoint_correct/vqvae/model_epoch100_flat_vqvae80x80_64x400codebook.pth"
torch.cuda.set_device(1) 
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"


#### validation set###

val_labels= np.load('./checkpoint_correct/vqvae/val_labels.npy')
val_labels = torch.from_numpy(val_labels)

val_quantizes = np.load('./checkpoint_correct/vqvae/val_codebook_vqvae_80x80_codebook_64x456.npy')


#### train set###

train_labels = np.load('./checkpoint_correct/vqvae/train_labels.npy')
train_labels = torch.from_numpy(train_labels)

train_quantizes = np.load('./checkpoint_correct/vqvae/train_codebook_vqvae_80x80_codebook_64x456.npy')


model_vqvae = FlatVQVAE().to(device)
model_vqvae.load_state_dict(torch.load(ckpt_vqvae, map_location=device))
model_vqvae = model_vqvae.to(device)
model_vqvae.eval()
reconstructed_images_train = decode_quantizes(model_vqvae, train_quantizes, device="cuda:1", batch_size=16)
reconstructed_images_val = decode_quantizes(model_vqvae, val_quantizes, device="cuda:1", batch_size=16)


train_dataset = ReconstructedDataset(reconstructed_images_train, train_labels)
val_dataset=  ReconstructedDataset(reconstructed_images_val, val_labels)



batchsize_modified=16
train_loader = DataLoader(train_dataset, batch_size=batchsize_modified, shuffle=True, num_workers=0)#256
val_loader = DataLoader(val_dataset, batch_size=batchsize_modified, shuffle=True, num_workers=0)

transform = transforms.Compose(
    [
        transforms.Resize((80,80)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)  
model.to(device)

nn.init.normal_(model.fc.weight)#, mean=0.0, std=0.01)
nn.init.zeros_(model.fc.bias)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 50  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader):
        inputs = transform(inputs)
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
    run["train/classifier-loss"].log(epoch_loss.item())

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    torch.save(model.state_dict(), f"/local/altamabp/checkpoint_correct/classifier/weights_epoch{str(epoch + 1).zfill(2)}.pth")

# Define classifier and load saved model(weights)
classifier = resnet50(pretrained=False)
models_list = os.listdir("/local/altamabp/checkpoint_correct/classifier")
models_list.sort()
for model_name in models_list:
    print(model_name)
    classifier.load_state_dict(torch.load(os.path.join("/local/altamabp/checkpoint_correct/classifier/",model_name)))
    classifier.to(device)
    classifier.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = transform(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    run["val/classifier-loss"].log(total_loss.item())
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'loss: {total_loss:.2f}%')

