import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from torchvision import utils, datasets, transforms
import distributed as dist
import matplotlib.pyplot as plt
from transformers import DistilBertForMaskedLM, DistilBertConfig
from vqvae import FlatVQVAE, EnhancedFlatVQVAE
from PIL import Image
import neptune.new as neptune
from torchvision.models import resnet50, ResNet50_Weights
import math, sys, os
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

run = neptune.init_run(
    project="UTKFaces",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
    capture_stdout=False,
    capture_stderr=False,
    #with_id="distil",
    source_files=["flat_reconstruct_random_based.py"]
)

import torch.nn.functional as F


def make_match_strip_plot(pred, true, save_path=None, figsize=(12, 1.6), title_prefix="Match Strip"):
    """
    Plot a 'match strip' (green = correct, 0/1 row) comparing integer indices in pred vs true.
    - pred, true: torch.Tensor or np.ndarray; any shape -> flattened to 1D
    - save_path: optional path to save the figure (e.g., Path(.../"..._strip.png"))
    - returns (fig, ax)
    """
    # to numpy 1D
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().flatten().cpu().numpy()
    else:
        pred_np = np.asarray(pred).flatten()

    if isinstance(true, torch.Tensor):
        true_np = true.detach().flatten().cpu().numpy()
    else:
        true_np = np.asarray(true).flatten()

    # match vector: 1 = correct, 0 = mismatch
    match = (pred_np == true_np).astype(np.int32)
    acc = (match.mean() * 100.0) #if match.size else 0.0

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(match[np.newaxis, :], aspect='auto', cmap='Greens', vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xlabel("Token Position")
    ax.set_title(f"{title_prefix} — green = correct ({acc:.1f}% acc)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, ax


def save_histogram_pred_true(pred, true, save_path, title="Index Distribution (Pred vs True)"):
    """
    Side-by-side histogram comparing counts per codebook index for pred vs true.
    pred, true: torch.Tensor or np.ndarray of integer indices; any shape -> flattened.
    save_path: file path to save the PNG (e.g., "..._hist.png")
    """
    # to numpy 1D
    pred = pred.detach().flatten().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred).flatten()
    true = true.detach().flatten().cpu().numpy() if isinstance(true, torch.Tensor) else np.asarray(true).flatten()

    num_bins = int(max(pred.max(initial=0), true.max(initial=0)) + 1)
    idx = np.arange(num_bins)
    bins = np.arange(num_bins + 1) - 0.5  # center bars on integer indices

    pred_counts, _ = np.histogram(pred, bins=bins)
    true_counts, _ = np.histogram(true, bins=bins)

    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.45
    ax.bar(idx - w/2, pred_counts, width=w, label=f"Predicted (total={pred_counts.sum()})")
    ax.bar(idx + w/2, true_counts, width=w, label=f"True (total={true_counts.sum()})")
    ax.set_xlabel("Codebook Index")
    ax.set_ylabel("Count per Index")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def gradual_masking( distil, quantized,indices, mask_percentage):
    total_num = quantized.shape[1] 
    total_unmasked_number = (int) (total_num * (1-mask_percentage))
    unmask_index = (int) (total_num/2)
    quantized_masked = torch.zeros_like(quantized)
    mask = torch.ones(quantized.shape[:2], dtype=torch.bool)
    already_unmasked = set()
    for i in range(0,total_unmasked_number):
        mask[0,unmask_index] = False
        already_unmasked.add(unmask_index)
        quantized_masked[0, unmask_index] = quantized[0,unmask_index]
        outputs = distil(inputs_embeds = quantized_masked, output_hidden_states = True)
        max_logits, max_indices = torch.max(outputs.logits, dim=-1)
        sorted_logits, sorted_indices = torch.sort(max_logits[0])
        for min_index in sorted_indices:
            if min_index.item() not in already_unmasked:
                unmask_index = min_index.item()
                break
    indices_masked = indices.clone()
    #indices_masked[~mask[0]] = -100
    return quantized_masked,indices_masked, mask[0]

def random_mask(unmasked, indices_unmasked,n_sample, n_token, mask_perc):
    
    mask = np.random.default_rng().choice([True, False], size=(1, n_token), p=[mask_perc, 1 - mask_perc])
    masked = unmasked.clone()
    masked[mask] = 0  # Assuming 0 is the mask token
    indices_masked = indices_unmasked.clone()
    #indices_masked[~mask[0]] = -100 # Assuming -100 is the mask label token
   
    return masked, indices_masked, mask#[0][0]

def confidence_based_mask(logits,
                                 unmasked, indices_unmasked, n_token,
                                 length, mask_percentage):

    max_logits = np.max(logits.detach().cpu().numpy(), axis=-1)
    flattened_max_logits = max_logits.flatten()
    num_locations = flattened_max_logits.size
    num_masked_locations = int(num_locations * mask_percentage)
    sorted_indices = np.argsort(flattened_max_logits)[::-1]
    mask = np.zeros_like(flattened_max_logits, dtype=bool)
    mask[sorted_indices[:num_masked_locations]] = True

    masked = np.copy(unmasked)
    masked[mask] = 0  # Assuming 0 is the mask token
    masked = torch.from_numpy(masked)
    indices_masked = np.copy(indices_unmasked)
    #indices_masked[~mask] = -100 # Assuming -100 is the mask label token
    indices_masked = torch.from_numpy(indices_masked)

    return masked, indices_masked, mask



def attach_bias_mask_feat(labels, masked_quantizes, mask, mask_perc):
    
    idx_t = torch.as_tensor(labels, dtype=torch.long, device=masked_quantizes.device)  # (N,)

    N, T, _ = masked_quantizes.shape

    idx_t_exp = idx_t.unsqueeze(1).expand(N, T)                         # (N, T)

    one_hot_labels = torch.nn.functional.one_hot(idx_t_exp, num_classes=10).to(dtype=masked_quantizes.dtype)  # (N,T,C)

    masked_exp_quantizes = torch.cat([masked_quantizes, one_hot_labels], dim=2)


    print('masked_exp_quantizes.shape: ',masked_exp_quantizes.shape)  # torch.Size([160000, 400, 74])
    # masked_exp_train_quantizes: torch.FloatTensor (N, T, 74)
    # mask_train: numpy bool array (N, T)  True = masked

    device = masked_exp_quantizes.device
    dtype  = masked_exp_quantizes.dtype
    
   # 1) NumPy -> Torch, cast to float (1.0 masked, 0.0 unmasked)
    mask_feat = torch.as_tensor(mask, device=device).to(dtype)   # (N, T)

    # 2) Add feature axis
    mask_feat = mask_feat.unsqueeze(-1)                                 # (N, T, 1)

    # 3) Concatenate
    masked_exp_mask_feat_quantizes = torch.cat([masked_exp_quantizes, mask_feat], dim=-1)  # (N, T, 75)
   
    return masked_exp_mask_feat_quantizes

def attach_bias_mask_feat(labels, masked_quantizes, mask, mask_perc,num_classes=10):
    # labels: (N,)
    labels= torch.as_tensor(labels, dtype=torch.long).unsqueeze(0)
    idx_t = torch.as_tensor(labels, dtype=torch.long)#, device=masked_quantizes.device)
    assert idx_t.ndim == 1, f"labels must be (N,), got {tuple(idx_t.shape)}"

    # features: accept (N,D) or (N,T,D); normalize to (N,T,D)
    if masked_quantizes.ndim == 2:
        # treat as a single time step: (N, D) -> (N, 1, D)
        masked_quantizes = masked_quantizes.unsqueeze(1)
    elif masked_quantizes.ndim != 3:
        raise ValueError(f"masked_quantizes must be (N,D) or (N,T,D), got {tuple(masked_quantizes.shape)}")

    N, T, D = masked_quantizes.shape
    assert idx_t.size(0) == N, f"batch mismatch: labels N={idx_t.size(0)} vs features N={N}"

    # expand labels across time: (N,) -> (N,1) -> (N,T)
    idx_t_exp = idx_t.unsqueeze(1).expand(N, T)              # long dtype

    # one-hot: (N,T) -> (N,T,C); cast to feature dtype for concat
    one_hot_labels = torch.nn.functional.one_hot(idx_t_exp, num_classes=num_classes).to(masked_quantizes.dtype)

    masked_exp_quantizes = torch.cat([masked_quantizes, one_hot_labels], dim=2)

    m = torch.as_tensor(mask, dtype=torch.long)

    if m.ndim == 0:
        # single scalar -> broadcast to (N,T,1)
        m = m.to(torch.long).expand(N, T, 1)
    elif m.ndim == 1:
        if m.numel() == T:               # (T,) -> (1,T,1) -> (N,T,1)
            m = m.view(1, T, 1).to(torch.long).expand(N, T, 1)
        elif m.numel() == N:             # (N,) -> (N,1,1) -> (N,T,1)
            m = m.view(N, 1, 1).to(torch.long).expand(N, T, 1)
        else:
            raise ValueError(f"mask 1D length must be N({N}) or T({T}), got {m.numel()}")
    elif m.ndim == 2:
        if m.shape == (N, T):            # (N,T) -> (N,T,1)
            m = m.unsqueeze(-1).to(torch.long)
        elif m.shape == (T, 1):          # (T,1) -> (1,T,1) -> (N,T,1)
            m = m.view(1, T, 1).to(torch.long).expand(N, T, 1)
        elif m.shape == (1, T):          # (1,T) -> (1,T,1) -> (N,T,1)
            m = m.view(1, T, 1).to(torch.long).expand(N, T, 1)
        else:
            raise ValueError(f"mask 2D must be (N,T), (T,1), or (1,T); got {tuple(m.shape)}")
    elif m.ndim == 3:
        # allow broadcastable shapes like (1,T,1), (N,1,1)
        if m.shape == (N, T, 1):
            m = m.to(torch.long)
        elif (m.size(0) in (1, N)) and (m.size(1) in (1, T)) and (m.size(2) in (1,)):
            m = m.to(torch.long).expand(N, T, 1)
        else:
            raise ValueError(f"mask 3D must be (N,T,1) or broadcastable to it; got {tuple(m.shape)}")
    else:
        raise ValueError(f"mask must be 0D/1D/2D/3D; got {m.ndim}D")                         # (N, T, 1)

    # 3) Concatenate
    masked_exp_mask_feat_quantizes = torch.cat([masked_exp_quantizes, m], dim=-1)  # (N, T, 75)
   
    return masked_exp_mask_feat_quantizes


def _to_np_1d(x):
    """Accepts torch.Tensor or np.ndarray of shape [1,N] or [N]; returns np.ndarray [N]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().flatten().cpu().numpy()
    else:
        x = np.asarray(x).flatten()
    return x

def plot_index_histograms(pred, true, title="Index Distribution (Pred vs True)"):
    """Side-by-side histogram of counts per codebook index."""
    pred = _to_np_1d(pred)
    true = _to_np_1d(true)

    num_bins = int(max(pred.max(initial=0), true.max(initial=0)) + 1)
    idx = np.arange(num_bins)
    bins = np.arange(num_bins + 1) - 0.5

    pred_counts, _ = np.histogram(pred, bins=bins)
    true_counts, _ = np.histogram(true, bins=bins)

    plt.figure(figsize=(12,5))
    w = 0.45
    plt.bar(idx - w/2, pred_counts, width=w, label=f"Predicted (total={pred_counts.sum()})")
    plt.bar(idx + w/2, true_counts, width=w, label=f"True (total={true_counts.sum()})")
    plt.xlabel("Codebook Index"); plt.ylabel("Count")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

def plot_confusion_matrix(pred, true, normalize_rows=True, title="Confusion Matrix"):
    """Confusion matrix: rows=true index, cols=pred index."""
    pred = _to_np_1d(pred).astype(int)
    true = _to_np_1d(true).astype(int)

    num_bins = int(max(pred.max(initial=0), true.max(initial=0)) + 1)
    cm = np.zeros((num_bins, num_bins), dtype=np.float64)
    for t, p in zip(true, pred):
        cm[t, p] += 1

    if normalize_rows:
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / row_sums

    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.colorbar(label='Proportion per True Index' if normalize_rows else 'Count')
    plt.xlabel("Predicted Index"); plt.ylabel("True Index")
    plt.title(title); plt.tight_layout(); plt.show()

def plot_match_strip(pred, true, title="Match Strip (green = correct)"):
    """Single-row image showing token-wise correctness along sequence."""
    pred = _to_np_1d(pred)
    true = _to_np_1d(true)
    match = (pred == true).astype(np.int32)

    acc = match.mean() if match.size else 0.0
    plt.figure(figsize=(12,1.6))
    plt.imshow(match[np.newaxis, :], aspect='auto', cmap='Greens', vmin=0, vmax=1)
    plt.yticks([]); plt.xlabel("Token Position")
    plt.title(f"{title} — accuracy: {acc*100:.1f}%")
    plt.tight_layout(); plt.show()

def plot_index_series(pred, true, max_points=None, title="Indices per Position (True vs Pred)"):
    """Line/marker plot over positions; optionally limit to first N points to reduce clutter."""
    pred = _to_np_1d(pred)
    true = _to_np_1d(true)

    if max_points is not None:
        pred = pred[:max_points]
        true = true[:max_points]

    x = np.arange(len(pred))
    plt.figure(figsize=(12,4))
    plt.plot(x, true, marker='o', linestyle='-', linewidth=1, markersize=3, label='True')
    plt.plot(x, pred, marker='x', linestyle='--', linewidth=1, markersize=3, label='Predicted')

    mismatch = pred != true
    if mismatch.any():
        plt.scatter(x[mismatch], pred[mismatch], s=18, marker='x', label='Mismatch (pred)')
        plt.scatter(x[mismatch], true[mismatch], s=18, marker='o', label='Mismatch (true)')

    plt.xlabel("Token Position"); plt.ylabel("Index")
    plt.title(title); plt.legend(ncol=2); plt.tight_layout(); plt.show()

def main(args):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=1)

port = (
    2 ** 15
    + 2 ** 14
    + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
)
parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
parser.add_argument('--ckpt_vqvae', type=str, default="./checkpoint/vqvae/model_epoch100_flat_vqvae80x80_64x400codebook.pth")
parser.add_argument('--ckpt_distil', type=str, default="./checkpoint/distil/80x80_100_UTKFace_flat_144x400codebook_50mask_epoch100_without-ignore-index.pt")
parser.add_argument('--ckpt_resnet50', type=str, default="./checkpoint/classifier/weights_epoch100_fullTrain.pth")

args = parser.parse_args()
dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

#def main(args):
#device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.set_device(2)
    torch.cuda.empty_cache()
else:
    device= 'cpu'
    
args.distributed = dist.get_world_size() > 1

labels= np.load('./checkpoint/vqvae/test_labels.npy')
 

indices = np.load('./checkpoint/vqvae/test_latent_space_vqvae_80x80_codebook_64x456.npy')
n, h, w = indices.shape
indices = indices.reshape(n, h * w)

quantizes = np.load('./checkpoint/vqvae/test_codebook_vqvae_80x80_codebook_64x456.npy')
quant_b = quantizes
n, c, h, w = quantizes.shape
quantizes = quantizes.transpose(0, 2, 3, 1)
quantizes = quantizes.reshape(n, h * w, c)

#Bottom data and parameters
n_sample = quantizes.shape[0]
d_embed_vec = quantizes.shape[2]
n_token = np.prod(quantizes.shape[1])
quantizes = quantizes.reshape((n_sample, n_token, d_embed_vec))
length = int(math.sqrt(n_token))
indices = indices.reshape((n_sample, n_token))
indices_to_sort = set(indices.flatten())
indices_to_sort = sorted(indices_to_sort)
vocab_size = indices_to_sort[-1] + 1






#Define Distilbert model
cfg = DistilBertConfig(
        vocab_size=vocab_size,
        hidden_size=d_embed_vec+11,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=5,
        max_position_embeddings=n_token
)
model_distil = DistilBertForMaskedLM(cfg).to(device)
model_distil.load_state_dict(torch.load(args.ckpt_distil,map_location=torch.device('cpu')))
model_distil = model_distil.to(device)
model_distil.eval()

#Define VQVAE model
model_vqvae = EnhancedFlatVQVAE().to(device) #FlatVQVAE().to(device)
model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=torch.device('cpu')))
model_vqvae = model_vqvae.to(device)
model_vqvae.eval()

# Define classifier and load saved model(weights)
# =============================================================================
#     weights = ResNet50_Weights.IMAGENET1K_V2
#     preprocess = weights.transforms()
#     classifier = resnet50(pretrained=False)
# =============================================================================

classifier = resnet50(weights=None)

classifier.fc = nn.Linear(classifier.fc.in_features, 10)  
classifier.load_state_dict(torch.load(args.ckpt_resnet50, map_location=torch.device('cpu')))
classifier.to(device)
classifier.eval()
classifier.weights = ResNet50_Weights.DEFAULT
preprocess = classifier.weights.transforms()



preprocess = transforms.Compose(
     [
         transforms.Resize((80,80)),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
     ]
 )

mask_percentages = np.arange(0.1, 1.1, 0.1)
mask_percentages = np.append(mask_percentages,[.85,.95])
mask_percentages = np.sort(mask_percentages)

average_errors = []

for x in range(1,quantizes.shape[0], 1000):
    reconstruction_errors = []
    cross_entropy_class_err = []
    
    criterion = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    correct_random_pred = 0
    tot_sample = 0
    
    for perc in mask_percentages:
        #if x%5000 ==0:

        #print(x)
        q = torch.from_numpy(quantizes[x])
        index = torch.from_numpy(indices[x])
        index = index.to(device)
        q = q.to(device)
        q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
        
        with torch.no_grad():
            q_masked, index_masked, mask = random_mask(q, index , n_sample, n_token,perc)                                
            q_masked = q_masked.to(device)
            index_masked = index_masked.to(device)
            
            masked_exp_mask_feat_train_quantizes = attach_bias_mask_feat (labels[x], q_masked, mask, perc)
        
            outputs = model_distil(inputs_embeds = masked_exp_mask_feat_train_quantizes, output_hidden_states = False)
            confidence_based_prediction = torch.argmax(outputs.logits, dim=2)
            confidence_based_recons_index = index.clone()
            #print('mask: ', mask)
            
            # mask: (1, 400) -> (400,), boolean
            m = np.squeeze(mask, axis=0).astype(bool)   # or: m = mask[0].astype(bool)
            
            # replace where mask is True
            #confidence_based_recons_index[m] = confidence_based_prediction[0][m]
            
            #Reconstruct with distil predictions
            confidence_based_recons_index = confidence_based_recons_index.to(device)
            distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_prediction, (1,length,length)).to(device))#confidence_based_recons_index
        
            #Reconstruct Original
            vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
            index_masked_forvis = index.clone()
            index_masked_forvis[mask[0]]=0
            vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))
        
            # Label outputs
            vqvae_out = vqvae_out.unsqueeze(0)
            vqvae_img = preprocess(vqvae_out)
            vqvae_img = vqvae_img.to(device)
            vqvae_img_prob = classifier(vqvae_img)
            _, vqvae_img_label = torch.max(vqvae_img_prob, 1)
            
            rand_mask_img = preprocess(distil_out)
            rand_mask_img = rand_mask_img.to(device)
            rand_mask_img_prob = classifier(rand_mask_img)
            _, rand_mask_img_label = torch.max(rand_mask_img_prob, 1)
            correct_random_pred += (rand_mask_img_label == vqvae_img_label).sum().item()
            print(f'rand_mask_img_label is {rand_mask_img_label}')
            print(f'vqvae_img_label is {vqvae_img_label}')
            #print(f'rand_mask_img_label is {rand_mask_img_label.item()}')
            #print(f'vqvae_img_label is {vqvae_img_label.item()}')
            print(f'correct_random_pred is {correct_random_pred}')
            tot_sample += 1
            print(tot_sample)
        
            # your main image path (same as in utils.save_image)
            base_path = Path("./image_correct/vqvae_reconstruction") / \
                f"80x80_random_{vqvae_img_label.item()}_{rand_mask_img_label.item()}_{int(perc*100)}_{str(x).zfill(5)}.png"
            
            # save the match strip 
            strip_path = base_path.with_name(base_path.stem + "_strip" + base_path.suffix)
            make_match_strip_plot(confidence_based_prediction, index, save_path=strip_path)
            
            # save the histogram figure
            hist_path = base_path.with_name(base_path.stem + "_hist" + base_path.suffix)
            save_histogram_pred_true(confidence_based_prediction, confidence_based_recons_index, save_path=hist_path)
            
            # save your main grid as you already do
            utils.save_image(
                torch.cat([vqvae_out, vqvae_masked_out, distil_out], 0).to(device),
                f"./image_correct/vqvae_reconstruction/80x80_random_{vqvae_img_label.item()}_{rand_mask_img_label.item()}_{int(perc*100)}_{str(x).zfill(5)}.png",
                nrow=3,
                normalize=True,
                value_range=(-1, 1),
            )


        recon_loss = criterion(distil_out, vqvae_out)
        run["recons/mse_per_image_random_mask"].log(recon_loss.item())
        reconstruction_errors.append(recon_loss.item())
        class_loss = criterion_class(rand_mask_img_prob, vqvae_img_prob)
        run["recons/cross_entropy_per_image_random_mask"].log(class_loss.item())
        cross_entropy_class_err.append(class_loss.item())
            
    run["recons/average_mse_per_precision_random_mask"].log(np.mean(reconstruction_errors))
    run["recons/average_cross_entropy_error_random_mask"].log(np.mean(cross_entropy_class_err))
    average_errors.append(np.mean(reconstruction_errors))
    pred_acc_random_mask = correct_random_pred/tot_sample
    pred_err_random_mask = 1-pred_acc_random_mask
    run["recons/average_classification_accuracy_random"].log(pred_err_random_mask)


# Plotting the reconstruction errors
plt.plot(mask_percentages * 100, average_errors, marker='o')
plt.xlabel('Mask Percentage')
plt.ylabel('Average Reconstruction Error (MSE)')
plt.title('Reconstruction Error for Random Mask vs Mask Percentage')
plt.grid(True)
plot_path = './image_correct/distil_reconstruction/random_error_vs_precision.png'
plt.savefig(plot_path)
plt.show()#close()



# =============================================================================
# for perc in mask_percentages:
#     reconstruction_errors = []
#     cross_entropy_class_err = []
# 
#     criterion = nn.MSELoss()
#     criterion_class = nn.CrossEntropyLoss()
#     correct_random_pred = 0
#     tot_sample = 0
#     for x in range(0,quantizes.shape[0], 5000):
#         #if x%5000 ==0:
#             print(x)
#             q = torch.from_numpy(quantizes[x])
#             index = torch.from_numpy(indices[x])
#             index = index.to(device)
#             q = q.to(device)
#             q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
#             
#             with torch.no_grad():
#                 q_masked, index_masked, mask = random_mask(q, index , n_sample, n_token,perc)                                
#                 q_masked = q_masked.to(device)
#                 index_masked = index_masked.to(device)
#                 
#                 masked_exp_mask_feat_train_quantizes = attach_bias_mask_feat (labels[x], q_masked, mask, perc)
# 
#             #Fill in predicted tokens
#             with torch.no_grad():
#                 outputs = model_distil(inputs_embeds = masked_exp_mask_feat_train_quantizes, output_hidden_states = True)
#                 confidence_based_prediction = torch.argmax(outputs.logits, dim=2)
#                 confidence_based_recons_index = index
#                 print('mask: ', mask)
#                 
#                 confidence_based_recons_index=confidence_based_prediction
# # =============================================================================
# #                     for p in range(0,n_token):
# #                         if(mask):#[p]):
# #                             #confidence_based_recons_index[p] = confidence_based_prediction.detach().cpu().numpy()[0][p] 
# #                             confidence_based_recons_index[p] =  confidence_based_prediction[0][p]
# # =============================================================================
#                 #Reconstruct with distil predictions
#                 confidence_based_recons_index = confidence_based_recons_index.to(device)
#                 distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))
# 
#                 #Reconstruct Original
#                 vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
#                 index_masked_forvis = index.clone()
#                 index_masked_forvis[mask]=0
#                 vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))
# 
#                 # Label outputs
#                 vqvae_out = vqvae_out.unsqueeze(0)
#                 vqvae_img = preprocess(vqvae_out)
#                 vqvae_img = vqvae_img.to(device)
#                 vqvae_img_prob = classifier(vqvae_img)
#                 _, vqvae_img_label = torch.max(vqvae_img_prob, 1)
#                 
#                 rand_mask_img = preprocess(distil_out)
#                 rand_mask_img = rand_mask_img.to(device)
#                 rand_mask_img_prob = classifier(rand_mask_img)
#                 _, rand_mask_img_label = torch.max(rand_mask_img_prob, 1)
#                 correct_random_pred += (rand_mask_img_label == vqvae_img_label).sum().item()
#                 print(f'rand_mask_img_label is {rand_mask_img_label}')
#                 print(f'vqvae_img_label is {vqvae_img_label}')
#                 #print(f'rand_mask_img_label is {rand_mask_img_label.item()}')
#                 #print(f'vqvae_img_label is {vqvae_img_label.item()}')
#                 print(f'correct_random_pred is {correct_random_pred}')
#                 tot_sample += 1
#                 print(tot_sample)
# 
# 
#                 #if x%1000 ==0:
#                 utils.save_image(
#                     torch.cat([vqvae_out, vqvae_masked_out, distil_out], 0).to(device),
#                     f"./image_correct/vqvae_reconstruction/random/80x80_random_{vqvae_img_label.item()}_{rand_mask_img_label.item()}_{int(perc*100)}_{str(x).zfill(5)}.png",
#                     nrow=3,
#                     normalize=True,
#                     value_range=(-1, 1),
#                 )
#             
#             recon_loss = criterion(distil_out, vqvae_out)
#             #run["recons/mse_per_image_random_mask"].log(recon_loss.item())
#             reconstruction_errors.append(recon_loss.item())
#             class_loss = criterion_class(rand_mask_img_prob, vqvae_img_prob)
#             #run["recons/cross_entropy_per_image_random_mask"].log(class_loss.item())
#             cross_entropy_class_err.append(class_loss.item())
#     
#     #run["recons/average_mse_per_precision_random_mask"].log(np.mean(reconstruction_errors))
#     #run["recons/average_cross_entropy_error_random_mask"].log(np.mean(cross_entropy_class_err))
#     average_errors.append(np.mean(reconstruction_errors))
#     pred_acc_random_mask = correct_random_pred/tot_sample
#     pred_err_random_mask = 1-pred_acc_random_mask
#     #run["recons/average_classification_accuracy_random"].log(pred_err_random_mask)
# 
# 
# # Plotting the reconstruction errors
# plt.plot(mask_percentages * 100, average_errors, marker='o')
# plt.xlabel('Mask Percentage')
# plt.ylabel('Average Reconstruction Error (MSE)')
# plt.title('Reconstruction Error for Random Mask vs Mask Percentage')
# plt.grid(True)
# plot_path = './image_correct/distil_reconstruction/random_error_vs_precision.png'
# plt.savefig(plot_path)
# plt.show()#close()
# =============================================================================
#run['random_error_vs_percentage'].upload(plot_path)

#if __name__ == "__main__":


