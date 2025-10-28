import os, yaml, argparse, json, torch, mlflow, sys
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from vqvae import FlatVQVAE
from data_utils import CustomImageNetDataV2, CustomLoader
from gpu_utils import select_gpus
import torch.distributed as tdist
import distributed as dist #local launcher/wrappers (avoid name clash with torch.distributed)
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms, utils 
from torch.utils.data import DataLoader

# ---------- Prioritize Task -----------
os.nice(19)

# ---------- Config & CLI ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ---------- Distributed Setup ----------
# =============================================================================
# def setup_distributed():
#     rank = int(os.environ['RANK'])
#     world_size = int(os.environ['WORLD_SIZE'])
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
#     return rank, world_size, local_rank
# =============================================================================
def setup_distributed():
    """
    Initialize process group if launched in distributed mode.
    Falls back to single-process defaults if env vars aren't set.
    """
    if all(k in os.environ for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')):
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        # initialize via torch.distributed
        tdist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return rank, world_size, local_rank
    else:
        # single-process fallback
        torch.cuda.set_device(0 if torch.cuda.is_available() else 'cpu')
        return 0, 1, 0





# ---------- Loader ----------
def get_loader(dataset, batch_size, shuffle, distributed, world_size=None, rank=None):
    return CustomLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        distributed=distributed, world_size=world_size, rank=rank).data_loader

# ---------- Load Best Params ----------
def load_best_params(model_path):
    for fname in os.listdir(model_path):
        if fname.startswith("best_vqvae_params_trial_") and fname.endswith(".json"):
            with open(os.path.join(model_path, fname), 'r') as f:
                return json.load(f)
    return None

# ---------- Load Best Model ----------
def load_best_model_path(model_path, epoch):
    return os.path.join(model_path, f"model_epoch{epoch}_flat_vqvae80x80_64x400codebook.pth") 

def save_indices_quantized_labels(model,loader,device,path_cfg, model_path, tag, best_params, calculate_loss=False):
    indices= [] # indices of all images - latent space
    quantizes = [] # codebooks of all images
    labels = [] # labels of all images
    
    if (calculate_loss):
        test_loss = 0.0
        model.eval()
        recon_criterion = nn.MSELoss()
    with torch.no_grad():
        for k, (inputs, batch_labels) in tqdm(enumerate(loader)):
            inputs = inputs.to(device)
            
            if (calculate_loss):
                recon, latent_loss, diversity_loss, _ = model(inputs)
                recon_loss = recon_criterion(recon, inputs)
                loss = recon_loss + best_params['latent_loss_weight'] * latent_loss + best_params['diversity_loss_weight'] * diversity_loss
                test_loss += loss.item() * inputs.size(0)

            ## ----------- store codebook, latent space, and corresponding labels -------------
            quant_b, _, id_b, _, _ = model.encode(inputs)
            outputs = model.decode(quant_b)
            indices.append(id_b.cpu())
            quantizes.append(quant_b.cpu())
            labels.extend(batch_labels.cpu().numpy().tolist())
            ### ---------- save reconstructed images ----------------
            for idx, (lbl, out) in enumerate(tqdm(zip(batch_labels, outputs), total=outputs.shape[0], desc="Saving reconstructions", leave=False)):#(zip(batch_labels, outputs)):
                lbl_int = int(lbl)
                #print(lbl_int)
                class_folder = os.path.join(path_cfg['recnstructed_imge'], str(lbl_int))
                os.makedirs(class_folder, exist_ok=True)
                save_file = os.path.join(class_folder, f"{lbl_int}_{k * loader.batch_size + idx + 1:05d}.png")
                # Save tensor CHW safely (normalize from [-1,1] to [0,1])
                utils.save_image(out, save_file, normalize=True, value_range=(-1, 1))

    # Concatenate all indices into a single tensor and save it
    indices_tensor = torch.cat(indices, dim=0)
    quantizes_tensor = torch.cat(quantizes, dim=0)
    labels = np.array(labels)

    indices_path = os.path.join(model_path, f"{tag}_latent_space_vqvae_80x80_codebook_64x456.npy")
    quantized_path = os.path.join(model_path, f"{tag}_codebook_vqvae_80x80_codebook_64x456.npy")

    np.save(indices_path, indices_tensor.numpy())
    np.save(quantized_path, quantizes_tensor.numpy())
    np.save(os.path.join(model_path, f'{tag}_labels.npy'), labels)
    
    if (calculate_loss):
        test_loss /= len(loader.dataset)
        print(f"✅ Final Test Loss: {test_loss:.4f}")
        rank, _, local_rank = setup_distributed()
        if rank == 0:
            mlflow.log_metric("test_loss", test_loss)
            mlflow.end_run()

# ---------- Main ----------
def main():
    args = parse_args()
    config = load_config(args.config)
    selected_gpu_ids, world_size = select_gpus(config['multiprocessing']['gpu'])
    rank, _, local_rank = setup_distributed()
    device = torch.device(f"cuda:{selected_gpu_ids[local_rank]}")

    # Load datasets
    path_cfg = config['path']
    #test_set = CustomImageNetDataV2(image_dir=path_cfg['UTKFace_test'], image_type='original', folder_label='int_id')

    model_path = path_cfg['vqvae_model']
    os.makedirs(model_path, exist_ok=True)

    # Load best parameters
    best_params = load_best_params(model_path)
    if best_params is None:
        print("⚠️ No best params found. Using defaults from config.")
        best_params = config['params']['vqvae']

    # Extract hyperparameters
    batch_size = best_params['batch_size']
    latent_loss_weight = best_params['latent_loss_weight']
    diversity_loss_weight = best_params['diversity_loss_weight']
    num_epochs = best_params['num_epochs']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']

     # Load best model checkpoint
    model_ckpt = load_best_model_path(model_path, epoch = num_epochs)
    model = FlatVQVAE().to(device)
    # Wrap with DDP only if torch.distributed is actually initialized
    if tdist.is_available() and tdist.is_initialized():
       model = DDP(model, device_ids=[device.index])

    model.load_state_dict(torch.load(model_ckpt))


    # ---------- MLflow Logging ----------
    if rank == 0:
        mlflow.start_run(run_name="vqvae_test_and_reconstructing")
        mlflow.log_params({
            "batch_size": batch_size,
            "latent_loss_weight": latent_loss_weight,
            "diversity_loss_weight": diversity_loss_weight,
            "num_epochs": num_epochs,
            "lr": lr,
            "weight_decay": weight_decay
        })

    # ---------- Test Evaluation ----------
    transform = transforms.Compose(
        [
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    
# calculated std and mean of UTKFace
#transforms.Normalize(mean=[0.6154290437698364, 0.46279090642929077, 0.38601234555244446], std=[0.24672381579875946, 0.22112978994846344, 0.21502047777175903])



    dataset_test = datasets.ImageFolder("/local/altamabp/UTKFace_dataset_subset_15000_structured", transform=transform)
    test_loader = get_loader(dataset_test, batch_size=batch_size, shuffle=False, distributed=False) 

    dataset_train = datasets.ImageFolder("/local/altamabp/UTKFace_dataset_subset_150000_structured", transform=transform)
    train_loader = get_loader(dataset_train, batch_size=batch_size, shuffle=False, distributed=False) 
    
    dataset_val = datasets.ImageFolder("/local/altamabp/UTKFace_dataset_subset_10000_structured", transform=transform)
    val_loader = get_loader(dataset_val, batch_size=batch_size, shuffle=False, distributed=False)
    
    save_indices_quantized_labels(model, test_loader, device, path_cfg, model_path, 'test', best_params, calculate_loss=True)
    #save_indices_quantized_labels(model, train_loader, device, path_cfg, model_path, 'train', best_params, calculate_loss=False)
    #save_indices_quantized_labels(model, val_loader, device, path_cfg, model_path, 'val', best_params, calculate_loss=False) 
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--save_path_models", default="/local/altamabp/checkpoint_correct/vqvae/")
    parser.add_argument("--save_path_imgs", default="/local/altamabp/image_correct/vqvae_reconstruction/")
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sched", type=str)
    # parser.add_argument('--ckpt_vqvae', type=str, default="checkpoint/flat_vqvae_80x80_144x456codebook_100class_051.pt")

    # parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)
    # Use custom multi-process launcher only when >1 GPU is requested.
    if args.n_gpu and args.n_gpu > 1:
        # NOTE: main() is arg-free; don't pass args to it.
        dist.launch(main, args.n_gpu, 1, 0, args.dist_url)
    else:
        # Single-process run avoids launcher & missing env var issues.
        main()

    

