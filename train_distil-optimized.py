import argparse
from sched import scheduler
import sys
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import distributed as dist

# =============================================================================
# DataParallel removed: on a single GPU, DataParallel wastes memory duplicating the model.
# 
# AMP (autocast + GradScaler): cuts activation/grad memory roughly in half for Transformers.
# 
# Gradient accumulation: keeps the effective batch size while lowering --batch_size if needed.
# 
# Allocator tweak expandable_segments:True: reduces fragmentation (helps after many steps).
# =============================================================================



# help the CUDA allocator avoid fragmentation (esp. with variable batch shapes)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
 

os.environ["TRANSFORMERS_NO_TF"] = "1"    # block TensorFlow import inside transformers
os.environ["TRANSFORMERS_NO_FLAX"] = "1"  # block Flax, too (optional)
# optional hard stop if TF somehow slips in:
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # temporary safety valve


from transformers import DistilBertForMaskedLM, DistilBertConfig
import neptune.new as neptune

run = neptune.init_run(
    project="UTKFaces",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
    capture_stdout=False,
    capture_stderr=False,
    #with_id="distil",
    source_files=["train_distil.py"]
)


class CustomDataset(Dataset):
    def __init__(self, inputs, labels, mask_perc, n_train_samples, n_tokens, mask_token):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        return self.inputs[idx], self.labels[idx]


def train(epoch, loader, model, optimizer, scheduler, device, val_loader=None, grad_accum_steps=1, scaler=None):
    if dist.is_primary():
        loader = tqdm(loader)
    model.train()
    
    optimizer.zero_grad(set_to_none=True)
    for step, (input, label) in enumerate(loader):
        input = input.to(device)
        label = label.to(device)
        

        # ---- mixed precision forward/backward ----
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs_embeds=input, labels=label)
                loss = outputs.loss.mean()
            scaler.scale(loss / grad_accum_steps).backward()
        else:
            outputs = model(inputs_embeds=input, labels=label)
            loss = outputs.loss.mean()
            (loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; loss: {loss:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )
        run["train/transformer-loss"].log(loss.item())
        run["train/transformer-lr"].log(lr)


    ##validation
    if val_loader is not None:
        if dist.is_primary():
            val_loader = tqdm(val_loader)
        model.eval()
        average_loss = 0

        total, count = 0.0, 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16 if scaler is not None else None):
            for input, label in val_loader:
                input = input.to(device)
                label = label.to(device)
                outputs = model(inputs_embeds=input, labels=label)
                val_loss = outputs.loss.mean()
                bs = input.size(0)
                total += val_loss.item() * bs
                count += bs
                if dist.is_primary():
                    val_loader.set_description(f"Validation loss: {val_loss.item():.5f}")
                    run["validation/transformer-loss"].log(val_loss.item())
        average_loss = total / max(1, count)
        if dist.is_primary():
            run["validation/transformer-average_loss_per_epoch"].log(average_loss)
        return average_loss
        

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.distributed = dist.get_world_size() > 1


    #### validation set###
    val_indices = np.load('/local/altamabp/checkpoint_correct/vqvae/val_latent_space_vqvae_80x80_codebook_64x456.npy')
    n, h, w = val_indices.shape
    val_indices = val_indices.reshape(n, h * w)
    
    val_quantizes = np.load('/local/altamabp/checkpoint_correct/vqvae/val_codebook_vqvae_80x80_codebook_64x456.npy')
    n, c, h, w = val_quantizes.shape
    val_quantizes = val_quantizes.transpose(0, 2, 3, 1)
    val_quantizes = val_quantizes.reshape(n, h * w, c)
    
    #### train set###
    train_indices = np.load('/local/altamabp/checkpoint_correct/vqvae/train_latent_space_vqvae_80x80_codebook_64x456.npy')
    n, h, w = train_indices.shape
    train_indices = train_indices.reshape(n, h * w)

    train_quantizes = np.load('/local/altamabp/checkpoint_correct/vqvae/train_codebook_vqvae_80x80_codebook_64x456.npy')
    n, c, h, w = train_quantizes.shape
    train_quantizes = train_quantizes.transpose(0, 2, 3, 1)
    train_quantizes = train_quantizes.reshape(n, h * w, c)
    
    
############ Data prepration and masking/ train set##############

    n_train_samples = train_quantizes.shape[0]
    d_embed_vec = train_quantizes.shape[2]
    n_tokens = train_quantizes.shape[1]
    print(f'n_train_samples: {n_train_samples}')
    print(f'train_quantizes.shape: {train_quantizes.shape}')
    print(f'n_tokens: {n_tokens}')

    mask_token =0 
    mask_token_label = -100
    mask_perc = 0.75
    mask_train = np.random.default_rng().choice([True, False], size=(n_train_samples, n_tokens), p=[mask_perc, 1 - mask_perc])
    run["data/mask_prec"].log(mask_perc)
    train_quantizes = train_quantizes.reshape((n_train_samples, n_tokens, d_embed_vec))
    train_indices = train_indices.reshape((n_train_samples, n_tokens))
    train_quantizes[mask_train] = mask_token
    masked_train_indices = np.copy(train_indices)
    masked_train_indices[mask_train] = mask_token

    train_indices_label = np.copy(train_indices)
    train_indices_label[~mask_train] = mask_token_label
    train_quantizes = torch.from_numpy(train_quantizes)
    masked_train_indices = torch.from_numpy(masked_train_indices)
    train_indices_label = torch.from_numpy(train_indices_label)

    indices = set(train_indices.flatten())
    indices = sorted(indices)
    vocab_size = indices[-1] + 1

############### Data prepration and masking/ validation set#####################

    n_val_samples = val_quantizes.shape[0]
    n_val_tokens = val_quantizes.shape[1]
    mask_val = np.random.default_rng().choice([True, False], size=(n_val_samples, n_val_tokens), p=[mask_perc, 1 - mask_perc])
    val_quantizes = val_quantizes.reshape((n_val_samples, n_val_tokens, d_embed_vec))
    val_indices = val_indices.reshape((n_val_samples, n_val_tokens))
    masked_val_data = np.copy(val_quantizes)
    masked_val_data[mask_val] = mask_token
    val_indices_label = np.copy(val_indices)
    val_indices_label[~mask_val] = mask_token_label
    masked_val_data = torch.from_numpy(masked_val_data)
    val_indices_label = torch.from_numpy(val_indices_label)

    val_indices = set(val_indices.flatten())
    val_indices = sorted(val_indices)
    val_vocab_size = val_indices[-1] + 1


################Create Data loaders###################

    print(f"vocab_size: {vocab_size} ")
    print(f"z_q shape: {train_quantizes.shape} ")
    print(f"indices shape: {train_indices.shape} ")
    print(n_train_samples)
    print(d_embed_vec)
    print(n_tokens)

    train_data = CustomDataset(train_quantizes, train_indices_label, mask_perc,n_train_samples,n_tokens, mask_token)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = CustomDataset(masked_val_data, val_indices_label, mask_perc,n_val_samples,n_val_tokens, mask_token)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)


#####################Model Config###########################
    cfg = DistilBertConfig(
            vocab_size=vocab_size,
            hidden_size=d_embed_vec,
            sinusoidal_pos_embds=False,
            n_layers=6,
            n_heads=4,
            max_position_embeddings=n_tokens
    )

    model = DistilBertForMaskedLM(cfg).to(device)
    # model.load_state_dict(torch.load(args.ckpt_distil))

    #Count model parameters
    parameters = list(model.parameters())
    if True:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    print("Parameters:")
    print("trainable")
    print(sum(p.numel() for p in unique))

    # DDP left disabled here since we are using a single visible GPU.

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    scheduler = None
    if args.sched == "linearW":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0006,
            pct_start = 0.01,
            steps_per_epoch=len(train_dataloader),
            epochs=args.epoch,
            anneal_strategy='linear')
    warmup = 0.01
    run["parameters/transformer-warmup"].log(warmup)



    # AMP scaler + gradient accumulation
    scaler = torch.cuda.amp.GradScaler()
    grad_accum_steps = max(1, 2 if args.batch_size >= 8 else 1)  # tweak as needed

    #Train
    j=0
    min_validation_loss = np.inf
    for i in range(args.epoch):
        j = j+1
        print(len(train_dataloader))
        torch.cuda.empty_cache()
        validation_loss = train(i, train_dataloader, model, optimizer, scheduler, device,
                                val_loader=val_dataloader, grad_accum_steps=grad_accum_steps, scaler=scaler)

        run["train/transformer-epoch"].log(j)
        if validation_loss< min_validation_loss:
            min_validation_loss = validation_loss
            print(f'Validation loss decreased to : {min_validation_loss}')
        
        if dist.is_primary():
            torch.save(model.state_dict(), f"/local/altamabp/checkpoint_correct/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch{str(j).zfill(3)}.pt")


batchsize_modified=16
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    #parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--epoch", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=batchsize_modified)#256
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sched", type=str, default="linearW")
    # parser.add_argument('--ckpt_distil', type=str, default="/home/abghamtm/work/masking_comparison/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch006.pt")
    args = parser.parse_args()

    params = {
    "lr": args.lr,
    "bs": args.batch_size,
    "scheduler": args.sched
}
    run["parameters"] = params

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

