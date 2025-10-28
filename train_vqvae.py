import argparse, sys, os, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils 
from torchvision.datasets import ImageNet
from tqdm import tqdm
from vqvae import FlatVQVAE
# from scheduler import CycleScheduler
import distributed as dist
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import neptune.new as neptune
from torch.optim.lr_scheduler import CyclicLR


os. nice (19)

run = neptune.init_run(
    project="UTKFaces",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
    capture_stdout=False,
    capture_stderr=False,
    source_files=["train_vqvae.py"]
)

def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.35
    diversity_loss_weight = 0.0001
    sample_size = 5

    mse_sum = 0
    mse_n = 0
    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)
        out, latent_loss, diversity_loss, codebook_usage = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )
            run["train/vqvae-mse"].log(recon_loss.item())
            run["train/vqvae-latent"].log(latent_loss.item())
            run["train/vqvae-epoch"].log(epoch + 1)
            run["train/vqvae-num_used_codebooks"].log(codebook_usage)
            

            if i % 9000 == 0:
                model.eval()
                sample = img[:sample_size]
                with torch.no_grad():
                    out, _ ,_,_= model(sample)
                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"image/sample/flat_vqvae_80x80codebook_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
                model.train()
batchsize_modified=1
def main(args):
    torch.cuda.set_device(3)  # Use GPU 1 (if desired)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder("/local/altamabp/UTKFace_dataset_subset_150000",transform=transform)
    data_loader = DataLoader(dataset, batch_size=batchsize_modified// args.n_gpu, shuffle=True, num_workers=12)#256
    print(len(dataset))
    print(len(dataset.classes))
    print(len(dataset[0]))
    print(dataset[0][0].shape)

    model = FlatVQVAE().to(device)


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    run["train/vqvae-lr"].log(args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CyclicLR(
        optimizer, 
        base_lr=args.lr * 0.1, 
        max_lr=args.lr, 
        step_size_up=len(data_loader) * args.epoch * 0.05, 
        mode="triangular",
        cycle_momentum=False)
    x=0
    for i in range(args.epoch):
        train(i, data_loader, model, optimizer, scheduler, device)
        x=i
        if dist.is_primary():
            model_path = os.path.join(args.save_path_models, f"model_epoch{i+1}_flat_vqvae80x80_64x400codebook.pth")
            torch.save(model.state_dict(), model_path)

        
    

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

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
