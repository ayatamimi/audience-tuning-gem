# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from torchvision import utils, datasets, transforms
import distributed as dist
import matplotlib.pyplot as plt
from transformers import DistilBertForMaskedLM, DistilBertConfig
from vqvae import FlatVQVAE
from PIL import Image
import neptune.new as neptune
from torchvision.models import resnet50, ResNet50_Weights
import math, sys, os
from torch.utils.data import DataLoader
from tqdm import tqdm

run = neptune.init_run(
    project="UTKFaces",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
    capture_stdout=False,
    capture_stderr=False,
    #with_id="distil",
    source_files=["audience-tuning-control_experiments.py"]
)


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


def random_mask(unmasked, indices_unmasked,n_sample, n_token, mask_perc):
    
    mask = np.random.default_rng().choice([True, False], size=(1, n_token), p=[mask_perc, 1 - mask_perc])
    masked = unmasked.clone()
    masked[mask] = 0  # Assuming 0 is the mask token
    indices_masked = indices_unmasked.clone()
    #indices_masked[~mask[0]] = -100 # Assuming -100 is the mask label token
   
    return masked, indices_masked, mask[0][0]


def main(args):
    torch.cuda.set_device(2)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.distributed = dist.get_world_size() > 1

    indices = np.load('/local/altamabp/test_latent_space_vqvae_80x80_codebook_64x456.npy')
    n, h, w = indices.shape
    indices = indices.reshape(n, h * w)

    quantizes = np.load('/local/altamabp/test_codebook_vqvae_80x80_codebook_64x456.npy')
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
            hidden_size=d_embed_vec,
            sinusoidal_pos_embds=False,
            n_layers=6,
            n_heads=5,
            max_position_embeddings=n_token
    )
    model_distil = DistilBertForMaskedLM(cfg).to(device)
    model_distil.load_state_dict(torch.load(args.ckpt_distil))
    model_distil = model_distil.to(device)
    model_distil.eval()

    #Define VQVAE model
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()

    # Define classifier and load saved model(weights)
    classifier = resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 10)  
    
    classifier.load_state_dict(torch.load(args.ckpt_resnet50))
    classifier.to(device)
    classifier.eval()

    mask_percentages = np.arange(0.1, 1.1, 0.1)
    mask_percentages = np.append(mask_percentages,[.85,.95])
    mask_percentages = np.sort(mask_percentages)

    average_errors = []

    for perc in mask_percentages:
        reconstruction_errors = []
        cross_entropy_class_err = []

        criterion = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()
        correct_random_pred = 0
        tot_sample = 0
        for x in range(0,quantizes.shape[0]):
            if x%1000 ==0:
                print(x)
                q = torch.from_numpy(quantizes[x])
                index = torch.from_numpy(indices[x])
                index = index.to(device)
                q = q.to(device)
                q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
                
                with torch.no_grad():
                    q_masked, index_masked, mask = random_mask(q, index , n_sample, n_token,perc)                                
                    q_masked = q_masked.to(device)
                    index_masked = index_masked.to(device)
    
                #Fill in predicted tokens
                with torch.no_grad():
                    outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                    confidence_based_prediction = torch.argmax(outputs.logits, dim=2)
                    confidence_based_recons_index = index
                    print(mask.shape)
                    for p in range(0,n_token):
                        if(mask[p]):
                            #confidence_based_recons_index[p] = confidence_based_prediction.detach().cpu().numpy()[0][p] 
                            confidence_based_recons_index[p] = confidence_based_prediction[0][p] 
                    
                    #Reconstruct with distil predictions
                    confidence_based_recons_index = confidence_based_recons_index.to(device)
                    distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))
    
                    #Reconstruct Original
                    vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
                    index_masked_forvis = index.clone()
                    index_masked_forvis[mask]=0
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
                    print(f'rand_mask_img_label is {rand_mask_img_label.item()}')
                    print(f'vqvae_img_label is {vqvae_img_label.item()}')
                    print(f'correct_random_pred is {correct_random_pred}')
                    tot_sample += 1
                    print(tot_sample)
    
    
                    #if x%1000 ==0:
                    utils.save_image(
                        torch.cat([vqvae_out, vqvae_masked_out, distil_out], 0).to(device),
                        f"image_correct/recons/random/80x80_random_{vqvae_img_label.item()}_{rand_mask_img_label.item()}_{int(perc*100)}_{str(x).zfill(5)}.png",
                        nrow=3,
                        normalize=True,
                        range=(-1, 1),
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
    plot_path = 'image/recons/random_error_vs_precision.png'
    plt.savefig(plot_path)
    plt.close()
    run['random_error_vs_percentage'].upload(plot_path)

batchsize_modified=1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=batchsize_modified)#
    parser.add_argument('--ckpt_vqvae', type=str, default="/local/altamabp/checkpoint_correct/vqvae/model_epoch100_flat_vqvae80x80_64x400codebook.pth")
    parser.add_argument('--ckpt_distil', type=str, default="/local/altamabp/checkpoint_correct/distil/80x80_100_UTKFace_flat_144x400codebook_50mask_epoch100-.pt")
    parser.add_argument('--ckpt_resnet50', type=str, default="/local/altamabp/checkpoint/classifier/weights_epoch100_fullTrain.pth")
    args = parser.parse_args()
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
