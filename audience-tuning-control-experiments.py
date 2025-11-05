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
import torch.nn.functional as F
from datetime import datetime

run = neptune.init_run(
    project="UTKFaces",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
    capture_stdout=False,
    capture_stderr=False,
    #with_id="distil",
    source_files=["audience-tuning-control_experiments.py"]
)


date=datetime.today().strftime('%Y-%m-%d')
   
os.makedirs(date+'_figures', exist_ok=True)


# =============================================================================
# def attach_bias_mask_feat(bias, masked_quantizes, mask):
#     device = masked_quantizes.device
#     dtype  = masked_quantizes.dtype
#     N, T, _ = masked_quantizes.shape
#     
# 
#     bias_t = torch.as_tensor(bias, device=device)
#  
#     # ---- Build bias features to shape (N, T, 10) ----    
#     if bias_t.dim()==0:
#         idx_t = torch.as_tensor(bias, dtype=torch.long, device=masked_quantizes.device)  # (N,)
#         idx_t_exp = idx_t.unsqueeze(1).expand(N, T)                         # (N, T)
#     
#         one_hot_labels = torch.nn.functional.one_hot(idx_t_exp, num_classes=10).to(dtype=masked_quantizes.dtype)  # (N,T,C)
#     
#         masked_exp_quantizes = torch.cat([masked_quantizes, one_hot_labels], dim=2)
#     else:
#         masked_exp_quantizes = torch.cat([masked_quantizes, bias_t], dim=2)
# 
# 
#     print('masked_exp_quantizes.shape: ',masked_exp_quantizes.shape)  # torch.Size([160000, 400, 74])
#     # masked_exp_train_quantizes: torch.FloatTensor (N, T, 74)
#     # mask_train: numpy bool array (N, T)  True = masked
# 
#     device = masked_exp_quantizes.device
#     dtype  = masked_exp_quantizes.dtype
#     
#    # 1) NumPy -> Torch, cast to float (1.0 masked, 0.0 unmasked)
#     mask_feat = torch.as_tensor(mask, device=device).to(dtype)   # (N, T)
# 
#     # 2) Add feature axis
#     mask_feat = mask_feat.unsqueeze(-1)                                 # (N, T, 1)
# 
#     # 3) Concatenate
#     masked_exp_mask_feat_quantizes = torch.cat([masked_exp_quantizes, mask_feat], dim=-1)  # (N, T, 75)
#    
#     return masked_exp_mask_feat_quantizes
# =============================================================================


def attach_bias_mask_feat(bias, masked_quantizes, mask):
    device = masked_quantizes.device
    dtype  = masked_quantizes.dtype
    N, T, _ = masked_quantizes.shape  # (N, 400, 64)

    # ---- Build bias features to shape (N, T, 10) ----
    bias_t = torch.as_tensor(bias, device=device)

    if bias_t.dim() == 0:
        # scalar -> class id for all samples
        idx = bias_t.long().expand(N)                 # (N,)
        idx_exp = idx.unsqueeze(1).expand(N, T)       # (N, T)
        bias_feat = F.one_hot(idx_exp, num_classes=10).to(dtype)  # (N, T, 10)
    else:
        # (10,) prob vector -> shared across all N,T
        assert bias_t.numel() == 10, "bias vector must have length 10"
        probs = bias_t.to(dtype).view(1, 1, 10)       # (1,1,10)
        bias_feat = probs.expand(N, T, 10).contiguous()

    # Concatenate quantized vectors with bias feature along last dim
    masked_exp_quantizes = torch.cat([masked_quantizes, bias_feat], dim=2)  # (N, T, 64+10)

    mask_t = torch.as_tensor(mask, device=device)        # bool or 0/1
    mask_t = mask_t.expand(N, T)

    # ---- Mask feature: 1.0 masked, 0.0 unmasked (broadcast to last dim=1) ----  # (N, T, 1)
    mask_feat = mask_t.to(dtype).unsqueeze(-1)           # (N, T, 1); 1.0 masked, 0.0 unmasked

    # Final concat: (N, T, 64 + 10 + 1) = (N, T, 75)
    masked_exp_mask_feat_quantizes = torch.cat([masked_exp_quantizes, mask_feat], dim=-1)

    # print('masked_exp_quantizes.shape:', masked_exp_quantizes.shape)
    # print('out.shape:', out.shape)

    return masked_exp_mask_feat_quantizes


def random_mask(unmasked, indices_unmasked,n_sample, n_token, mask_perc):
    
    mask = np.random.default_rng().choice([True, False], size=(1, n_token), p=[mask_perc, 1 - mask_perc])
    masked = unmasked.clone()
    masked[mask] = 0  # Assuming 0 is the mask token
    indices_masked = indices_unmasked.clone()
    #indices_masked[~mask[0]] = -100 # Assuming -100 is the mask label token
   
    return masked, indices_masked, mask[0][0]



############ Data prepration and masking ############
def mask_quantizes(quantizes, mask_perc, mask_token =0):
    n_samples = quantizes.shape[0]
    n_tokens = quantizes.shape[1]
    
    mask = np.random.default_rng().choice([True, False], size=(n_samples, n_tokens), p=[mask_perc, 1 - mask_perc])
    run["data/mask_prec"].log(mask_perc)

    masked_quantizes = np.copy(quantizes)
    masked_quantizes[mask] = mask_token

    masked_quantizes = torch.from_numpy(masked_quantizes)
        
    return masked_quantizes, mask

def recons(distil_model, quantizes, mask_perc, labels):
    
    
    masked_quantizes, mask= mask_quantizes(quantizes, mask_perc)

    
    masked_quantizes_exp_bias_mask_feat = attach_bias_mask_feat (labels, masked_quantizes, mask)
    print('masked_quantizes_exp_bias_mask_feat.shape: ',masked_quantizes_exp_bias_mask_feat.shape)
    
    outputs = distil_model(inputs_embeds = masked_quantizes_exp_bias_mask_feat, output_hidden_states = False)
    logits=outputs.logits
    confidence_based_prediction = torch.argmax(logits, dim=2)
    
    return confidence_based_prediction, logits, masked_quantizes, masked_quantizes_exp_bias_mask_feat


def retrieve(vqvae_model, most_probable):
    device = most_probable.device
    priors = np.reshape(most_probable, (-1,20,20))
    generated = vqvae_model.decode_code(torch.from_numpy(priors).to(device))
    zq= vqvae_model.quantize_b.embed_code(torch.LongTensor(most_probable))
    return generated, zq


def merge_traces_modified(Zq_C1I_list, Zq_C2I_list, ratio1, ratio2):
    """
    PyTorch version of merge_traces_modified with identical behavior/IO.
    - Inputs: lists of 49x64 tensors
    - Masks: zero out ratio1 of rows in C1I and ratio2 of rows in C2I (non-overlapping)
    - Output: tensor of shape (num_samples, 49, 64) with element-wise average of the three arrays
    """
    num_samples = len(Zq_C1I_list)
    height, width = Zq_C1I_list[0].shape  # (49, 64)

    num_rows_C1I = int(ratio1 * height)
    num_rows_C2I = int(ratio2 * height)

    averaged_results = []

    for i in range(num_samples):
        Zq_C1I = Zq_C1I_list[i]
        Zq_C2I = Zq_C2I_list[i]

        device = Zq_C1I.device
        dtype  = Zq_C1I.dtype

        # zero array
        zero_array = torch.zeros((height, width), device=device, dtype=dtype)

        # random row indices
        all_row_indices = torch.randperm(height, device=device)

        # select rows for C1I
        rows_C1I = all_row_indices[:num_rows_C1I]

        # remaining rows (no overlap), reshuffle
        remaining_rows = all_row_indices[num_rows_C1I:]
        if remaining_rows.numel() > 0:
            perm = torch.randperm(remaining_rows.numel(), device=device)
            remaining_rows = remaining_rows[perm]
        rows_C2I = remaining_rows[:num_rows_C2I]

        # copies
        modified_C1I = Zq_C1I.clone()
        modified_C2I = Zq_C2I.clone()

        # zero selected rows
        if rows_C1I.numel() > 0:
            modified_C1I[rows_C1I, :] = 0
        if rows_C2I.numel() > 0:
            modified_C2I[rows_C2I, :] = 0

        # average
        averaged_image = (modified_C1I + modified_C2I + zero_array) / 3.0
        averaged_results.append(averaged_image)

    return torch.stack(averaged_results, dim=0)

    
def mse(img1, img2):
    """
    Compute Mean Squared Error between two images or batches of images.
    Each input can be shape (3,80,80) or (N,3,80,80).
    Returns a scalar tensor (the mean MSE).
    """
    # Ensure both are tensors on the same device and dtype
    img1 = torch.as_tensor(img1)
    img2 = torch.as_tensor(img2, device=img1.device, dtype=img1.dtype)

    # If single images, add batch dimension
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Compute mean squared error
    mse_val = F.mse_loss(img1, img2, reduction='mean')
    return mse_val

def plot_valence(final5_all,final5_all_std):
    # Data values
    categories = ['Positive', 'Negative', 'Positive', 'Negative']
    phases = ['Message', 'Message', 'Recall', 'Recall']
    shared_reality = final5_all[0]
    non_shared_reality = final5_all[1]
    
    # Standard deviation (for error bars)
    error_shared = final5_all_std[0]
    error_non_shared = final5_all_std[1]
    
    # Define bar positions
    x = np.arange(len(categories))  # [0, 1, 2, 3]
    width = 0.3  # Width of bars
    
    # Create the figure and axis
    fig_valence, ax = plt.subplots(figsize=(8, 4))
    
    # Plot bars
    bars1 = ax.bar(x - width/2, shared_reality, width, label="Shared reality", yerr=error_shared, capsize=5, color='C0', zorder=2)
    bars2 = ax.bar(x + width/2, non_shared_reality, width, label="Non-shared reality", yerr=error_non_shared, capsize=5, color='C1', zorder=2)
    
    # Formatting the x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    
    # Adding hierarchical x-axis labels
    ax.set_xlabel("")
    ax.set_title("Valence - Simulation results")


    # Add legend
    ax.legend()
    
    
    # Create a two-level x-axis (Message vs Recall)
    ax.text(0.5, -4.2, "Message", ha="center", va="center", fontsize=10)
    ax.text(2.5, -4.2, "Recall", ha="center", va="center", fontsize=10)
    
    ax.plot([0, 1], [-3.9, -3.9], color="black", lw=1)
    ax.plot([2, 3], [-3.9, -3.9], color="black", lw=1)
    ax.set_ylim(-5,5)
    ax.set_ylabel('Valence')
    ax.yaxis.grid(True,  alpha=0.7,zorder=0)  # Dashed lines with transparency
    ax.grid(b=True, axis='y')


    # Show the plot
    #fig_valence.suptitle(r"$\bf{Valence}$", fontsize='16')#, y=1.05)
    plt.tight_layout()
    plt.savefig('./' + date + '_figures/valence_simulation.png', bbox_inches='tight')
    plt.show()




#plot accuracy
def plot_accuracy(final_acc_all, final_acc_all_std):
    
    categories = ['Message', 'Recall']
    shared_reality_acc_pos= 100*(0.09-((final_acc_all[0,0]+final_acc_all[0,1])/2))
    shared_reality_acc_neg= 100*(0.09-((final_acc_all[0,2]+final_acc_all[0,3])/2))
    shared_reality = [shared_reality_acc_pos, shared_reality_acc_neg]
    
    non_shared_reality_acc_pos= 100*(0.09-((final_acc_all[1,0]+final_acc_all[1,1])/2))
    non_shared_reality_acc_neg= 100*(0.09-((final_acc_all[1,2]+final_acc_all[1,3])/2))    
    non_shared_reality = [non_shared_reality_acc_pos,  non_shared_reality_acc_neg]

    # Standard deviation (for error bars)
    err_shared_reality_acc_pos= 100*((final_acc_all_std[0,0]+final_acc_all_std[0,1]))
    err_shared_reality_acc_neg= 100*((final_acc_all_std[0,2]+final_acc_all_std[0,3]))
    error_shared = [err_shared_reality_acc_pos,err_shared_reality_acc_neg]

    err_non_shared_reality_acc_pos= 100*((final_acc_all_std[1,0]+final_acc_all_std[1,1]))
    err_non_shared_reality_acc_neg= 100*((final_acc_all_std[1,2]+final_acc_all_std[1,3]))
    error_non_shared =[err_non_shared_reality_acc_pos,err_non_shared_reality_acc_neg] 
            
    
    # Define bar positions
    x = np.arange(len(categories))  # [0, 1]
    width = 0.25  # Width of bars
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot bars
    bars1 = ax.bar(x - width/2, shared_reality, width, label="shared reality", yerr=error_shared, capsize=5, color='C0', zorder=2)
    bars2 = ax.bar(x + width/2, non_shared_reality, width, label="No shared reality", yerr=error_non_shared, capsize=5, color='C1', zorder=2)
    
    # Formatting the x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0,12)
    ax.set_xlim(-0.5,1.5)
    # Add horizontal grid lines
    ax.yaxis.grid(True,  alpha=0.7, zorder=0)
    
    # Set title
    ax.set_title('Recall - Simulation results')
    
    # Add legend
    ax.legend()
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('./' + date + '_figures/recall_simulation.png', bbox_inches='tight')
    plt.show()



def main(args):
    torch.cuda.set_device(2)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.distributed = dist.get_world_size() > 1


    #Define VQVAE model
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()


    preprocess = transforms.Compose(
        [
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # dataset/loader
    dataset = datasets.ImageFolder("./UTKFace_dataset_subset_15000_structured", transform=preprocess)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)#, num_workers=4, pin_memory=True)
    
    images=[]
    quantizes=[]
    indices=[]
    true_labels=[]


    for imgs, labels in loader:
        quants, _, idxs, _, _ = model_vqvae.encode(imgs)
        images.append(imgs)
        quantizes.append(quants)
        indices.append(idxs)
        true_labels.append(labels)
        
        

    n, h, w = indices.shape
    indices = indices.reshape(n, h * w)

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

    n_samples = quantizes.shape[0]
    d_embed_vec = quantizes.shape[2]
    n_tokens = quantizes.shape[1]
    print(f'n_samples: {n_samples}')
    print(f'quantizes.shape: {quantizes.shape}')
    print(f'n_tokens: {n_tokens}')


    indices = set(indices.flatten())
    indices = sorted(indices)
    vocab_size = indices[-1] + 1
    


    labels = torch.from_numpy(labels)




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
    model_distil.load_state_dict(torch.load(args.ckpt_distil))
    model_distil = model_distil.to(device)
    model_distil.eval()



    # Define classifier and load saved model(weights)
    classifier = resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 10)  
    
    classifier.load_state_dict(torch.load(args.ckpt_resnet50))
    classifier.to(device)
    classifier.eval()


####----------------------------------Message production phase------------------------------####

    ####----shared reality----####
    
    #get predicted indices
    S2_s, _, _, _=recons(model_distil, quantizes, 0.5, labels[:,1])
    S2_w, _, _, _=recons(model_distil, quantizes, 0.5, labels[:,0])
    
    #retrieve image from predicted indices
    M2_s, Zq_S2_s=retrieve(model_vqvae, S2_s)
    M2_w, Zq_S2_w=retrieve(model_vqvae, S2_w)
    
    #calculate mse between retrieved image and original input image
    mse2_s=mse(M2_s,images)
    mse2_w=mse(M2_w,images)
    #(mse between retrieved image from the strong label vs original image)
    mse2_s_all=np.zeros((n_sample))
    for i in range(n_sample):
        mse2_s_all[i]=mse(M2_s[i], images[i])
    #(mse between retrieved image from the weak label vs original image)        
    mse2_w_all=np.zeros((n_sample))
    for i in range(n_sample):
        mse2_w_all[i]=mse(M2_w[i], images[i])
    
    #get std of MSE
    mse2_s_std=np.std(mse2_s_all)
    mse2_w_std=np.std(mse2_w_all)
    
    #classify retrieved image
    J2_s=classifier(M2_s)
    J2_w=classifier(M2_w)
    
    #get predicted label for retrieved image
    eval2_s=np.argmax(J2_s,axis=1)
    eval2_w=np.argmax(J2_w,axis=1)
    
    #calculate the mean of misclassifications
    error2s= (eval2_s != labels[:n_sample,1]).mean()
    error2w= (eval2_w != labels[:n_sample,0]).mean()
    
    #calculate the difference of judgements between strong and weak labels (assuming that the V2_s probs maintains the correct position of class probability for the correct original strong image label and weak original label)
    er2s_all=np.zeros((n_sample))
    for i in range(n_sample):
        er2s_all[i]=J2_s[i,labels[i,1]]-J2_s[i,labels[i,0]]
    er2s=np.mean(er2s_all)
    
    #calculate std of error in valence (strong)
    er2s_std=np.std(er2s_all)
    #er2s_std=er2s_std/2
    
    #calculate std of error in valence (weak)
    er2w_all=np.zeros((n_sample))
    for i in range(n_sample):
        er2w_all[i]=J2_w[i,labels[i,1]]-J2_w[i,labels[i,0]]
    er2w=np.mean(er2w_all)
    
    er2w_std=np.std(er2w_all)



    ####----no shared reality----####

    J2_sn, _, _, _= recons(model_distil, quantizes, 0.6, labels[:,1])
    J2_wn, _, _, _= recons(model_distil, quantizes, 0.6, labels[:,0])
    
    
    M2_sn, Zq_S2_sn= retrieve(model_vqvae, J2_sn)
    M2_wn, Zq_S2_wn= retrieve(model_vqvae, J2_wn)
    
    mse2_sn=mse(M2_sn,images)
    mse2_wn=mse(M2_wn,images)
    
    
    mse2_sn_all=np.zeros((n_sample))
    for i in range(n_sample):
        mse2_sn_all[i]=mse(M2_sn[i], images[i])
    
    
    mse2_wn_all=np.zeros((n_sample))
    for i in range(n_sample):
        mse2_wn_all[i]=mse(M2_wn[i], images[i])
    
    mse2_sn_std=np.std(mse2_sn_all)
    mse2_wn_std=np.std(mse2_wn_all)
    
    
    J2_sn=classifier(M2_sn)
    J2_wn=classifier(M2_wn)
    eval2_sn=np.argmax(J2_sn,axis=1)
    eval2_wn=np.argmax(J2_wn,axis=1)
    
    error2sn= (eval2_sn != labels[:n_sample,1]).mean()
    error2wn= (eval2_wn != labels[:n_sample,0]).mean()
    
    er2sn_all=np.zeros((n_sample))
    for i in range(n_sample):
        er2sn_all[i]=J2_sn[i,labels[i,1]]-J2_sn[i,labels[i,0]]
    er2sn=np.mean(er2sn_all)
    
    er2sn_std=np.std(er2sn_all)
    #er2sn_std=er2sn_std/2

    er2wn_all=np.zeros((n_sample))
    for i in range(n_sample):
        er2wn_all[i]=J2_wn[i,labels[i,1]]-J2_wn[i,labels[i,0]]
    er2wn=np.mean(er2wn_all)
    
    er2wn_std=np.std(er2wn_all)
    #er2wn_std=er2wn_std/2
    

#####-----------------------------Free recall phase----------------------------#####


    ####----shared reality----####
    
    
    #retrieve masked quantizes
    masked_quantizes=mask_quantizes(quantizes, 0.5)
    J1=classifier(images)
    
    zq_S1_S2_s=merge_traces_modified(masked_quantizes,Zq_S2_s, ratio1=0.83, ratio2=0.77)
    zq_S1_S2_w=merge_traces_modified(masked_quantizes,Zq_S2_w,ratio1=0.83, ratio2=0.77)
    
    
    
    #take weighted average of the probalistic biases
    JC2_JC1_prob_s=[]
    for i in range(n_sample):
        JC2_JC1_s=np.vstack(( [0.8*J2_s[i], 0.2*J1[i]]))
        JC2_JC1_prob_s.append(np.mean(JC2_JC1_s, axis=0))
        
    JC2_JC1_prob_s=np.array(JC2_JC1_prob_s)
    
    
    JC2_JC1_prob_w=[]
    for i in range(n_sample):
        JC2_JC1_w=np.vstack(( [0.8*J2_w[i], 0.2*J1[i]]))
        JC2_JC1_prob_w.append(np.mean(JC2_JC1_w, axis=0))
        
    JC2_JC1_prob_w=np.array(JC2_JC1_prob_w)
    
    
    
    #get strong bias
    j1_j2_s=np.argpartition(JC2_JC1_prob_s,-2, axis=1)[:,-1]
    #get weak bias
    j1_j2_w=np.argpartition(JC2_JC1_prob_w,-2, axis=1)[:,-1]



    S3_s, _, _, _=recons(model_distil, zq_S1_S2_s, 0.5, JC2_JC1_prob_s)
    S3_w, _, _, _=recons(model_distil, zq_S1_S2_w, 0.5, JC2_JC1_prob_w)
     
    #retrieve image from predicted indices
    M3_s, Zq_S3_s=retrieve(model_vqvae, S3_s)
    M3_w, Zq_S3_w=retrieve(model_vqvae, S3_w)
    
    #calculate mse between retrieved image and original input image
    mse3_s=mse(M3_s,images)
    mse3_w=mse(M3_w,images)
    #(mse between retrieved image from the strong label vs original image)
    mse3_s_all=np.zeros((n_sample))
    for i in range(n_sample):
        mse3_s_all[i]=mse(M3_s[i], images[i])
    #(mse between retrieved image from the weak label vs original image)        
    mse3_w_all=np.zeros((n_sample))
    for i in range(n_sample):
        mse3_w_all[i]=mse(M3_w[i], images[i])
    
    #get std of MSE
    mse3_s_std=np.std(mse3_s_all)
    mse3_w_std=np.std(mse3_w_all)
    
    #classify retrieved image
    J3_s=classifier(M3_s)
    J3_w=classifier(M3_w)
    
    #get predicted label for retrieved image
    eval3_s=np.argmax(J3_s,axis=1)
    eval3_w=np.argmax(J3_w,axis=1)
    
    #calculate the mean of misclassifications
    error3s= (eval3_s != labels[:n_sample,1]).mean()
    error3w= (eval3_w != labels[:n_sample,0]).mean()
    
    #calculate the difference of judgements between strong and weak labels (assuming that the V2_s probs maintains the correct position of class probability for the correct original strong image label and weak original label)
    er3s_all=np.zeros((n_sample))
    for i in range(n_sample):
        er3s_all[i]=J3_s[i,labels[i,1]]-J3_s[i,labels[i,0]]
    er3s=np.mean(er3s_all)
    
    #calculate std of error in valence (strong)
    er3s_std=np.std(er3s_all)
    #er2s_std=er2s_std/2
    
    #calculate std of error in valence (weak)
    er3w_all=np.zeros((n_sample))
    for i in range(n_sample):
        er3w_all[i]=J3_w[i,labels[i,1]]-J3_w[i,labels[i,0]]
    er3w=np.mean(er3w_all)
    
    er3w_std=np.std(er3w_all)


    ####----no shared reality----####
    
    #take weighted average of the probalistic biases
    JC2_JC1_prob_sn=[]
    for i in range(n_sample):
        JC2_JC1_sn=np.vstack(( [0.2*J2_sn[i], 0.8*J1[i]]))
        JC2_JC1_prob_sn.append(np.mean(JC2_JC1_sn, axis=0))
        
    JC2_JC1_prob_sn=np.array(JC2_JC1_prob_sn)
    
    
    JC2_JC1_prob_wn=[]
    for i in range(n_sample):
        JC2_JC1_wn=np.vstack(( [0.2*J2_wn[i], 0.8*J1[i]]))
        JC2_JC1_prob_wn.append(np.mean(JC2_JC1_wn, axis=0))
        
    JC2_JC1_prob_wn=np.array(JC2_JC1_prob_wn)
    
    
    
    #get strong bias
    j1_j2_s=np.argpartition(JC2_JC1_prob_sn,-2, axis=1)[:,-1]
    #get weak bias
    j1_j2_w=np.argpartition(JC2_JC1_prob_wn,-2, axis=1)[:,-1]
    
    zq_S1_S2_sn=merge_traces_modified(masked_quantizes,Zq_S2_sn, ratio1=0.77, ratio2=0.83)
    zq_S1_S2_wn=merge_traces_modified(masked_quantizes,Zq_S2_wn,ratio1=0.77, ratio2=0.83)
    
    

    S3_sn, _, _, _= recons(model_distil, zq_S1_S2_sn, 0.6, JC2_JC1_prob_sn)
    S3_wn, _, _, _= recons(model_distil, zq_S1_S2_wn, 0.6, JC2_JC1_prob_wn)
    
    
    M3_sn, Zq_S3_sn= retrieve(model_vqvae, S3_sn)
    M3_wn, Zq_S3_wn= retrieve(model_vqvae, S3_wn)
    
    mse3_sn=mse(M3_sn,images)
    mse3_wn=mse(M3_wn,images)
    
    
    mse3_sn_all=np.zeros((n_sample))
    for i in range(n_sample):
        mse3_sn_all[i]=mse(M3_sn[i], images[i])
    
    
    mse3_wn_all=np.zeros((n_sample))
    for i in range(n_sample):
        mse3_wn_all[i]=mse(M3_wn[i], images[i])
    
    mse3_sn_std=np.std(mse3_sn_all)
    mse3_wn_std=np.std(mse3_wn_all)
    
    
    J3_sn=classifier(M3_sn)
    J3_wn=classifier(M3_wn)
    eval3_sn=np.argmax(J3_sn,axis=1)
    eval3_wn=np.argmax(J3_wn,axis=1)
    
    error3sn= (eval3_sn != labels[:n_sample,1]).mean()
    error3wn= (eval3_wn != labels[:n_sample,0]).mean()
    
    er3sn_all=np.zeros((n_sample))
    for i in range(n_sample):
        er3sn_all[i]=J3_sn[i,labels[i,1]]-J3_sn[i,labels[i,0]]
    er3sn=np.mean(er3sn_all)
    
    er3sn_std=np.std(er3sn_all)
    #er2sn_std=er2sn_std/2

    er3wn_all=np.zeros((n_sample))
    for i in range(n_sample):
        er3wn_all[i]=J3_wn[i,labels[i,1]]-J3_wn[i,labels[i,0]]
    er3wn=np.mean(er3wn_all)
    
    er3wn_std=np.std(er3wn_all)
    #er2wn_std=er2wn_std/2



####---------------------valence and accuracy calculations--------------------------####
    
    Final=np.array([[er2s,er2w,er3s,er3w],[er2sn,er2wn,er3sn,er3wn]])
    final5= 5*Final
    
    Final_acc=np.array([[mse2_s,mse2_w,mse3_s,mse3_w],[mse2_sn,mse2_wn,mse3_sn,mse3_wn]])
    
    final5_all_std=(5*np.array([[er2s_std,er2w_std,er3s_std,er3w_std],[er2sn_std,er2wn_std,er3sn_std,er3wn_std]]))/2
    final5_all=5*np.array([[er2s,er2w,er3s,er3w],[er2sn,er2wn,er3sn,er3wn]])
    
    final_acc_all_std=np.array([[mse2_s_std,mse2_w_std,mse3_s_std,mse3_w_std],[mse2_sn_std,mse2_wn_std,mse3_sn_std,mse3_wn_std]])
    final_acc_all=np.array([[mse2_s,mse2_w,mse3_s,mse3_w],[mse2_sn,mse2_wn,mse3_sn,mse3_wn]])
    

    
    plot_valence(final5_all,final5_all_std)
    #plot_valence([[1.2525, -0.555, 0.77, -0.3075], [2.294, -2.31, 0.198, 0.292]], [[0.8925, 0.9075, 0.475, 0.53875], [0.908, 0.963, 0.607, 0.6]])
    
    plot_accuracy(final_acc_all,final_acc_all_std)
    #acc_all_experiment=np.array([[6.2875, 7.4425, 3.655, 3.8875], [4.722, 6.798, 3.364, 3.804]])
    #plot_accuracy(acc_all_experiment)
    



    
# =============================================================================
#     preprocess = transforms.Compose(
#          [
#              transforms.Resize((80,80)),
#              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#          ]
#      )
# 
#     mask_percentages = np.arange(0.1, 1.1, 0.1)
#     mask_percentages = np.append(mask_percentages,[.85,.95])
#     mask_percentages = np.sort(mask_percentages)
# 
#     average_errors = []
# 
#     for perc in mask_percentages:
#         reconstruction_errors = []
#         cross_entropy_class_err = []
# 
#         criterion = nn.MSELoss()
#         criterion_class = nn.CrossEntropyLoss()
#         correct_random_pred = 0
#         tot_sample = 0
#         for x in range(0,quantizes.shape[0]):
#             if x%1000 ==0:
#                 print(x)
#                 q = torch.from_numpy(quantizes[x])
#                 index = torch.from_numpy(indices[x])
#                 index = index.to(device)
#                 q = q.to(device)
#                 q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
#                 
#                 with torch.no_grad():
#                     q_masked, index_masked, mask = random_mask(q, index , n_sample, n_token,perc)                                
#                     q_masked = q_masked.to(device)
#                     index_masked = index_masked.to(device)
#     
#                 #Fill in predicted tokens
#                 with torch.no_grad():
#                     outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
#                     confidence_based_prediction = torch.argmax(outputs.logits, dim=2)
#                     confidence_based_recons_index = index
#                     print(mask.shape)
#                     for p in range(0,n_token):
#                         if(mask[p]):
#                             #confidence_based_recons_index[p] = confidence_based_prediction.detach().cpu().numpy()[0][p] 
#                             confidence_based_recons_index[p] = confidence_based_prediction[0][p] 
#                     
#                     #Reconstruct with distil predictions
#                     confidence_based_recons_index = confidence_based_recons_index.to(device)
#                     distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))
#     
#                     #Reconstruct Original
#                     vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
#                     index_masked_forvis = index.clone()
#                     index_masked_forvis[mask]=0
#                     vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))
#     
#                     # Label outputs
#                     vqvae_out = vqvae_out.unsqueeze(0)
#                     vqvae_img = preprocess(vqvae_out)
#                     vqvae_img = vqvae_img.to(device)
#                     vqvae_img_prob = classifier(vqvae_img)
#                     _, vqvae_img_label = torch.max(vqvae_img_prob, 1)
#                     
#                     rand_mask_img = preprocess(distil_out)
#                     rand_mask_img = rand_mask_img.to(device)
#                     rand_mask_img_prob = classifier(rand_mask_img)
#                     _, rand_mask_img_label = torch.max(rand_mask_img_prob, 1)
#                     correct_random_pred += (rand_mask_img_label == vqvae_img_label).sum().item()
#                     print(f'rand_mask_img_label is {rand_mask_img_label}')
#                     print(f'vqvae_img_label is {vqvae_img_label}')
#                     print(f'rand_mask_img_label is {rand_mask_img_label.item()}')
#                     print(f'vqvae_img_label is {vqvae_img_label.item()}')
#                     print(f'correct_random_pred is {correct_random_pred}')
#                     tot_sample += 1
#                     print(tot_sample)
#     
#     
#                     #if x%1000 ==0:
#                     utils.save_image(
#                         torch.cat([vqvae_out, vqvae_masked_out, distil_out], 0).to(device),
#                         f"image_correct/recons/random/80x80_random_{vqvae_img_label.item()}_{rand_mask_img_label.item()}_{int(perc*100)}_{str(x).zfill(5)}.png",
#                         nrow=3,
#                         normalize=True,
#                         range=(-1, 1),
#                     )
#                 
#                 recon_loss = criterion(distil_out, vqvae_out)
#                 run["recons/mse_per_image_random_mask"].log(recon_loss.item())
#                 reconstruction_errors.append(recon_loss.item())
#                 class_loss = criterion_class(rand_mask_img_prob, vqvae_img_prob)
#                 run["recons/cross_entropy_per_image_random_mask"].log(class_loss.item())
#                 cross_entropy_class_err.append(class_loss.item())
#         
#         run["recons/average_mse_per_precision_random_mask"].log(np.mean(reconstruction_errors))
#         run["recons/average_cross_entropy_error_random_mask"].log(np.mean(cross_entropy_class_err))
#         average_errors.append(np.mean(reconstruction_errors))
#         pred_acc_random_mask = correct_random_pred/tot_sample
#         pred_err_random_mask = 1-pred_acc_random_mask
#         run["recons/average_classification_accuracy_random"].log(pred_err_random_mask)
# 
# 
#     # Plotting the reconstruction errors
#     plt.plot(mask_percentages * 100, average_errors, marker='o')
#     plt.xlabel('Mask Percentage')
#     plt.ylabel('Average Reconstruction Error (MSE)')
#     plt.title('Reconstruction Error for Random Mask vs Mask Percentage')
#     plt.grid(True)
#     plot_path = 'image/recons/random_error_vs_precision.png'
#     plt.savefig(plot_path)
#     plt.close()
#     run['random_error_vs_percentage'].upload(plot_path)
# 
# =============================================================================
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
    parser.add_argument('--ckpt_resnet50', type=str, default="/local/altamabp/checkpoint_correct/classifier/weights_epoch50.pth")
    args = parser.parse_args()
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
