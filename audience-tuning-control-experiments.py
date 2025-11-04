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
    priors = np.reshape(most_probable, (-1,20,20))
    zq = vqvae_model.decode_code(priors)
    generated = vqvae_model.decode(torch.from_numpy(zq).to(device))
    return generated 

    
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




####-------------------------shared reality---------------------------####

#get predicted indices
S2_s, _, _, _=recons(model_distil, quantizes, 0.5, labels[:,1])
S2_w, _, _, _=recons(model_distil, quantizes, 0.5, labels[:,0])

#retrieve image from predicted indices
M2_s=retrieve(model_vqvae, S2_s)
M2_w=retrieve(model_vqvae, S2_w)

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
V2_s=classifier(M2_s)
V2_w=classifier(M2_w)

#get predicted label for retrieved image
eval2_s=np.argmax(V2_s,axis=1)
eval2_w=np.argmax(V2_w,axis=1)

#calculate the mean of misclassifications
error2s= (eval2_s != labels[:n_sample,1]).mean()
error2w= (eval2_w != labels[:n_sample,0]).mean()

#calculate the difference of judgements between strong and weak labels (assuming that the V2_s probs maintains the correct position of class probability for the correct original strong image label and weak original label)
er2s_all=np.zeros((n_sample))
for i in range(n_sample):
    er2s_all[i]=V2_s[i,labels[i,1]]-V2_s[i,labels[i,0]]
er2s=np.mean(er2s_all)

#calculate std of error in valence (strong)
er2s_std=np.std(er2s_all)
#er2s_std=er2s_std/2

#calculate std of error in valence (weak)
er2w_all=np.zeros((n_sample))
for i in range(n_sample):
    er2w_all[i]=V2_w[i,labels[i,1]]-V2_w[i,labels[i,0]]
er2w=np.mean(er2w_all)

er2w_std=np.std(er2w_all)
















    
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
    parser.add_argument('--ckpt_resnet50', type=str, default="/local/altamabp/checkpoint_correct/classifier/weights_epoch50.pth")
    args = parser.parse_args()
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
