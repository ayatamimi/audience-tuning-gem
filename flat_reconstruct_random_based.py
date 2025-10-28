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
    project="tns/Vqvae-transformer",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODg4OTU0Yy0xODAyLTRiM2QtYjYzYi0xMWQxYThmYWJlOWQifQ==",
    capture_stdout = False,
    capture_stderr = False,
    # with_id="MAS-389"
)

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
    indices_masked[~mask[0]] = -100
    return quantized_masked,indices_masked, mask[0]

def random_mask(unmasked, indices_unmasked,n_sample, n_token, mask_perc):
    
    mask = np.random.default_rng().choice([True, False], size=(1,1, n_token), p=[mask_perc, 1 - mask_perc])
    masked = unmasked.clone()
    masked[mask] = 0  # Assuming 0 is the mask token
    indices_masked = indices_unmasked.clone()
    indices_masked[~mask[0]] = -100 # Assuming -100 is the mask label token
   
    return masked, indices_masked, mask[0][0]

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
    indices_masked[~mask] = -100 # Assuming -100 is the mask label token
    indices_masked = torch.from_numpy(indices_masked)

    return masked, indices_masked, mask

def main(args):
    torch.cuda.set_device(2)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.distributed = dist.get_world_size() > 1

    indices = np.load('/home/abghamtm/work/masking_comparison/checkpoint/vqvae/indices_epoch80_flat_vqvae80x80_144x456codebook.npy')
    n, h, w = indices.shape
    indices = indices.reshape(n, h * w)

    quantizes = np.load('/home/abghamtm/work/masking_comparison/checkpoint/vqvae/quantized_epoch80_flat_vqvae80x80_144x456codebook.npy')
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
            n_heads=4,
            max_position_embeddings=n_token
    )
    model_distil = DistilBertForMaskedLM(cfg).to(device)
    model_distil.load_state_dict(torch.load(args.ckpt_distil_combined))
    model_distil = model_distil.to(device)
    model_distil.eval()

    #Define VQVAE model
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()

    # Define classifier and load saved model(weights)
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    classifier = resnet50(pretrained=False)
    classifier.load_state_dict(torch.load('/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/weights_epoch30.pth'))
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


                if x%5 ==0:
                    utils.save_image(
                        torch.cat([vqvae_out, vqvae_masked_out, distil_out], 0).to(device),
                        f"image/recons/random/80x80_random_{vqvae_img_label.item()}_{rand_mask_img_label.item()}_{int(perc*100)}_{str(x).zfill(5)}.png",
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
    parser.add_argument("--batch_size", type=int, default=batchsize_modified)#64
    parser.add_argument('--ckpt_vqvae', type=str, default="/home/abghamtm/work/masking_comparison/checkpoint/vqvae/model_epoch80_flat_vqvae80x80_144x456codebook.pth")
    parser.add_argument('--ckpt_distil_combined', type=str, default="/home/abghamtm/work/masking_comparison/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt")

    args = parser.parse_args()
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

