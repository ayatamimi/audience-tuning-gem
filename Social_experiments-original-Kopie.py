# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:26:47 2022

@author: ZahraFayyaz
"""


from pathlib import Path

import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForMaskedLM, DistilBertConfig, AdamWeightDecay

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import backend as Kb
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_absolute_error

from datetime import datetime
import os 

date=datetime.today().strftime('%Y-%m-%d')
   
os.makedirs(date+'_figures', exist_ok=True)

def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train[..., None] / 255., y_train), (x_test[..., None] / 255., y_test)

(x_train, y_train), (x_test, y_test) = load_data('Data/mnist.npz')



# Hyperparameters
NUM_LATENT_K = 20                 # Number of codebook entries
NUM_LATENT_D = 64                 # Dimension of each codebook entries
BETA = 1.0                        # Weight for the commitment loss



INPUT_SHAPE = (28,28,1)
SIZE = None                       # Spatial size of latent embedding
                                  # will be set dynamically in `build_vqvae

VQVAE_BATCH_SIZE = 128            # Batch size for training the VQVAE
VQVAE_NUM_EPOCHS = 20             # Number of epochs
VQVAE_LEARNING_RATE = 3e-4        # Learning rate
VQVAE_LAYERS = [16, 32]           # Number of filters for each layer in the encoder



# # Building the generative model
# 
# The first step is to build the main VQ-VAE model. It consists of a standard encoder-decoder architecture with convolutional blocks. The main novelty lies in the intermediate **Vector Quantizer** layer (`VQ`) that takes care of building a **discrete** latent space.
# 
# More specifically, the encoder, `f` is a fully-convolutional neural network that maps input images to latent codes of size $(w, h, d)$, where $d$ is the dimension of the latent space, and $w \times h$ the size of the final feature map. The output of the encoder is then mapped to the closest entry in a discrete **codebook** of $K$ latent codes, $\mathcal E = \{e_0 \dots e_{K-1} \}$ where $\forall i, e_i \in \mathbb{R}^d$.
# 
# \begin{align}
# &\textbf{input }x \tag{W x H x C}\\
# z_e &= f(x) \tag{w x h x d}\\
# z_q^{i, j} &= \arg\min_{e \in \mathcal E} \| z_e^{i, j} - e \|^2
# \end{align}
# 
# The Vector Quantization process is implemented as the following `Keras` Layer:

# In[7]:


class VectorQuantizer(K.layers.Layer):  
    def __init__(self, k, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.k = k
    
    def build(self, input_shape):
        self.d = int(input_shape[-1])
        rand_init = K.initializers.VarianceScaling(distribution="uniform")
        self.codebook = self.add_weight(shape=(self.k, self.d), initializer=rand_init, trainable=True)
        
    def call(self, inputs):
        # Map z_e of shape (b, w,, h, d) to indices in the codebook
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        z_e = tf.expand_dims(inputs, -2)
        dist = tf.norm(z_e - lookup_, axis=-1)
        k_index = tf.argmin(dist, axis=-1)
        return k_index
    
    def sample(self, k_index):
        # Map indices array of shape (b, w, h) to actual codebook z_q
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        k_index_one_hot = tf.one_hot(k_index, self.k)
        z_q = lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q


# The decoder, $g$, then takes the quantized codes $z_q$ as inputs and generates the output image. Here we consider a simple architecture with transposed convolution blocks, mirroring the encoder architecture:

# In[8]:


def encoder_pass(inputs, d, num_layers=[16, 32]):
    x = inputs
    for i, filters in enumerate(num_layers):
        x = K.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', activation='relu', 
                            strides=(2, 2), name="conv{}".format(i + 1))(x)
    z_e = K.layers.Conv2D(filters=d, kernel_size=3, padding='SAME', activation=None,
                          strides=(1, 1), name='z_e')(x)
    return z_e

def decoder_pass(inputs, num_layers=[32, 16]):
    y = inputs
    for i, filters in enumerate(num_layers):
        y = K.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(2, 2), padding="SAME", 
                                     activation='relu', name="convT{}".format(i + 1))(y)
    decoded = K.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), 
                                       padding="SAME", activation='sigmoid', name='output')(y)
    return decoded


# Once these three building blocks are done, we can build the full `VQ-VAE`. One subtility is how we can estimate gradient through the Vector Quantizer: In fact, the transition from $z_e$ to $z_q$ does not allow to backpropagate gradient due to the argmin function. Instead, the authors propose to use a *straight-through estimator*, that directly copies the gradient received by $z_q$ to $z_e$. 

# In[9]:


def build_vqvae(k, d, input_shape=(28, 28, 1), num_layers=[16, 32]):
    global SIZE
    ## Encoder
    encoder_inputs = K.layers.Input(shape=input_shape, name='encoder_inputs')
    z_e = encoder_pass(encoder_inputs, d, num_layers=num_layers)
    SIZE = int(z_e.get_shape()[1])

    ## Vector Quantization
    vector_quantizer = VectorQuantizer(k, name="vector_quantizer")
    codebook_indices = vector_quantizer(z_e)
    encoder = K.Model(inputs=encoder_inputs, outputs=codebook_indices, name='encoder')

    ## Decoder
    decoder_inputs = K.layers.Input(shape=(SIZE, SIZE, d), name='decoder_inputs')
    decoded = decoder_pass(decoder_inputs, num_layers=num_layers[::-1])
    decoder = K.Model(inputs=decoder_inputs, outputs=decoded, name='decoder')
    
    ## VQVAE Model (training)
    sampling_layer = K.layers.Lambda(lambda x: vector_quantizer.sample(x), name="sample_from_codebook")
    z_q = sampling_layer(codebook_indices)
    codes = tf.stack([z_e, z_q], axis=-1)
    codes = K.layers.Lambda(lambda x: x, name='latent_codes')(codes)
    straight_through = K.layers.Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]), name="straight_through_estimator")
    straight_through_zq = straight_through([z_q, z_e])
    reconstructed = decoder(straight_through_zq)
    vq_vae = K.Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')
    
    ## VQVAE model (inference)
    codebook_indices = K.layers.Input(shape=(SIZE, SIZE), name='discrete_codes', dtype=tf.int32)
    z_q = sampling_layer(codebook_indices)
    generated = decoder(z_q)
    vq_vae_sampler = K.Model(inputs=codebook_indices, outputs=generated, name='vq-vae-sampler')
    
    ## Transition from codebook indices to model (for training the prior later)
    indices = K.layers.Input(shape=(SIZE, SIZE), name='codes_sampler_inputs', dtype='int32')
    z_q = sampling_layer(indices)
    codes_sampler = K.Model(inputs=indices, outputs=z_q, name="codes_sampler")
    
    ## Getter to easily access the codebook for vizualisation
    indices = K.layers.Input(shape=(), dtype='int32')
    vector_model = K.Model(inputs=indices, outputs=vector_quantizer.sample(indices[:, None, None]), name='get_codebook')
    def get_vq_vae_codebook():
        codebook = vector_model.predict(np.arange(k))
        codebook = np.reshape(codebook, (k, d))
        return codebook
    
    return vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, get_vq_vae_codebook

vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, get_vq_vae_codebook = build_vqvae(
    NUM_LATENT_K, NUM_LATENT_D, input_shape=INPUT_SHAPE, num_layers=VQVAE_LAYERS)
vq_vae.summary()

def mse_loss(ground_truth, predictions):
    mse_loss = tf.reduce_mean((ground_truth - predictions)**2, name="mse_loss")
    return mse_loss

def latent_loss(dummy_ground_truth, outputs):
    global BETA
    del dummy_ground_truth
    z_e, z_q = tf.split(outputs, 2, axis=-1)
    vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q)**2)
    commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q))**2)
    latent_loss = tf.identity(vq_loss + BETA * commit_loss, name="latent_loss")
    return latent_loss    
#vq_vae.load_weights('vqvae_aya.h5')
#vq_vae.load_weights('Data/vqvae_trained_on_vae_data_new.h5')#('Data/vqvae_weights_20.h5')#
vq_vae.load_weights('./UTKFace/vqvae_trained_on_UTKFace-40codebooks_50epochs.h5')

###dirty mnist trained with vqvae_weights_20.h5
# 
# 
#W1 = vq_vae.layers[1].get_weights()
#w2 = vq_vae.layers[1].get_weights()



#helper function for plotting results    
def decode(codes, codes_sampler,x_test, size, n_row, n_col,title):
    #"""plot decoded codes"""
    n = n_col * n_row
    zq = codes_sampler(codes)
    decoded = decoder.predict(zq, steps=1)
    plt.figure(figsize=(15, 8))
    plt.title('Car Prices are Increasing')
    for i in range(n):
        plt.subplot(n_row, 2 * n_col, 2 * i + 1)
        plt.imshow(x_test[i], cmap='gray')
        plt.axis('off')
        plt.subplot(n_row, 2 * n_col, 2 * i + 2)
        plt.imshow(decoded[i,:,:,0], cmap='gray')
        plt.axis('off')
    plt.suptitle(('masking level=',str(title)))    
    plt.show()




#prepare data:
data= np.load('amb_data.npz')

train=data['data']
labels=data['labels']
train=train[...,None]


from keras.models import load_model
classifier= load_model('Data/MNIST_keras_CNN.h5')
v1=classifier.predict(train)
#clipping
clipping='false'
if (clipping=='true'):
    train[train<0.3]=0
    train[train>0.8]=1
    labels = np.argpartition(v1,-2, axis=1)[:,-2:]

def plotmnist(data,label,n_row=5,n_col=5,start=0):   
    n=n_col*n_row
    fig = plt.figure(figsize=(10, 12))
    for i in range(n):
        plt.subplot(n_row,n_col,i+1)
        if data.shape[-1]==28:
            plt.imshow(data[i+start,:,:], cmap='gray')
        else:
            plt.imshow(data[i+start,:,:,0], cmap='gray')
        plt.title(str(label[i+start,1]) +','+ str(label[i+start,0]))
        plt.axis('off')
    plt.show() 
    return()
plotmnist(train,labels,6,6,0)



encoded_outputs = vq_vae.predict(train)
encoded_outputs = encoded_outputs[1][...,1] #Zq

codebook_indices = encoder.predict(train)
train_X = encoded_outputs
train_y = codebook_indices


# =============================================================================
# #Creating VAE train and test data for training the latent classifier
# #vae_data=np.load('ambig_data/amb_mnist.npz')
# vae_train=np.load('Data/vae_amb_train.npy')#vae_data['train']
# vae_test=np.load('Data/vae_amb_test.npy')#vae_data['test']
# 
# vae_train=vae_train.reshape(-1,28,28,1)
# vae_test=vae_test.reshape(-1,28,28,1)
# 
# vae_train_pred=classifier.predict(vae_train)
# vae_test_pred=classifier.predict(vae_test)
# np.save('Data/vae_train_pred.npy', vae_train_pred)
# np.save('Data/vae_test_pred.npy', vae_test_pred)
# 
# vae_train_zq_ze=vq_vae.predict(vae_train)
# vae_train_zq=vae_train_zq_ze[1][...,1] #Zq
# vae_train_ze=vae_train_zq_ze[1][...,0] #Ze
# 
# vae_test_zq_ze=vq_vae.predict(vae_test)
# vae_test_zq=vae_test_zq_ze[1][...,1] #Zq
# vae_test_ze=vae_test_zq_ze[1][...,0] #Ze
# 
# np.save('Data/vae_train_zq.npy', vae_train_zq)
# np.save('Data/vae_test_zq.npy', vae_test_zq)
# np.save('Data/vae_train_ze.npy', vae_train_ze)
# np.save('Data/vae_test_ze.npy', vae_test_ze)
# =============================================================================


#distribution of codebooks by image class
# =============================================================================
# codebook_indices_zeros=encoder.predict(imgs_by_label[0][0].reshape(-1,28,28,1))
# codebook_indices_ones=encoder.predict(imgs_by_label[1][0].reshape(-1,28,28,1))
# 
# plt.figure(figsize = (10,5))
# plt.hist(codebook_indices_ones.flatten(), density=True, bins=20) #codebook indicies of mnist data
# plt.show()
# 
# =============================================================================



def plot_images(i,data,label,n_row=5,n_col=5,start=0):   
    n=n_col*n_row
    fig = plt.figure(figsize=(20, 15))
    for i in range(n):
        plt.subplot(n_row,n_col,i+1)
        if data.shape[-1]==28:
            plt.imshow(data[i+start,:,:], cmap='gray')
        else:
            plt.imshow(data[i+start,:,:,0], cmap='gray')
        plt.title(str(label[i+start,1]) +','+ str())
        label1=label[i+start,1]
        label2=label[i+start,0]

        plt.title(f"({label1},{label2})")
        plt.axis('off')
    plt.suptitle('Reiterated Zqs \n Iteration number '+str(v), fontsize=20)
    plt.show()
    fig.savefig('./' + date + '_figures/Reiterated_zqs'+str(v)+'.png')
    
reiterated_zq=np.array(encoded_outputs)

for v in range (1):
    reiterated_zq=np.array(reiterated_zq)       
    decoded_images=decoder.predict(reiterated_zq.reshape((-1,7,7,64)))
    reiterated_zq=vq_vae.predict(decoded_images.reshape((-1,28,28,1)))
    reiterated_zq = reiterated_zq[1][...,1]
    #reiterated_zq=np.array(reiterated_zq)
#    decoded_images=decoder.predict(reiterated_zq.reshape((-1,7,7,64)))
#    if (v%10==0):      
    plot_images(v,decoded_images,labels,5,10,0)



######## transformer

n_train_samples= train_X.shape[0]
d_embed_vec = train_X.shape[3]
n_tokens = np.prod(train_X.shape[1:3])

# flatten out x/y dimensions and select quantized vectors
train_X = train_X[...].reshape((n_train_samples, n_tokens, d_embed_vec))
train_y = train_y.reshape((n_train_samples, n_tokens))



# find largest codebook index for vocabulary size
indices = set(train_y.flatten())
indices = sorted(indices)
vocab_size = indices[-1] + 1

mask_perc = 0.8
mask_token = 0  # does 0 make sense?

mask_train = np.random.default_rng().choice([True, False], size=(n_train_samples, n_tokens), p=[mask_perc, 1 - mask_perc])

masked_train_X = np.copy(train_X)
masked_train_X[mask_train] = mask_token


train_onehot = 2*(tf.keras.utils.to_categorical(labels[:,0], num_classes = 10))-1
train_expanded = np.repeat(train_onehot[:, np.newaxis,:], 49, axis=1)
masked_exp_train_X = np.concatenate((train_expanded, masked_train_X ),axis=2)

d_batch = 256
n_epochs = 60
n_warmup_epochs = 50
lr = 0.001

cfg = DistilBertConfig(
    vocab_size=20,
    hidden_size=d_embed_vec+10,  #######notice the change
    num_hidden_layers=4, #change it increase it
    num_attention_heads=2, #######notice the change
    intermediate_size=2048,
    max_position_embeddings=n_tokens
)

#check token change performance and time

class LinearScheduleWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    # How the schedule looks:
    # https://huggingface.co/transformers/v3.0.2/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup
    # or look at plot in testing section
    
    def __init__(self, learning_rate, n_warmup_epochs, n_train_samples, n_epochs, d_batch):
        self.learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        self.n_warmup_epochs = tf.convert_to_tensor(n_warmup_epochs, dtype=tf.float32)
        self.n_train_samples = tf.convert_to_tensor(n_train_samples, dtype=tf.float32)
        self.n_epochs = tf.convert_to_tensor(n_epochs, dtype=tf.float32)
        self.d_batch = tf.convert_to_tensor(d_batch, dtype=tf.float32)
        
        self.steps_per_epoch = tf.convert_to_tensor(round(n_train_samples / d_batch), dtype=tf.float32)
        self.total_steps = tf.convert_to_tensor(self.steps_per_epoch * n_epochs, dtype=tf.float32)
        self.b = self.learning_rate * self.n_epochs / (self.n_epochs - self.n_warmup_epochs)
        
    def __call__(self, step):
        def true_fn():
            return self.learning_rate / self.n_warmup_epochs * step / self.steps_per_epoch
        def false_fn():
            return - self.learning_rate / (self.n_epochs - self.n_warmup_epochs) * (step / self.steps_per_epoch) + self.b
            
        ret = tf.cond(step / self.steps_per_epoch < self.n_warmup_epochs, true_fn, false_fn)
        return ret


model = TFDistilBertForMaskedLM(cfg)
lr_schedule = LinearScheduleWithWarmup(lr, n_warmup_epochs, n_train_samples, n_epochs, d_batch)
optimizer = AdamWeightDecay(learning_rate=lr_schedule)
model.compile(optimizer=optimizer)
model({'inputs_embeds': masked_exp_train_X[0, None], 'labels': train_y[0, None]})  # call model once with some input to get it built so we can do model.summary()
model.summary()


#load_status = model.load_weights('./Data/distilBert-vqvae-and-transformer-trained-on-vae-data.h5')#('Data/distilBERT_weights_dirtymnist_softlabels.h5')#distilBert-vqvae-and-transformer-trained-on-vae-data_ohe-bias.h5')#('Data/distilBERT_weights_dirtymnist_softlabels.h5')
load_status=model.load_weights('./UTKFace/distilBert-vqvae-and-transformer-trained-on-UTKFace_ohe-bias.h5')


mask_perc = 0.6
mask_token = 0  # does 0 make sense?

mask_train = np.random.default_rng().choice([True, False], size=(n_train_samples, n_tokens), p=[mask_perc, 1 - mask_perc])

masked_train_X = np.copy(train_X)
masked_train_X[mask_train] = mask_token

train_onehot = (tf.keras.utils.to_categorical(labels[:,0], num_classes = 10))
train_expanded = np.repeat(train_onehot[:, np.newaxis,:], 49, axis=1)
masked_exp_train_X = np.concatenate((train_expanded, masked_train_X ),axis=2)

n_rec = 679#100
reconstructions = model.predict({'inputs_embeds': masked_exp_train_X[:n_rec]}, batch_size=256)#, 'labels': train_y[:n_rec]}, batch_size=256)

logits = reconstructions.logits
most_probable = logits.argmax(axis=-1)

def old_recons(train_X,train_y,mask_perc,labels,n_rec):
    mask_token = 0  # does 0 make sense?
    mask_train = np.random.default_rng().choice([True, False], size=(n_rec, n_tokens), p=[mask_perc, 1 - mask_perc])
    masked_train_X = np.copy(train_X[:n_rec])
    masked_train_X[mask_train] = mask_token
    if len(labels.shape)==1:
        train_onehot = (tf.keras.utils.to_categorical(labels[:n_rec], num_classes = 10))
    else: train_onehot = labels[:n_rec]
    train_expanded = np.repeat(train_onehot[:, np.newaxis,:], 49, axis=1)
    masked_exp_train_X = np.concatenate((train_expanded, masked_train_X ),axis=2)
    reconstructions = model.predict({'inputs_embeds': masked_exp_train_X}, batch_size=256)#, 'labels': train_y[:n_rec]}, batch_size=256)    
    logits = reconstructions.logits
    most_probable = logits.argmax(axis=-1)    
    return most_probable,logits



def recons(train_X,train_y,mask_perc,prob,n_rec):
    #mask_token = 0  # does 0 make sense?
    mask_train = np.random.default_rng().choice([True, False], size=np.shape(train_X), p=[mask_perc, 1 - mask_perc])
    if( n_rec==1):
        masked_train_X = np.copy(train_X)
    else:
        masked_train_X = np.copy(train_X[:n_rec])
    masked_train_X[mask_train] = mask_token
    train_prob = prob
    if( n_rec==1):
        train_expanded = np.repeat(train_prob[np.newaxis,:], 49, axis=0)
        masked_exp_train_X = np.concatenate((train_expanded, masked_train_X ),axis=1)
        reconstructions = model.predict({'inputs_embeds': np.expand_dims(masked_exp_train_X, axis=0)}, batch_size=256)#'labels': train_y[:n_rec]}, batch_size=256)       

    else:
        train_expanded = np.repeat(train_prob[:, np.newaxis,:], 49, axis=1)
        masked_exp_train_X = np.concatenate((train_expanded, masked_train_X ),axis=2)
        reconstructions = model.predict({'inputs_embeds': masked_exp_train_X}, batch_size=256)#'labels': train_y[:n_rec]}, batch_size=256)       
    logits = reconstructions.logits
    most_probable = logits.argmax(axis=-1)    
    return most_probable,logits


def retrieve(most_probable):
    priors = np.reshape(most_probable, (-1,7,7))
    zq = codes_sampler(priors)
    generated = decoder.predict(zq, steps=1)
    return generated 
n_row = 5
n_col = 5
n = n_row * n_col
plt.figure(figsize = (20,10))
for i in range(n):
        priors = most_probable[i].reshape(7,7)
        zq = codes_sampler(priors)
        generated = decoder.predict(zq, steps=1)
        plt.subplot(n_row, 2 * n_col, 2 * i + 1)
        plt.imshow(train[i].reshape(28,28), cmap='gray')
        plt.title(str(labels[i,0]) +','+ str(labels[i,1]))
        plt.axis('off')
        plt.subplot(n_row, 2 * n_col, 2 * i + 2)
        plt.imshow(generated.reshape(28,28), cmap='gray')
        plt.title("recons", fontsize=7)
        plt.axis('off')
plt.show()

#######
#check to see if the two types of data have similar structure
# =============================================================================
# counts, bins = np.histogram(train)
# plt.stairs(bins, counts)
# =============================================================================

plt.figure(figsize = (10,5))
plt.hist(train.flatten(), density=True, bins=30) #ambiguous data
plt.title('Ambiguous data distribution')
plt.show()

plt.figure(figsize = (10,5))
plt.hist(x_train.flatten(), density=True, bins=30) #mnist data
plt.title('MNIST data distribution')
plt.show()

plt.figure(figsize = (10,5))
plt.hist(train_y.flatten(), density=True, bins=20) #codebook indicies of amb data
plt.title('Distribution of ambiguous data codebook indices')
plt.show()

#z = vq_vae.predict(x_train)
#z = z[1][...,1]
#y = encoder.predict(x_train)#[:1000])

#plt.figure(figsize = (10,5))
#plt.hist(y.flatten(), density=True, bins=20) #codebook indicies of mnist data
#plt.title('Distribution of MNIST data codebook indices')
#plt.show()


######



preds=classifier.predict(generated)
evals = np.argpartition(preds,-2, axis=1)[:,-2:]
max_indices= np.max(preds,axis=1)
second_max = np.partition(preds,-2)[:,-2]
or_weak=np.mean(second_max) #0.08
or_strong=np.mean(max_indices) #0.89

######
#step1: soft labels on input
v1=classifier.predict(train)
evals1 = np.argpartition(v1,-2, axis=1)[:,-2:]
v1softmax1= np.max(v1,axis=1)
v1softmax2 = np.partition(v1,-2)[:,-2]
or_weak=np.mean(v1softmax2) #0.41
or_strong=np.mean(v1softmax1) #0.56



# Convert each inner list to a tuple
pairs = [tuple(pair) for pair in evals1]


from collections import Counter

#def most_frequent_pair(pairs):
# Count the occurrences of each pair
counter = Counter(map(tuple, pairs))


# Find the most common pair and its frequency
most_common = counter.most_common(1)

if most_common:
   result= most_common[0]
else:
    None

# Example usage

#result = most_frequent_pair(pairs)

if result:
    print(f"The most frequent pair is {result[0]} and it occurs {result[1]} times.")
else:
    print("No pairs found.")

#Limit the ambiguous images to be the ones that are ambiguous between 1 and 4 (because 1&4 are the most cases among all the ambiguous imgaes)
#idxs= [i for i in range(len(evals1)) if (evals1[i, 0] == 4 and evals1[i, 1] == 1) or (evals1[i, 0] == 1 and evals1[i, 1] == 4)]
#train=train[idxs]


#=============================================================================================
###create ambiguous images from the transformer
#prepare the softmax labels representation for the pair that appear in ambiguous numbers (50% strongest and 50% second strongest, rest is 0)
pairs= dict(counter)
pairs_array = np.array(list(pairs.keys()))  
pairs_preds = np.zeros((50, 10))

# Set the values in pairs_preds for the specified keys in pairs_array
for i in range(50):
    class_1 = pairs_array[i, 0]
    class_2 = pairs_array[i, 1]
    pairs_preds[i, class_1] = 0.5
    pairs_preds[i, class_2] = 0.5

    # Normalize to make the sum of each row equal to 1 (softmax normalization)
    pairs_preds[i] /= pairs_preds[i].sum()

# Now pairs_preds is a 50x10 array with 50% value for the classes in pairs_array
print(pairs_preds)




#===============================================================================================================



#######
#shared reality:
n_rec=len(train)#679#100
def mse(m,train):
    n=m.shape[0]
    return np.mean((m- train[:n])**2)

def mae(m,train):
    return np.mean(np.abs(np.array(train) - np.array(m)))

    
# =============================================================================
# ################## REVERT BACK TO ORIGINAL HERE
# collected_images = []
# for i in range(679):
#     target=labels[i][1]
#     collected_images.append(imgs_by_label[target][i])
# collected_images=np.array(collected_images).reshape(679,28,28,1)
# 
# train_X = vq_vae.predict(collected_images)#(train)#
# train_X = train_X[1][...,1]
# train_y = encoder.predict(collected_images)#train)#
# train_X = train_X[...].reshape((len(train), n_tokens, d_embed_vec))#n_train_samples, n_tokens, d_embed_vec))#
# 
# masked_train_X = np.copy(train_X)
# masked_train_X[mask_train] = mask_token
# =============================================================================



latent_classifier= load_model('Data/latent_classifier_trained_on_combined_vqvae_output_pred_label_mse_loss_test_highly_amb_10epochs_non-clipped.h5')
C1j=latent_classifier.predict(train_X)

def plot_amb_data_dis():
    fig10, ax10 = plt.subplots()
    counts, edges, bars = ax10.hist(np.max(v1, axis=1))#, orientation='horizontal')
    rects = ax10.patches 
      
    for rect, label in zip(rects, counts): 
        height = rect.get_height() 
        plt.text( rect.get_x() + rect.get_width() / 2,height+0.01, int(label), 
                ha='center', va='bottom', fontsize=8) 
    plt.ylabel('Count') 
    plt.xlabel('Probability of highest judgement (Image classifier)')
    plt.title('Ambiguous data distribution')
    ax10.set_xlim([0.0, 1.0])
    #ax10.grid(axis='y')
    plt.show()
    
plot_amb_data_dis()




#Take audience judgement and give it a high bias (certain, 1) (take it from evals1 not from labels, because labels defere from classifier output (only when applying the clipping)).
audience_onehot = (tf.keras.utils.to_categorical(evals1[:,0], num_classes = 10))
bias_message_construction=np.mean(np.array([audience_onehot, v1,v1]), axis=0 )
bias_MR= np.argpartition(bias_message_construction,-2, axis=1)[:,-2:]




def are_arrays_equal(arr1, arr2):
    # Check if arrays have the same shape
    if arr1.shape != arr2.shape:
        return False
    
    # Sort each row in both arrays and compare
    sorted_arr1 = np.sort(arr1, axis=1)  # Sort each row in arr1
    sorted_arr2 = np.sort(arr2, axis=1)  # Sort each row in arr2

    # Check if the sorted arrays are equal
    return np.array_equal(sorted_arr1, sorted_arr2)

# Example usage:
arr1 = bias_MR
arr2 = evals1

print(are_arrays_equal(arr1, arr2))  # True



def calculate_batch_similarity(original_images, reconstructed_images):
    """
    Calculate the similarity (accuracy) of multiple reconstructed MNIST images compared to original images.
    
    Parameters:
    - original_images (numpy array): A 2D array where each row is a flattened MNIST image (N x 28*28).
    - reconstructed_images (numpy array): A 2D array where each row is a flattened reconstructed MNIST image (N x 28*28).
    
    Returns:
    - numpy array: A 1D array containing the accuracy (percentage of matching pixels) for each image pair.
    """
    # Ensure the batch sizes match
    if original_images.shape != reconstructed_images.shape:
        raise ValueError("The two image batches must have the same shape.")
    
    # Ensure each image is of size (28, 28)
    #if original_images.shape[1] != 28*28 or reconstructed_images.shape[1] != 28*28:
        #raise ValueError("Each image should be flattened to size (28*28).")
    
    # Reshape the images to 28x28 for easier pixel-wise comparison
    original_images = original_images.reshape(-1, 28, 28)
    reconstructed_images = reconstructed_images.reshape(-1, 28, 28)
    
    # Calculate pixel-wise accuracy for each image pair in the batch
    accuracies = np.sum(original_images == reconstructed_images, axis=(1, 2)) / (28 * 28) * 100
    
    return accuracies





S2_s,_=old_recons(train_X,train_y, 0.5, labels[:,1], n_rec)
S2_w,_=old_recons(train_X,train_y, 0.5, labels[:,0], n_rec)
# =============================================================================
# zeros=np.zeros((679))
# 
# S2_s,_=old_recons(train_X,train_y, 0.5, zeros, n_rec)
# S2_w,_=old_recons(train_X,train_y, 0.5, zeros, n_rec)
# =============================================================================
#S2_s,_=old_recons(train_X,train_y, 0.5, labels[:,0], n_rec)
#S2_w,_=old_recons(train_X,train_y, 0.5, labels[:,1], n_rec)

M2_s=retrieve(S2_s)
M2_w=retrieve(S2_w)

mse2_s=mse(M2_s,train)
mse2_w=mse(M2_w,train)

#mse2_s=np.mean(calculate_batch_similarity(M2_s,train))
#mse2_w=np.mean(calculate_batch_similarity(M2_w,train))


mse2_s_all=np.zeros((n_rec))
for i in range(n_rec):
    mse2_s_all[i]=mse(M2_s[i], train[i])
    
    
mse2_w_all=np.zeros((n_rec))
for i in range(n_rec):
    mse2_w_all[i]=mse(M2_w[i], train[i])

mse2_s_std=np.std(mse2_s_all)
mse2_w_std=np.std(mse2_w_all)

#mse2_s_std=np.std(calculate_batch_similarity(M2_s,train))
#mse2_w_std=np.std(calculate_batch_similarity(M2_w,train))

V2_s=classifier.predict(M2_s)
V2_w=classifier.predict(M2_w)
eval2_s=np.argmax(V2_s,axis=1)
eval2_w=np.argmax(V2_w,axis=1)

error2s= (eval2_s != labels[:n_rec,1]).mean()
error2w= (eval2_w != labels[:n_rec,0]).mean()

er2s_all=np.zeros((n_rec))
for i in range(n_rec):
    er2s_all[i]=V2_s[i,labels[i,1]]-V2_s[i,labels[i,0]]
er2s=np.mean(er2s_all)

er2s_std=np.std(er2s_all)
#er2s_std=er2s_std/2

er2w_all=np.zeros((n_rec))
for i in range(n_rec):
    er2w_all[i]=V2_w[i,labels[i,1]]-V2_w[i,labels[i,0]]
er2w=np.mean(er2w_all)

er2w_std=np.std(er2w_all)
#er2w_std=er2w_std/2


#Er2s= V2_s[:,labels[:n_rec,1]] #-V2_s[:,labels[:n_rec,0]]
#Er2w= np.mean(V2_w[:,labels[:,1]]-V2_w[:,labels[:,0]])



########
#no shared reality



#S2_sn,_=old_recons(train_X,train_y, 0.6, zeros, n_rec)#labels[:,1], n_rec)
#S2_wn,_=old_recons(train_X,train_y, 0.6, zeros, n_rec)#labels[:,0], n_rec)

S2_sn,_=old_recons(train_X,train_y, 0.6, labels[:,1], n_rec)
S2_wn,_=old_recons(train_X,train_y, 0.6, labels[:,0], n_rec)


M2_sn=retrieve(S2_sn)
M2_wn=retrieve(S2_wn)

mse2_sn=mse(M2_sn,train)
mse2_wn=mse(M2_wn,train)

#mse2_sn=np.mean(calculate_batch_similarity(M2_sn,train))
#mse2_wn=np.mean(calculate_batch_similarity(M2_wn,train))

mse2_sn_all=np.zeros((n_rec))
for i in range(n_rec):
    mse2_sn_all[i]=mse(M2_sn[i], train[i])
    
    
mse2_wn_all=np.zeros((n_rec))
for i in range(n_rec):
    mse2_wn_all[i]=mse(M2_wn[i], train[i])

mse2_sn_std=np.std(mse2_sn_all)
mse2_wn_std=np.std(mse2_wn_all)

#mse2_sn_std=np.std(calculate_batch_similarity(M2_sn,train))
#mse2_wn_std=np.std(calculate_batch_similarity(M2_wn,train))


V2_sn=classifier.predict(M2_sn)
V2_wn=classifier.predict(M2_wn)
eval2_sn=np.argmax(V2_sn,axis=1)
eval2_wn=np.argmax(V2_wn,axis=1)

error2sn= (eval2_sn != labels[:n_rec,1]).mean()
error2wn= (eval2_wn != labels[:n_rec,0]).mean()

er2sn_all=np.zeros((n_rec))
for i in range(n_rec):
    er2sn_all[i]=V2_sn[i,labels[i,1]]-V2_sn[i,labels[i,0]]
er2sn=np.mean(er2sn_all)

er2sn_std=np.std(er2sn_all)
#er2sn_std=er2sn_std/2

er2wn_all=np.zeros((n_rec))
for i in range(n_rec):
    er2wn_all[i]=V2_wn[i,labels[i,1]]-V2_wn[i,labels[i,0]]
er2wn=np.mean(er2wn_all)

er2wn_std=np.std(er2wn_all)
#er2wn_std=er2wn_std/2


####### Scatter plot for comparing the highest probabilities of the original amb image against the ambiguous image
# Example: Two arrays with shape (679, 10), representing softmax outputs
array1 = v1
array2 = V2_sn

# Get index of highest value in each row (predicted class)
max_index1 = np.argmax(array1, axis=1)
max_index2 = np.argmax(array2, axis=1)
max_value1 = array1[np.arange(len(array1)), max_index1]
max_value2 = array2[np.arange(len(array2)), max_index2]

# Separate points where class stayed the same vs changed
same_mask = max_index1 == max_index2
changed_mask = ~same_mask

# Count for legend
same_count = np.sum(same_mask)
changed_count = np.sum(changed_mask)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(max_value1[same_mask], max_value2[same_mask], color='blue', alpha=0.6,
            label=f'Same class ({same_count})')
plt.scatter(max_value1[changed_mask], max_value2[changed_mask], color='red', alpha=0.6,
            label=f'Changed class ({changed_count})')

plt.xlabel("Max softmax value in M1J")
plt.ylabel("Max softmax value in M2J")
plt.title("Change in Predicted Class Index - Non-shared reality")
#plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





#########

#Accessibility experiments

mnist=np.load('Data/mnist.npz')
x=mnist['x_train']
y=mnist['y_train']

#x=x[::80,:,:]
#y=y[::80]

x_encoded_outputs = vq_vae.predict(x)
x_encoded_outputs = x_encoded_outputs[1][...,1] #Zq


x_encoded_outputs = x_encoded_outputs[...].reshape((x.shape[0], n_tokens, d_embed_vec))
masked_x_encoded_outputs = np.copy(x_encoded_outputs)
mask_train_x_encoded_outputs = np.random.default_rng().choice([True, False], size=(x.shape[0], n_tokens), p=[mask_perc, 1 - mask_perc])

masked_x_encoded_outputs[mask_train_x_encoded_outputs] = mask_token


# Prepare a list of 10 arrays, one for each label
zqs_by_label = [[] for _ in range(10)]

# Iterate over the train images and labels to group them by label
for img, label in zip(masked_x_encoded_outputs, y):
    zqs_by_label[label].append(img)

# Convert each list of images to a numpy array
zqs_by_label = [np.array(arr) for arr in zqs_by_label]

# Print the shapes of each array to confirm
for i, array in enumerate(zqs_by_label):
    print(f"Label {i}: {array.shape}")

# Prepare a list of 10 arrays, one for each label
imgs_by_label = [[] for _ in range(10)]

# Iterate over the train images and labels to group them by label
for img, label in zip(x, y):
    imgs_by_label[label].append(img)

# Convert each list of images to a numpy array
imgs_by_label = [np.array(arr) for arr in imgs_by_label]

# Print the shapes of each array to confirm
for i, array in enumerate(zqs_by_label):
    print(f"Label {i}: {array.shape}")


#plot average images of MNIST along with their labels that you get from the image classifier
# =============================================================================
# for label_index in range (10):
#     images = imgs_by_label[label_index]  # Get the images of label 0
# 
#     # Convert the list of images to a NumPy array if it's not already
#     images = np.array(images)
#     
#     # Calculate the pixel-wise average
#     average_image = np.mean(images, axis=0)
#     print(average_image.shape)
#     label_avg=classifier.predict(average_image.reshape(-1,28,28,1))
#     label_num = np.argpartition(label_avg,-2, axis=1)[:,-1:]
#     
#     # Display the result
#     plt.axis('off')
#     plt.imshow(average_image, cmap='gray')
#     plt.title(f'Average Image, classifier label {label_num}')
#     plt.show()
# =============================================================================



######
#final recall
#shared reality



#S = vq_vae.predict(train)
#S = train_X[1][...,1]
#train_y = encoder.predict(train)
#train_X = train_X[...].reshape((n_train_samples, n_tokens, d_embed_vec))

V_fs = (V2_s+v1[:n_rec])/2
V_fw = (V2_w+v1[:n_rec])/2

#S3_s=recons(train_X,train_y, 0.5, V_fs, n_rec)
#S3_w=recons(train_X,train_y, 0.5, V_fw, n_rec)

#get the Zqs out of the stored memory trace
Zq_S2_s=np.array(codes_sampler(S2_s.reshape((679,7,7)))).reshape((679,49,64))
Zq_S2_w=np.array(codes_sampler(S2_w.reshape((679,7,7)))).reshape((679,49,64))


#take weighted average of the Zqs
#zq_S1_S2_s= np.mean( np.array([ 0.2*masked_train_X, 0.8*Zq_S2_s ]), axis=0 ) 
#zq_S1_S2_w= np.mean( np.array([ 0.2*masked_train_X, 0.8*Zq_S2_w ]), axis=0 ) 

# =============================================================================
# def merge_traces_suboptimal(trace_pairs, condition):
#     """
#     Stitch together multiple traces by taking 33% from the first,
#     17% from the second, and filling the rest with zeros.
# 
#     Parameters:
#     - image_pairs (list of tuples): Each tuple contains two traces.
# 
#     Returns:
#     - numpy array: Stitched traces
#     
#     """
#     
#     if(condition=='shared_reality'):
#         height, width = trace_pairs[0][0].shape  # Get trace dimensions
#         rows_from_trace1 = int(0.25 * height)  # 33% from first trace
#         rows_from_trace2 = int(0.25 * height)  # 17% from second trace
#         rows_remaining = height - (rows_from_trace1 + rows_from_trace2)  # Remaining zero rows
# 
#     if(condition=='non_shared_reality'):
#             
#         height, width = trace_pairs[0][0].shape  # Get trace dimensions
#         rows_from_trace1 = int(0.33 * height)  # 33% from first trace
#         rows_from_trace2 = int(0.17 * height)  # 17% from second trace
#         rows_remaining = height - (rows_from_trace1 + rows_from_trace2)  # Remaining zero rows
#         
#     stitched_traces = []  # List to store stitched traces
#     
#     for trace1, trace2 in trace_pairs:
#         # Extract required portions
#         part1 = trace1[:rows_from_trace1, :]
#         part2 = trace2[rows_from_trace1:rows_from_trace2+rows_from_trace1, :]
#         zero_part = np.zeros((rows_remaining, width))  # Zero padding
#         
#         # Stack vertically
#         stitched_trace = np.vstack([part1, part2, zero_part])
#         stitched_traces.append(stitched_trace)
#     
#     stitched_traces = np.array(stitched_traces)  # Convert to NumPy array
#     
#     return stitched_traces  # Return the stitched images
# =============================================================================


def merge_traces_suboptimal(trace_pairs, condition):
    """
    Stitch together multiple traces by taking 33% from the first,
    17% from the second, and filling the rest with zeros.

    Parameters:
    - trace_pairs (numpy array): A 2D array where each row contains two trace arrays.
    - condition (str): The condition that determines the stitching ratio ('shared_reality' or 'non_shared_reality').

    Returns:
    - numpy array: Stitched traces
    """
    
    if condition == 'shared_reality':
        height, width = trace_pairs[0][0].shape  # Get trace dimensions
        rows_from_trace1 = int(0.17 * height)  # 17% from first trace
        rows_from_trace2 = int(0.33 * height)  # 33% from second trace
        rows_remaining = height - (rows_from_trace1 + rows_from_trace2)  # Remaining zero rows

    if condition == 'non_shared_reality':
        height, width = trace_pairs[0][0].shape  # Get trace dimensions
        rows_from_trace1 = int(0.33 * height)  # 33% from first trace
        rows_from_trace2 = int(0.17 * height)  # 17% from second trace
        rows_remaining = height - (rows_from_trace1 + rows_from_trace2)  # Remaining zero rows
        
    stitched_traces = []  # List to store stitched traces
    
    for trace_pair in trace_pairs:
        trace1 = trace_pair[0]  # Get the first trace
        trace2 = trace_pair[1]  # Get the second trace
        
        # Extract required portions
        part1 = trace1[:rows_from_trace1, :]
        part2 = trace2[rows_from_trace1:rows_from_trace2 + rows_from_trace1, :]
        zero_part = np.zeros((rows_remaining, width))  # Zero padding
        
        # Stack vertically
        stitched_trace = np.vstack([part1, part2, zero_part])
        stitched_traces.append(stitched_trace)
    
    stitched_traces = np.array(stitched_traces)  # Convert to NumPy array
    
    return stitched_traces  # Return the stitched traces




def merge_traces(Zq_C1I_list, Zq_C2I_list, ratio1, ratio2):
    """
    Processes multiple 49×64 arrays, setting 77% of pixels in Zq_C1I to zero,
    83% in Zq_C2I (without overlap), then averages the three arrays.

    Parameters:
    - Zq_C1I_list (list of numpy arrays): List of first 49×64 arrays.
    - Zq_C2I_list (list of numpy arrays): List of second 49×64 arrays.
    - ratio1 (float): Fraction of pixels to turn to zero in Zq_C1I (default: 77%).
    - ratio2 (float): Fraction of pixels to turn to zero in Zq_C2I (default: 83%).
    - show (bool): Whether to display the results. Default is True.

    Returns:
    - numpy array: Averaged batch of 49×64 arrays (shape: (num_samples, 49, 64)).
    """
    num_samples = len(Zq_C1I_list)
    height, width = Zq_C1I_list[0].shape  # Should be (49, 64)
    total_pixels = height * width  # Total number of pixels = 49*64
    num_pixels_C1I = int(ratio1 * total_pixels)  # 77% from Zq_C1I
    num_pixels_C2I = int(ratio2 * total_pixels)  # 83% from Zq_C2I

    averaged_results = []  # Store results

    for i in range(num_samples):
        Zq_C1I = Zq_C1I_list[i]
        Zq_C2I = Zq_C2I_list[i]

        # Initialize zero array
        zero_array = np.zeros((height, width))

        # Generate random indices without overlap
        all_indices = np.arange(total_pixels)
        np.random.shuffle(all_indices)  # Shuffle all indices randomly

        # Select 77% of pixels from Zq_C1I to turn to zero
        indices_C1I = all_indices[:num_pixels_C1I]

        # Select 83% of pixels from Zq_C2I (non-overlapping with C1I)
        remaining_indices = np.setdiff1d(all_indices, indices_C1I, assume_unique=True)
        np.random.shuffle(remaining_indices)
        indices_C2I = remaining_indices[:num_pixels_C2I]

        # Convert flat indices to (row, col) positions
        rows_C1I, cols_C1I = np.unravel_index(indices_C1I, (height, width))
        rows_C2I, cols_C2I = np.unravel_index(indices_C2I, (height, width))

        # Create copies of the original arrays
        modified_C1I = Zq_C1I.copy()
        modified_C2I = Zq_C2I.copy()

        # Set selected pixels to zero
        modified_C1I[rows_C1I, cols_C1I] = 0
        modified_C2I[rows_C2I, cols_C2I] = 0

        # Compute the element-wise average
        averaged_image = (modified_C1I + modified_C2I + zero_array) / 3.0
        averaged_results.append(averaged_image)

    averaged_results = np.array(averaged_results)  # Convert to NumPy array

  
    return averaged_results





def merge_traces_modified(Zq_C1I_list, Zq_C2I_list, ratio1, ratio2):
    """
    Processes multiple 49×64 arrays, setting 77% of rows in Zq_C1I to zero,
    83% of rows in Zq_C2I (without overlap), then averages the three arrays.

    Parameters:
    - Zq_C1I_list (list of numpy arrays): List of first 49×64 arrays.
    - Zq_C2I_list (list of numpy arrays): List of second 49×64 arrays.
    - ratio1 (float): Fraction of rows to turn to zero in Zq_C1I.
    - ratio2 (float): Fraction of rows to turn to zero in Zq_C2I.

    Returns:
    - numpy array: Averaged batch of 49×64 arrays (shape: (num_samples, 49, 64)).
    """
    num_samples = len(Zq_C1I_list)
    height, width = Zq_C1I_list[0].shape  # Should be (49, 64)
    
    num_rows_C1I = int(ratio1 * height)  # Number of rows to zero in Zq_C1I
    num_rows_C2I = int(ratio2 * height)  # Number of rows to zero in Zq_C2I

    averaged_results = []  # Store results

    for i in range(num_samples):
        Zq_C1I = Zq_C1I_list[i]
        Zq_C2I = Zq_C2I_list[i]

        # Initialize zero array
        zero_array = np.zeros((height, width))

        # Generate random row indices without overlap
        all_row_indices = np.arange(height)
        np.random.shuffle(all_row_indices)  # Shuffle row indices

        # Select 77% of rows from Zq_C1I to turn to zero
        rows_C1I = all_row_indices[:num_rows_C1I]

        # Select 83% of rows from Zq_C2I (non-overlapping with C1I)
        remaining_rows = np.setdiff1d(all_row_indices, rows_C1I, assume_unique=True)
        np.random.shuffle(remaining_rows)
        rows_C2I = remaining_rows[:num_rows_C2I]

        # Create copies of the original arrays
        modified_C1I = Zq_C1I.copy()
        modified_C2I = Zq_C2I.copy()

        # Set selected rows to zero
        modified_C1I[rows_C1I, :] = 0
        modified_C2I[rows_C2I, :] = 0

        # Compute the element-wise average
        averaged_image = (modified_C1I + modified_C2I + zero_array) / 3.0
        averaged_results.append(averaged_image)

    averaged_results = np.array(averaged_results)  # Convert to NumPy array

    return averaged_results


def get_pairs(data_by_label,pairs_array):
    arr1_arr2=[]
    for i in range(len(pairs_array)):
            
        arr1=np.array(data_by_label[pairs_array[i][0]][i])
        arr2=np.array(data_by_label[pairs_array[i][1]][i])
        #zq1=np.zeros((49,64))
        #zq2=np.zeros((49,64))
        arr1_arr2.append(np.array([arr1,arr2]))
    arr1_arr2=np.array(arr1_arr2)
    return arr1_arr2



def plot_recons(data,label,pred,mask,n_row=5,n_col=5,start=0):   
    n=n_col*n_row
    fig = plt.figure(figsize=(20, 15))
    for i in range(n):
        plt.subplot(n_row,n_col,i+1)
        if data.shape[-1]==28:
            plt.imshow(data[i+start,:,:], cmap='gray')
        else:
            plt.imshow(data[i+start,:,:,0], cmap='gray')
        plt.title(str(label[i+start,1]) +','+ str())
        label1=label[i+start,1]
        label2=label[i+start,0]
        pred1=pred[i+start,1]
        pred2=pred[i+start,0]
        plt.title(f"({label1}: {pred1:.2f})\n({label2}: {pred2:.2f})")
        plt.axis('off')
    plt.suptitle('50-50 Biased MNIST images '+str(int(mask*100))+'% masking', fontsize=20)
    plt.show()
    fig.savefig('./' + date + '_figures/biased-MNIST_masking'+str(int(mask*100))+'.png')
    

    
#control experiments for creating ambiguous images using the transformer
zq1_zq2=get_pairs(zqs_by_label,pairs_array)

first_rows = []  # to collect first 10 images per masking level

masking=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
# =============================================================================
# for idx,mask in enumerate(masking):
#     transformer_amb_Zq=merge_traces_modified(zq1_zq2[:,0],zq1_zq2[:,1],0.5,0.5)
#     transformer_amb_Zq_generated=decoder.predict(transformer_amb_Zq.reshape((-1,7,7,64)))
#     
#     
#     transformer_amb_trace,transformer_amb_trace_logits=recons(transformer_amb_Zq,train_y, mask, pairs_preds, 50)
#     transformer_amb_img=retrieve(transformer_amb_trace)
#     transformer_amb_img_pred=classifier.predict(transformer_amb_img.reshape(-1,28,28,1))
#     
#     #transformer_amb_img_label=np.argmax(transformer_amb_img_pred,1)
#     transformer_amb_img_label_s_w=np.argpartition(transformer_amb_img_pred,-2, axis=1)[:,-2:]
#     transformer_amb_img_pred_s_w=np.sort(transformer_amb_img_pred, axis=1)[:,-2:]
#     #plot_recons(transformer_amb_img,transformer_amb_img_label_s_w,transformer_amb_img_pred_s_w,mask,5,10,0)
#     selected_values = np.stack([transformer_amb_img_pred[np.arange(50), pairs_array[:, 0]],transformer_amb_img_pred[np.arange(50), pairs_array[:, 1]]], axis=1)
#     plot_recons(transformer_amb_img,pairs_array,selected_values,mask,5,10,0)
# 
# =============================================================================

#For varying the masking and plotting reconstructions against each masking level
first_rows = []       # To store images
first_row_labels = [] # To store labels
first_row_preds = []  # To store predictions

for idx, mask in enumerate(masking):
    transformer_amb_Zq = merge_traces_modified(zq1_zq2[:, 0], zq1_zq2[:, 1], 0.5, 0.5)
    transformer_amb_Zq_generated = decoder.predict(transformer_amb_Zq.reshape((-1, 7, 7, 64)))

    transformer_amb_trace, transformer_amb_trace_logits = recons(
        transformer_amb_Zq, train_y, mask, pairs_preds, 50
    )
    transformer_amb_img = retrieve(transformer_amb_trace)
    transformer_amb_img_pred = classifier.predict(transformer_amb_img.reshape(-1, 28, 28, 1))

    # Save image, label, and prediction of the first 10 items
    selected_imgs = transformer_amb_img[:10]
    selected_pairs = pairs_array[:10]
    selected_preds = transformer_amb_img_pred[:10]
    selected_values = np.stack([
        selected_preds[np.arange(10), selected_pairs[:, 0]],
        selected_preds[np.arange(10), selected_pairs[:, 1]]
    ], axis=1)

    first_rows.append(selected_imgs)
    first_row_labels.append(selected_pairs)
    first_row_preds.append(selected_values)


def plot_stacked_first_rows_with_labels(images_by_mask, labels_by_mask, preds_by_mask, masking):
    n_rows = len(images_by_mask)
    n_cols = images_by_mask[0].shape[0]  # 10 images per row

    fig = plt.figure(figsize=(n_cols * 1.6, n_rows * 1.8))

    for row in range(n_rows):
        for col in range(n_cols):
            i = row * n_cols + col
            plt.subplot(n_rows, n_cols, i + 1)
            img = images_by_mask[row][col]
            label1 = labels_by_mask[row][col, 1]
            label2 = labels_by_mask[row][col, 0]
            pred1 = preds_by_mask[row][col, 1]
            pred2 = preds_by_mask[row][col, 0]

            if img.ndim == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img[:, :, 0], cmap='gray')

            plt.title(f"({label1}:{pred1:.2f})\n({label2}:{pred2:.2f})", fontsize=10)
            if col == 0:
                plt.ylabel(f"{int(masking[row]*100)}%", fontsize=12)
            plt.axis('off')

    #plt.suptitle("First 10 Reconstructions with Labels Across Masking Levels", fontsize=16)
    mask_labels = ", ".join([f"{int(m*100)}%" for m in masking])
    plt.suptitle(f"Reconstructed MNIST by Masking Level (Rows = {mask_labels})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('./' + date + '_figures/biased-MNIST_masking-all.png')

    plt.show()

plot_stacked_first_rows_with_labels(first_rows, first_row_labels, first_row_preds, masking)



img1_img2=get_pairs(imgs_by_label,pairs_array)
mnist_merged_img=merge_traces(img1_img2[:,0],img1_img2[:,1], 0.5,0.5)#merge_traces_suboptimal(img1_img2, condition='shared_reality')#
mnist_merged_zq = vq_vae.predict(mnist_merged_img.reshape(-1,28,28,1))
mnist_merged_zq = mnist_merged_zq[1][...,1] #Zq
mnist_merged_zq=mnist_merged_zq.reshape(-1,49,64)


amb_images_preds = np.zeros((len(train), 10))

# Set the values in pairs_preds for the specified keys in pairs_array
for i in range(len(train)):
    cls_1 = evals1[i, 0]
    cls_2 = evals1[i, 1]
    amb_images_preds[i, cls_1] = 0.5
    amb_images_preds[i, cls_2] = 0.5

    # Normalize to make the sum of each row equal to 1 (softmax normalization)
    amb_images_preds[i] /= amb_images_preds[i].sum()

import random
selected_images = []
# Pick one random image from each of the target labels
for j in range(n_rec):
    selected_image = random.choice(imgs_by_label[evals1[j][0]])  # Randomly pick an image from the selected class
    selected_images.append(selected_image)  # Append the image and its label
selected_images=np.array(selected_images)
selected_images_zqs = vq_vae.predict(selected_images.reshape(-1,28,28,1))
selected_images_zqs = selected_images_zqs[1][...,1] #Zq
selected_images_zqs=selected_images_zqs.reshape(-1,49,64)


masking=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
for idx,mask in enumerate(masking):
    mnist_merged_trace,mnist_merged_trace_logits=recons(selected_images_zqs,train_y, mask, amb_images_preds, n_rec)
    mnist_merged_trace_img=retrieve(mnist_merged_trace)
    mnist_merged_trace_pred=classifier.predict(mnist_merged_trace_img.reshape(-1,28,28,1))
    
    #mnist_merged_trace_label=np.argmax(mnist_merged_trace_pred,1)
    mnist_merged_trace_label_s_w=np.argpartition(mnist_merged_trace_pred,-2, axis=1)[:,-2:]
    mnist_merged_trace_pred_s_w=np.sort(mnist_merged_trace_pred, axis=1)[:,-2:]
    
    plot_recons(mnist_merged_trace_img,mnist_merged_trace_label_s_w,mnist_merged_trace_pred_s_w,mask,5,10,0)


from scipy.special import softmax



# =============================================================================
# max_pred=np.round(np.max(transformer_amb_img_pred,1),2)
# for i in range (len(pairs_array)): 
#     plt.imshow(transformer_amb_img[i].reshape(28,28),cmap='gray')
#     label=transformer_amb_img_label[i]
#     pred=max_pred[i]
#     plt.title(f"Zero traces \n 0.9 masking \n ({label}: {pred:.2f})")
#     plt.axis('off')
#     plt.show()
# =============================================================================


def plot_images(data,label,n_row=5,n_col=5,start=0):   
    n=n_col*n_row
    fig = plt.figure(figsize=(20, 15))
    for i in range(n):
        plt.subplot(n_row,n_col,i+1)
        if data.shape[-1]==28:
            plt.imshow(data[i+start,:,:], cmap='gray')
        else:
            plt.imshow(data[i+start,:,:,0], cmap='gray')
        plt.title(str(label[i+start,1]) +','+ str())
        label1=label[i+start,1]
        label2=label[i+start,0]

        plt.title(f"({label1},{label2})")
        plt.axis('off')
    plt.suptitle('Optimally merged 50%-50% MNIST Zqs', fontsize=20)
    plt.show()
    fig.savefig('./' + date + '_figures/merged-50-mnist-zq.png')


plot_images(transformer_amb_Zq_generated,pairs_array,5,10,0)
plot_images(mnist_merged_img,pairs_array,5,10,0)





plot_recons(transformer_amb_img,transformer_amb_img_label_s_w,transformer_amb_img_pred_s_w,6,6,0)
    
transformer_amb_img_conf=np.mean(np.max(transformer_amb_trace_logits, axis=-1).reshape(-1, 49))
transformer_amb_trace_most_probable= transformer_amb_trace_logits.argmax(axis=-1)




# =============================================================================
# def merge_traces(trace1,trace2):
#     height, width = trace1.shape
#     # Compute the number of rows to take from each image
#     rows_from_trace1 = int(0.33 * height)  # 33% from first image
#     rows_from_trace2 = int(0.17 * height)  # 17% from second image
#     
#     # Take the required rows from each image
#     part1 = trace1[:rows_from_trace1, :]
#     part2 = trace2[:rows_from_trace2, :]
#     
#     # Compute the remaining rows to fill with zeros
#     rows_remaining = height - (rows_from_trace1 + rows_from_trace2)
#     
#     # Create a zero matrix for the remaining part
#     zero_part = np.zeros((rows_remaining, width))
#     
#     # Stitch them together (stack vertically)
#     stiched_trace = np.vstack([part1, part2, zero_part])
#     
#     return stiched_trace
# =============================================================================


#zq_S1_S2_s=merge_traces_suboptimal(list(zip(masked_train_X,Zq_S2_s)), condition='shared_reality')#, ratio1=0.83, ratio2=0.77 )
#zq_S1_S2_w=merge_traces_suboptimal(list(zip(masked_train_X,Zq_S2_w)), condition='shared_reality')#, ratio1=0.83, ratio2=0.77)
zq_S1_S2_s=merge_traces_modified(masked_train_X,Zq_S2_s, ratio1=0.83, ratio2=0.77)
zq_S1_S2_w=merge_traces_modified(masked_train_X,Zq_S2_w,ratio1=0.83, ratio2=0.77)



#take weighted average of the probalistic biases
VC2_VC1_prob_s=[]
for i in range(n_rec):
    VC2_VC1_s=np.vstack(( [0.8*V2_s[i], 0.2*v1[i]]))
    VC2_VC1_prob_s.append(np.mean(VC2_VC1_s, axis=0))
    
VC2_VC1_prob_s=np.array(VC2_VC1_prob_s)


VC2_VC1_prob_w=[]
for i in range(n_rec):
    VC2_VC1_w=np.vstack(( [0.8*V2_w[i], 0.2*v1[i]]))
    VC2_VC1_prob_w.append(np.mean(VC2_VC1_w, axis=0))
    
VC2_VC1_prob_w=np.array(VC2_VC1_prob_w)



#get strong bias
v1_v2_s=np.argpartition(VC2_VC1_prob_s,-2, axis=1)[:,-1]
#get weak bias
v1_v2_w=np.argpartition(VC2_VC1_prob_w,-2, axis=1)[:,-1]



#control experimets before paper publishiing
# =============================================================================
# def compare_arrays(arr1, arr2):
#     # Ensure that arr2 is a 2D array
#     assert arr2.ndim == 2 and arr2.shape[1] == 2, "arr2 must be a 2D array with 2 columns"
#     
#     # List to store the indices where arr1[i] matches arr2[i, 0] or arr2[i, 1]
#     matching_indices_0 = []
#     matching_indices_1 = []
#     matching_indices_none=[]
# 
# 
#     # Iterate over the 1D array and compare with the corresponding elements in the 2D array
#     for i in range(len(arr1)):
#         if arr1[i] == arr2[i, 0]:
#             matching_indices_0.append(i)
#         if arr1[i] == arr2[i, 1]:
#             matching_indices_1.append(i)
#         else:
#             matching_indices_none.append(i)
#     
#     return matching_indices_0,matching_indices_1,matching_indices_none
# 
# 
# # Get the indices where recall combined bias matches evals1[i,0] or evals1[i,1] or none.
# result = compare_arrays(v1_v2_s, evals1)
# 
# print("Number of indices where v1_v2_s matches weak label:", len(result[0]))
# print("Number of indices where v1_v2_s matches strong label:", len(result[1]))
# print("Number of indices where v1_v2_s matches none:", len(result[2]))
# 
# 
# 
# import time
# 
# time1=[]
# time2=[]
# time3=[]
# time4=[]
# 
# logits_mnist_s_img_congruent = []
# logits_mnist_s_img_incongruent = []
# 
# logits_mnist_w_img_congruent = []
# logits_mnist_w_img_incongruent = []
# 
# 
# prob_mnist_s_img_congruent = []
# prob_mnist_s_img_incongruent = []
# 
# prob_mnist_w_img_congruent = []
# prob_mnist_w_img_incongruent = []
# 
# for i in range (len(v1_v2_s)):
#     #congurent (in shared reality, the original weak label (evals1[:,0]) is congruent with the recall phase strong label)
#     if (v1_v2_s[i]==evals1[i,0]):
#         #get zqs of MNIST mean image
#         #zq_s_mean_img=np.mean(zqs_by_label[v1_v2_s[i]], axis=0) no mean image is used
#         mnist_s_img_congruent,logits_s_reaction_time=recons(zqs_by_label[v1_v2_s[i]][i], train_y, 0.01,VC2_VC1_prob_s[i] ,1)
#         logits_mnist_s_img_congruent.append(logits_s_reaction_time)
#         M_mnist_s_img_congruent=retrieve(mnist_s_img_congruent)
#         batch_start_time = time.time()
#         prob_mnist_s_img_congruent.append(classifier.predict(M_mnist_s_img_congruent))
#         batch_end_time = time.time()
#         time1.append((batch_end_time - batch_start_time) * 1000)  # Convert to ms
# 
#     #incongurent
#     if (v1_v2_s[i]==evals1[i,1]):   
#         mnist_s_img_incongruent,logits_s_reaction_time=recons(zqs_by_label[v1_v2_s[i]][i], train_y, 0.01,VC2_VC1_prob_s[i] ,1)
#         logits_mnist_s_img_incongruent.append(logits_s_reaction_time)
#         M_mnist_s_img_incongruent=retrieve(mnist_s_img_incongruent)
#         batch_start_time = time.time()
#         prob_mnist_s_img_incongruent.append(classifier.predict(M_mnist_s_img_incongruent ))
#         batch_end_time = time.time()
#         time2.append((batch_end_time - batch_start_time) * 1000)  # Convert to ms
#        
# for i in range (len(v1_v2_w)):
#     #congurent (in shared reality, the original strong label (evals1[:,1]) is congurent with the recall phase weak label)
#     if (v1_v2_w[i]==evals1[i,1]):
#         
#         #zq_w_mean_img=np.mean(zqs_by_label[v1_v2_w[i]], axis=0) no mean image is used
#         mnist_w_img_congruent,logits_w_reaction_time=recons(zqs_by_label[v1_v2_w[i]][i], train_y, 0.01,VC2_VC1_prob_w[i] ,1)
#         logits_mnist_w_img_congruent.append(logits_w_reaction_time)
#         M_mnist_w_img_congruent=retrieve(mnist_w_img_congruent)
#         batch_start_time = time.time()
#         prob_mnist_w_img_congruent.append(classifier.predict(M_mnist_w_img_congruent ))
#         batch_end_time = time.time()
#         time3.append((batch_end_time - batch_start_time) * 1000)  # Convert to ms
# 
# 
#     #incongruent
#     if(v1_v2_w[i]==evals1[i,0]):
#         mnist_w_img_incongruent,logits_w_reaction_time=recons(zqs_by_label[v1_v2_w[i]][i], train_y, 0.01,VC2_VC1_prob_w[i] ,1)
#         logits_mnist_w_img_incongruent.append(logits_w_reaction_time)
#         M_mnist_w_img_incongruent=retrieve(mnist_w_img_incongruent)
#         batch_start_time = time.time()
#         prob_mnist_w_img_incongruent.append(classifier.predict(M_mnist_w_img_incongruent ))
#         batch_end_time = time.time()
#         time4.append((batch_end_time - batch_start_time) * 1000)  # Convert to ms
# 
# 
#     
# max_logits_mnist_s_img_congruent=np.mean(np.max(logits_mnist_s_img_congruent, axis=-1).reshape(len(logits_mnist_s_img_congruent), 49))
# max_logits_mnist_s_img_incongruent=np.mean(np.max(logits_mnist_s_img_incongruent, axis=-1).reshape(len(logits_mnist_s_img_incongruent), 49))
# max_logits_mnist_w_img_congruent=np.mean(np.max(logits_mnist_w_img_congruent, axis=-1).reshape(len(logits_mnist_w_img_congruent), 49))
# max_logits_mnist_w_img_incongruent=np.mean(np.max(logits_mnist_w_img_incongruent, axis=-1).reshape(len(logits_mnist_w_img_incongruent), 49))
# 
# 
# 
# # Sample 2D array
# array1=np.copy(prob_mnist_s_img_congruent).reshape(len(prob_mnist_s_img_congruent), -1)
# array2=np.copy(prob_mnist_w_img_congruent).reshape(len(prob_mnist_w_img_congruent), -1)
# array3=np.copy(prob_mnist_s_img_incongruent).reshape(len(prob_mnist_s_img_incongruent), -1)
# array4=np.copy(prob_mnist_w_img_incongruent).reshape(len(prob_mnist_w_img_incongruent), -1)
# 
# 
# # Get the maximum value along the second dimension (columns)
# max_values1 = np.max(array1, axis=1)
# max_values2 = np.max(array2, axis=1)
# max_values3 = np.max(array3, axis=1)
# max_values4 = np.max(array4, axis=1)
# 
# 
# max_prob_mnist_s_img_congruent=np.mean(max_values1)
# max_prob_mnist_w_img_congruent=np.mean(max_values2)
# max_prob_mnist_s_img_incongruent=np.mean(max_values3)
# max_prob_mnist_w_img_incongruent=np.mean(max_values4)
# 
# 
# mean_time1=np.mean(time1)
# mean_time2=np.mean(time2)
# mean_time3=np.mean(time3)
# mean_time4=np.mean(time4)
# 
# 
# 
# mean_max_prob_mnist_s_img=np.mean([max_prob_mnist_s_img_congruent,max_prob_mnist_s_img_incongruent])
# 
# mean_max_prob_mnist_w_img=np.mean([max_prob_mnist_w_img_congruent, max_prob_mnist_w_img_incongruent])
# 
# differene_pos=max_prob_mnist_w_img_incongruent- max_prob_mnist_s_img_incongruent
# difference_neg=max_prob_mnist_s_img_congruent- max_prob_mnist_w_img_congruent
# 
# 
# import matplotlib.pyplot as plt
# 
# # Example numbers
# number_1 = max_logits_mnist_s_img_congruent
# number_2 = max_logits_mnist_w_img_congruent
# 
# # Compute the difference
# difference = abs(number_1 - number_2)
# 
# # Create a bar plot
# plt.bar(['+ve Congruent', '+ve Incongruent', '-ve Congruent', '-ve Incongruent'], [mean_time2, mean_time1, mean_time3, mean_time4], color=['blue', 'orange', 'blue', 'orange' ])
# #plt.title('congruent')
# plt.ylabel('time')
# # Optionally, display the difference as a separate bar
# #plt.bar('Difference', difference, color='green', alpha=0.5)
# 
# # Show the plot
# plt.show()
# 
# =============================================================================
# =============================================================================
# import seaborn as sns
# 
# # Calculate the difference
# diff = max_logits_S3_s_img - max_logits_avg_s_img
# 
# # Plot the difference as a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(diff, cmap='coolwarm', annot=True, fmt='.2f', cbar=True)
# plt.title("Heatmap of Difference Between the confidences of MNIST and ambiguous stimuli at recall")
# plt.xlabel("Index")
# plt.ylabel("Index")
# plt.tight_layout()
# plt.show()
# 
# # Plot the difference
# plt.imshow(diff, cmap='coolwarm', interpolation='nearest')
# plt.colorbar(label="Difference")
# plt.title("Element-wise Difference Between the confidences of MNIST and ambiguous stimuli at recall")
# plt.xlabel("Index")
# plt.ylabel("Index")
# plt.show()
# 
# 
# # Set up the subplots
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# 
# # Plot the first array
# sns.heatmap(max_logits_S3_s_img, ax=axes[0], cmap='Blues', annot=True, fmt='.2f', cbar=True)
# axes[0].set_title("Array 1")
# 
# # Plot the second array
# sns.heatmap(max_logits_avg_s_img, ax=axes[1], cmap='Reds', annot=True, fmt='.2f', cbar=True)
# axes[1].set_title("Array 2")
# 
# plt.suptitle("Side-by-Side Comparison of the confidences of MNIST and ambiguous stimuli at recall")
# plt.tight_layout()
# plt.show()
# 
# 
# # Plot contour of the difference
# plt.contourf(diff, cmap='coolwarm', levels=50)
# plt.colorbar(label="Difference")
# plt.title("Contour Plot of Difference Between the confidences of MNIST and ambiguous stimuli at recall")
# plt.xlabel("Index")
# plt.ylabel("Index")
# plt.show()
# 
# # Plot histogram of the differences
# plt.hist(diff.flatten(), bins=30, color='b', edgecolor='black')
# plt.title("Histogram of Differences Between the confidences of MNIST and ambiguous stimuli at recall")
# plt.xlabel("Difference Value")
# plt.ylabel("Frequency")
# plt.show()
# =============================================================================


#shared reality 
zeros_prob=np.zeros((679,10))   
#S3_s,_=recons(zq_S1_S2_s,train_y, 0.5, zeros_prob, n_rec)#VC2_VC1_prob_s, n_rec)
#S3_w,_=recons(zq_S1_S2_w,train_y, 0.5, zeros_prob, n_rec)#VC2_VC1_prob_w, n_rec)

S3_s,_=recons(zq_S1_S2_s,train_y, 0.5, VC2_VC1_prob_s, n_rec)
S3_w,_=recons(zq_S1_S2_w,train_y, 0.5, VC2_VC1_prob_w, n_rec)


M3_s=retrieve(S3_s)
M3_w=retrieve(S3_w)

mse3_s=mse(M3_s,train)
mse3_w=mse(M3_w,train)

#mse3_s=np.mean(calculate_batch_similarity(M3_s,train))
#mse3_w=np.mean(calculate_batch_similarity(M3_w,train))

mse3_s_all=np.zeros((n_rec))
for i in range(n_rec):
    mse3_s_all[i]=mse(M3_s[i], train[i])
    
    
mse3_w_all=np.zeros((n_rec))
for i in range(n_rec):
    mse3_w_all[i]=mse(M3_w[i], train[i])

mse3_s_std=np.std(mse3_s_all)
mse3_w_std=np.std(mse3_w_all)

#mse3_s_std=np.std(calculate_batch_similarity(M3_s,train))
#mse3_w_std=np.std(calculate_batch_similarity(M3_w,train))


V3_s=classifier.predict(M3_s)
V3_w=classifier.predict(M3_w)
eval3_s=np.argmax(V3_s,axis=1)
eval3_w=np.argmax(V3_w,axis=1)


error3s= (eval3_s != labels[:n_rec,1]).mean()
error3w= (eval3_w != labels[:n_rec,0]).mean()

er3s_all=np.zeros((n_rec))
for i in range(n_rec):
    er3s_all[i]=V3_s[i,labels[i,1]]-V3_s[i,labels[i,0]]
er3s=np.mean(er3s_all)

er3s_std=np.std(er3s_all)
#er3s_std=er3s_std/2

er3w_all=np.zeros((n_rec))
for i in range(n_rec):
    er3w_all[i]=V3_w[i,labels[i,1]]-V3_w[i,labels[i,0]]
er3w=np.mean(er3w_all)

er3w_std=np.std(er3w_all)
#er3w_std=er3w_std/2


#####
#final recall
#no shared reality

VC2_VC1_prob_sn=[]
for i in range(n_rec):
    VC2_VC1_sn=np.vstack(( [0.2*V2_sn[i], 0.8*v1[i]]))
    VC2_VC1_prob_sn.append(np.mean(VC2_VC1_sn, axis=0))
        
VC2_VC1_prob_sn=np.array(VC2_VC1_prob_sn)


VC2_VC1_prob_wn=[]
for i in range(n_rec):
    VC2_VC1_wn=np.vstack(( [0.2*V2_wn[i], 0.8*v1[i]]))
    VC2_VC1_prob_wn.append(np.mean(VC2_VC1_wn, axis=0))
        
VC2_VC1_prob_wn=np.array(VC2_VC1_prob_wn)


v1_v2_sn=np.argpartition(VC2_VC1_prob_sn,-2, axis=1)[:,-1]
v1_v2_wn=np.argpartition(VC2_VC1_prob_wn,-2, axis=1)[:,-1]



#get the Zqs out of the stored memory trace
Zq_S2_sn=np.array(codes_sampler(S2_sn.reshape((679,7,7)))).reshape((679,49,64))
Zq_S2_wn=np.array(codes_sampler(S2_wn.reshape((679,7,7)))).reshape((679,49,64))

#take weighted average of the Zqs
#zq_S1_S2_sn= np.mean( np.array([ 0.8*masked_train_X, 0.2*Zq_S2_sn ]), axis=0 ) 
#zq_S1_S2_wn= np.mean( np.array([ 0.8*masked_train_X, 0.2*Zq_S2_wn ]), axis=0 ) 


#zq_S1_S2_sn=merge_traces_suboptimal(list(zip(masked_train_X,Zq_S2_sn)), condition='non_shared_reality')#, ratio1=0.77, ratio2=0.83)
#zq_S1_S2_wn=merge_traces_suboptimal(list(zip(masked_train_X,Zq_S2_wn)), condition='non_shared_reality')#, ratio1=0.77, ratio2=0.83)
zq_S1_S2_sn=merge_traces_modified(masked_train_X,Zq_S2_sn, ratio1=0.77, ratio2=0.83)
zq_S1_S2_wn=merge_traces_modified(masked_train_X,Zq_S2_wn, ratio1=0.77, ratio2=0.83)



#S3_sn,_=recons(zq_S1_S2_sn,train_y, 0.5, zeros_prob, n_rec)#VC2_VC1_prob_sn, n_rec)
#S3_wn,_=recons(zq_S1_S2_wn,train_y, 0.5, zeros_prob, n_rec)#VC2_VC1_prob_wn, n_rec)
S3_sn,_=recons(zq_S1_S2_sn,train_y, 0.5, VC2_VC1_prob_sn, n_rec)
S3_wn,_=recons(zq_S1_S2_wn,train_y, 0.5, VC2_VC1_prob_wn, n_rec)


M3_sn=retrieve(S3_sn)
M3_wn=retrieve(S3_wn)


mse3_sn=mse(M3_sn,train)
mse3_wn=mse(M3_wn,train)

#mse3_sn=np.mean(calculate_batch_similarity(M3_sn,train))
#mse3_wn=np.mean(calculate_batch_similarity(M3_wn,train))



mse3_sn_all=np.zeros((n_rec))
for i in range(n_rec):
    mse3_sn_all[i]=mse(M3_sn[i], train[i])
    
    
mse3_wn_all=np.zeros((n_rec))
for i in range(n_rec):
    mse3_wn_all[i]=mse(M3_wn[i], train[i])

mse3_sn_std=np.std(mse3_sn_all)
mse3_wn_std=np.std(mse3_wn_all)

#mse3_sn_std=np.std(calculate_batch_similarity(M3_sn,train))
#mse3_wn_std=np.std(calculate_batch_similarity(M3_wn,train))

V3_sn=classifier.predict(M3_sn)
V3_wn=classifier.predict(M3_wn)
eval3_sn=np.argmax(V3_sn,axis=1)
eval3_wn=np.argmax(V3_wn,axis=1)

error3sn= (eval3_sn != labels[:n_rec,1]).mean()
error3wn= (eval3_wn != labels[:n_rec,0]).mean()

#these are the errors that i actually used
er3sn_all=np.zeros((n_rec))
for i in range(n_rec):
    er3sn_all[i]=V3_sn[i,labels[i,1]]-V3_sn[i,labels[i,0]]
er3sn=np.mean(er3sn_all)

er3sn_std=np.std(er3sn_all)
#er3sn_std=er3sn_std/2

er3wn_all=np.zeros((n_rec))
for i in range(n_rec):
    er3wn_all[i]=V3_wn[i,labels[i,1]]-V3_wn[i,labels[i,0]]
er3wn=np.mean(er3wn_all)

er3wn_std=np.std(er3wn_all)
#er3wn_std=er3wn_std/2


#Filtering indicies so that strong and weak judgements stay above a threshold throughout the experiments
#####
filtered_idxs=[]

for i in range(len(train)):
    if((V2_s[i,labels[i,1]]+V2_s[i,labels[i,0]])>0.7 and (V2_sn[i,labels[i,1]]+V2_sn[i,labels[i,0]])>0.7 and (V2_w[i,labels[i,1]]+V2_w[i,labels[i,0]])>0.7 and (V2_wn[i,labels[i,1]]+V2_wn[i,labels[i,0]])>0.7):
        filtered_idxs.append(i)

# =============================================================================
# n_rec=len(filtered_idxs)
# 
# er2s_all=np.zeros((n_rec))
# for i in range(n_rec):
#     er2s_all[i]=V2_s[i,labels[i,1]]-V2_s[i,labels[i,0]]
# er2s=np.mean(er2s_all)
# 
# er2s_std=np.std(er2s_all)
# er2s_std=er2s_std/2
# 
# er2w_all=np.zeros((n_rec))
# for i in range(n_rec):
#     er2w_all[i]=V2_w[i,labels[i,1]]-V2_w[i,labels[i,0]]
# er2w=np.mean(er2w_all)
# 
# er2w_std=np.std(er2w_all)
# er2w_std=er2w_std/2
# 
# 
# 
# er2sn_all=np.zeros((n_rec))
# for i in range(n_rec):
#     er2sn_all[i]=V2_sn[i,labels[i,1]]-V2_sn[i,labels[i,0]]
# er2sn=np.mean(er2sn_all)
# 
# er2sn_std=np.std(er2sn_all)
# er2sn_std=er2sn_std/2
# 
# er2wn_all=np.zeros((n_rec))
# for i in range(n_rec):
#     er2wn_all[i]=V2_wn[i,labels[i,1]]-V2_wn[i,labels[i,0]]
# er2wn=np.mean(er2wn_all)
# 
# er2wn_std=np.std(er2wn_all)
# er2wn_std=er2wn_std/2
# 
# 
# 
# er3s_all=np.zeros((n_rec))
# for i in range(n_rec):
#     er3s_all[i]=V3_s[i,labels[i,1]]-V3_s[i,labels[i,0]]
# er3s=np.mean(er3s_all)
# 
# er3s_std=np.std(er3s_all)
# er3s_std=er3s_std/2
# 
# er3w_all=np.zeros((n_rec))
# for i in range(n_rec):
#     er3w_all[i]=V3_w[i,labels[i,1]]-V3_w[i,labels[i,0]]
# er3w=np.mean(er3w_all)
# 
# er3w_std=np.std(er3w_all)
# er3w_std=er3w_std/2
# 
# 
# 
# 
# er3sn_all=np.zeros((n_rec))
# for i in range(n_rec):
#     er3sn_all[i]=V3_sn[i,labels[i,1]]-V3_sn[i,labels[i,0]]
# er3sn=np.mean(er3sn_all)
# 
# er3sn_std=np.std(er3sn_all)
# er3sn_std=er3sn_std/2
# 
# er3wn_all=np.zeros((n_rec))
# for i in range(n_rec):
#     er3wn_all[i]=V3_wn[i,labels[i,1]]-V3_wn[i,labels[i,0]]
# er3wn=np.mean(er3wn_all)
# 
# er3wn_std=np.std(er3wn_all)
# er3wn_std=er3wn_std/2
# =============================================================================




#####
Final=np.array([[er2s,er2w,er3s,er3w],[er2sn,er2wn,er3sn,er3wn]])
final5= 5*Final

Final_acc=np.array([[mse2_s,mse2_w,mse3_s,mse3_w],[mse2_sn,mse2_wn,mse3_sn,mse3_wn]])

final5_all_std=(5*np.array([[er2s_std,er2w_std,er3s_std,er3w_std],[er2sn_std,er2wn_std,er3sn_std,er3wn_std]]))/2
final5_all=5*np.array([[er2s,er2w,er3s,er3w],[er2sn,er2wn,er3sn,er3wn]])

final_acc_all_std=np.array([[mse2_s_std,mse2_w_std,mse3_s_std,mse3_w_std],[mse2_sn_std,mse2_wn_std,mse3_sn_std,mse3_wn_std]])
final_acc_all=np.array([[mse2_s,mse2_w,mse3_s,mse3_w],[mse2_sn,mse2_wn,mse3_sn,mse3_wn]])

###
#average mse mnist
print(np.mean(x_train[:30000]-x_train[30000:]**2)) # = 0.02
print(np.min(x_train[:30000]-x_train[30000:]**2)) # = 1

#average ambiguous images and message reconstruction and recall images (MSE)
print(np.mean(M3_s-train[:679]**2)) # = 0.07601259    #0.06384365
print(np.mean(M3_sn-train[:679]**2)) # = 0.077348374  #0.064102694
print(np.mean(M3_w-train[:679]**2)) # = 0.07598332    #0.06375574
print(np.mean(M3_wn-train[:679]**2)) # = 0.077540584  #0.06359563

print(np.mean(M2_s-train[:679]**2)) # = 0.088930525
print(np.mean(M2_sn-train[:679]**2)) # = 0.088666536
print(np.mean(M2_w-train[:679]**2)) # = 0.08884759
print(np.mean(M2_wn-train[:679]**2)) # = 0.08824396

#average ambiguous images and message reconstruction and recall images (MAE)
print(np.mean(np.abs(M3_s-train[:679]))) # = 0.109132014
print(np.mean(np.abs(M3_sn-train[:679]))) # = 0.109574065
print(np.mean(np.abs(M3_w-train[:679]))) # = 0.10901911
print(np.mean(np.abs(M3_wn-train[:679]))) # = 0.11043289

print(np.mean(np.abs(M2_s-train[:679]))) # = 0.0891552
print(np.mean(np.abs(M2_sn-train[:679]))) # = 0.093074776
print(np.mean(np.abs(M2_w-train[:679]))) # = 0.08939454
print(np.mean(np.abs(M2_wn-train[:679]))) # = 0.094227225





######
#does the model  wants to reduce ambigiousness?

# #save Zqs of ambiguous data
# priors = most_probable.reshape(-1,7,7)
# zq = codes_sampler(priors)
# np.save('ambig_data_zqs.npy',zq )
# amb_generated = decoder.predict(zq, steps=1)
# from keras.models import load_model
# classifier= load_model('Data/MNIST_keras_CNN.h5')
# amb_preds=classifier.predict(amb_generated)
# np.save('amb_preds.npy',amb_preds )

#amb_Zq= np.load('./Data/amb_mnist_zqs_non-clipped.npy')

#saving Zqs of ambiguous data (non-clipped or clipped depending on the 'clipping' flag)
# =============================================================================
# np.save('./Data/amb_data_zqs_non-clipped.npy',encoded_outputs )
# np.save('./Data/amb_data_preds_non-clipped.npy', v1)
# for i in range (10):
#     img=decoder.predict(encoded_outputs[i].reshape((1,7,7,64)))
#     plt.imshow(img.reshape(28,28), cmap='gray')
#     plt.show()
# =============================================================================



#plot valence



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


plot_valence(final5_all,final5_all_std)
#plot_valence([[1.2525, -0.555, 0.77, -0.3075], [2.294, -2.31, 0.198, 0.292]], [[0.8925, 0.9075, 0.475, 0.53875], [0.908, 0.963, 0.607, 0.6]])



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

plot_accuracy(final_acc_all,final_acc_all_std)
#acc_all_experiment=np.array([[6.2875, 7.4425, 3.655, 3.8875], [4.722, 6.798, 3.364, 3.804]])
#plot_accuracy(acc_all_experiment)

def plot_acc_experiment():
    categories = ['Message', 'Recall']
    shared_reality = [6.2875,7.4425]  
    non_shared_reality = [4.722,6.798]
    
    # Standard deviation (for error bars)
    error_shared = [3.655/2, 3.8875/2]
    error_non_shared = [3.364/2, 3.804/2]
            
    
    # Define bar positions
    x = np.arange(len(categories))  # [0, 1]
    width = 0.25  # Width of bars
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot bars
    bars1 = ax.bar(x - width/2, shared_reality, width, label="shared reality",yerr=error_shared, capsize=5, color='C0', zorder=2)
    bars2 = ax.bar(x + width/2, non_shared_reality, width, label="No shared reality",yerr=error_non_shared, capsize=5, color='C1', zorder=2)
    
    # Formatting the x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0,12)
    ax.set_xlim(-0.5,1.5)
    # Add horizontal grid lines
    ax.yaxis.grid(True,  alpha=0.7)
    
    # Set title
    ax.set_title('Recall - Experiment results')
    
    # Add legend
    ax.legend()
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('./' + date + '_figures/recall_experiment.png', bbox_inches='tight')
    plt.show()

plot_acc_experiment()



v1_sorted=np.sort(v1)



# =============================================================================
# def plot_all_phases_positive():
#     s = np.argpartition(V2_s,-2, axis=1)[:,-2:]
#     sn = np.argpartition(V2_sn,-2, axis=1)[:,-2:]
#     s3 = np.argpartition(V3_s,-2, axis=1)[:,-2:]
#     s3n = np.argpartition(V3_sn,-2, axis=1)[:,-2:]      
#     for i in range(len(train)):
#             if (v1_sorted[i][-1]< 0.65):
#                     fig_s_all, axs = plt.subplots(3,3 ,layout="constrained", figsize=(10, 10))
#                     img=decoder.predict(masked_train_X[i].reshape((1,7,7,64)))
#                     axs[0][0].imshow(img.reshape(28,28) , cmap='gray')
#                     axs[0][0].title.set_text('50% masking')
#                     axs[0][0].set_xlabel('Decoded '+r"$\bf{C1I}$", rotation=0, labelpad=5, loc='center', fontsize='large')
#                     #axs[0][0].set_ylabel(r"$\bf{C1I}$", rotation=0, labelpad=-60, loc='center', fontsize='large')
#                     axs[0][0].set_xticks([])
#                     axs[0][0].set_yticks([])
# 
#                     img=decoder.predict(train_X[i].reshape((1,7,7,64)))
#                     axs[1][0].imshow(img.reshape(28,28), cmap='gray')
#                     axs[1][0].set_xlabel('Decoded '+r"$\bf{Zq(C1)}$", rotation=0, labelpad=5, loc='center', fontsize='large')
#                     #axs[1][0].set_ylabel(r"$\bf{M1J}$" +'                    '+r"$\bf{M1}$"+' \n strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=124, loc='bottom',fontsize ='large')#+str(np.flip(labels[1])))
#                     axs[1][0].set_xticks([])
#                     axs[1][0].set_yticks([])
# 
#                     axs[0][1].imshow(M2_s[i], cmap='gray')
#                     axs[0][1].title.set_text(r"$\bf{C2J}$"+'\n  1st  ('+str(s[i][-1])+': '+str(np.round(np.sort(V2_s[i])[-1],2))+')'+'\n2nd ('+str(s[i][-2])+': '+str(np.round(np.sort(V2_s[i])[-2],2))+')')
#                     axs[0][1].set_xlabel(r"$\bf{C2I}$"+' \nShared-reality', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     #axs[0][1].set_ylabel(r"$\bf{C2I}$", labelpad=70, loc='bottom', fontsize='large')
#                     axs[0][1].set_xticks([])
#                     axs[0][1].set_yticks([])
# 
#                     axs[1][1].imshow(M2_sn[i], cmap='gray')
#                     axs[1][1].title.set_text(r"$\bf{C2J}$"+'\n  1st  ('+str(sn[i][-1])+': '+str(np.round(np.sort(V2_sn[i])[-1],2))+') '+'\n 2nd ('+str(sn[i][-2])+': '+str(np.round(np.sort(V2_sn[i])[-2],2))+')')
#                     axs[1][1].set_xlabel(r"$\bf{C2I}$"+' \nNon-shared reality', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     #axs[1][1].set_ylabel(r"$\bf{C2J}$"+'                      '+r"$\bf{C2I}$"+' \n 1st  ('+str(sn[i][-1])+': '+str(np.round(np.sort(V2_sn[i])[-1],2))+') '+'\n 2nd ('+str(sn[i][-2])+': '+str(np.round(np.sort(V2_sn[i])[-2],2))+')', rotation=0, labelpad=124, loc='bottom', fontsize='large')
#                     axs[1][1].set_xticks([])
#                     axs[1][1].set_yticks([])
#                     
#                     axs[0][2].imshow(M3_s[i], cmap='gray')
#                     axs[0][2].title.set_text(r"$\bf{C3J}$"+'\n  1st  ('+str(s3[i][-1])+': '+str(np.round(np.sort(V3_s[i])[-1],2))+') '+'\n2nd ('+str(s3[i][-2])+': '+str(np.round(np.sort(V3_s[i])[-2],2))+')')
#                     axs[0][2].set_xlabel(r"$\bf{C3I}$"+' \nShared-reality', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     #axs[0][2].set_ylabel(r"$\bf{C3J}$"+'                       '+r"$\bf{C3I}$"+' \n 1st  ('+str(s3[i][-1])+': '+str(np.round(np.sort(V3_s[i])[-1],2))+') '+'\n 2nd ('+str(s3[i][-2])+': '+str(np.round(np.sort(V3_s[i])[-2],2))+')', rotation=0, labelpad=124, loc='bottom', fontsize='large')
#                     axs[0][2].set_xticks([])
#                     axs[0][2].set_yticks([])
#                     #axarr[0].axis('off')  
#                     axs[1][2].imshow(M3_sn[i], cmap='gray')
#                     axs[1][2].title.set_text(r"$\bf{C3J}$"+'\n  1st  ('+str(s3n[i][-1])+': '+str(np.round(np.sort(V3_sn[i])[-1],2))+') '+'\n  2nd ('+str(s3n[i][-2])+': '+str(np.round(np.sort(V3_sn[i])[-2],2))+')')
#                     
#                     axs[1][2].set_xlabel(r"$\bf{C3I}$"+' \nNon-shared reality', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     #axs[1][2].set_ylabel(r"$\bf{C3J}$"+'                        '+r"$\bf{C3I}$"+' \n 1st  ('+str(s3n[i][-1])+': '+str(np.round(np.sort(V3_sn[i])[-1],2))+') '+'\n 2nd ('+str(s3n[i][-2])+': '+str(np.round(np.sort(V3_sn[i])[-2],2))+')', rotation=0, labelpad=124, loc='bottom', fontsize='large')
#                     axs[1][2].set_xticks([])
#                     axs[1][2].set_yticks([])
#                     
#                     axs[2][0].imshow(train[i], cmap='gray')
#                     axs[2][0].title.set_text(r"$\bf{M1J}$"+'\n strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')')
#                     axs[2][0].set_xlabel(r"$\bf{M1}$", rotation=0, labelpad=5, loc='center', fontsize='large')
#                     axs[2][0].set_xticks([])
#                     axs[2][0].set_yticks([])
#                     axs[2][1].axis('off')
#                     axs[2][2].axis('off')
#                     fig_s_all.suptitle(r"$\bf{Positive}$", fontsize='16')#, y=1.05)  # You can adjust the `y` parameter to position the title
#                     fig_s_all.tight_layout()
#                     fig_s_all.show()
#                     fig_s_all.savefig('./' + date + '_figures/all_positive' + str(i) + '.png', bbox_inches='tight')
# 
#     
# =============================================================================

V2_s_sorted=np.sort(V2_s)
V2_sn_sorted=np.sort(V2_sn)
V3_s_sorted=np.sort(V3_s)
V3_sn_sorted=np.sort(V3_sn)

V2_w_sorted=np.sort(V2_w)
V2_wn_sorted=np.sort(V2_wn)
V3_w_sorted=np.sort(V3_w)
V3_wn_sorted=np.sort(V3_wn)







cols = ['Input phase' +'\n\n'+ 'Decoded '+r"$\bf{C1I}$", 'Message production phase' +'\n\n'+r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$" , 'Free-recall phase' +'\n\n'+r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$"]
# =============================================================================
# def plot_all_phases_positive():
#     s2 = np.argpartition(V2_s,-2, axis=1)[:,-2:]
#     s2n = np.argpartition(V2_sn,-2, axis=1)[:,-2:]
#     s3 = np.argpartition(V3_s,-2, axis=1)[:,-2:]
#     s3n = np.argpartition(V3_sn,-2, axis=1)[:,-2:]      
#     for i in range(1):#len(train)):
#             if (v1_sorted[i][-1]< 0.65):
#                 if((V2_s_sorted[i][-1]+V2_s_sorted[i][-2])>0.8 and (V2_sn_sorted[i][-1]+V2_sn_sorted[i][-2])>0.8 and (V3_s_sorted[i][-1]+V3_s_sorted[i][-2])>0.8 and (V3_sn_sorted[i][-1]+V3_sn_sorted[i][-2])>0.8):
#                     fig_w_all, axs = plt.subplots(3,3 ,layout="constrained", figsize=(8 ,8.3))
#                     img=decoder.predict(masked_train_X[i].reshape((1,7,7,64)))
#                     axs[0][0].imshow(img.reshape(28,28) , cmap='gray')
#                     axs[0][0].title.set_text('Decoded '+r"$\bf{C1I}$")
#                     axs[0][0].set_xlabel('50% masking', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     #axs[0][0].set_ylabel(r"$\bf{C1I}$", rotation=0, labelpad=-60, loc='center', fontsize='large')
#                     axs[0][0].set_xticks([])
#                     axs[0][0].set_yticks([])
# 
#                     img=decoder.predict(train_X[i].reshape((1,7,7,64)))
#                     axs[1][0].imshow(img.reshape(28,28), cmap='gray')
#                     axs[1][0].title.set_text('Decoded '+r"$\bf{Zq(C1)}$")
#                     #axw[1][0].set_xlabel('Decoded '+r"$\bf{Zq(C1)}$", rotation=0, labelpad=5, loc='center', fontsize='large')
#                     #axs[1][0].set_ylabel(r"$\bf{M1J}$" +'                    '+r"$\bf{M1}$"+' \n strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=124, loc='bottom',fontsize ='large')#+str(np.flip(labels[1])))
#                     axs[1][0].set_xticks([])
#                     axs[1][0].set_yticks([])
# 
#                     axs[0][1].imshow(M2_s[i], cmap='gray')
#                     axs[0][1].title.set_text(r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$")
#                     axs[0][1].set_xlabel(' '+r"$\bf{C2J}$"+'  1st  ('+str(s2[i][-1])+': '+str(np.round(np.sort(V2_s[i])[-1],2))+')'+'\n         2nd ('+str(s2[i][-2])+': '+str(np.round(np.sort(V2_s[i])[-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     axs[0][1].set_ylabel('Shared reality', rotation=90, fontsize='large')
#                     axs[0][1].set_xticks([])
#                     axs[0][1].set_yticks([])
# 
#                     axs[1][1].imshow(M2_sn[i], cmap='gray')
#                     axs[1][1].title.set_text(r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$")
#                     axs[1][1].set_xlabel(' '+r"$\bf{C2J}$"+'  1st  ('+str(s2n[i][-1])+': '+str(np.round(np.sort(V2_sn[i])[-1],2))+') '+'\n         2nd ('+str(s2n[i][-2])+': '+str(np.round(np.sort(V2_sn[i])[-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     axs[1][1].set_ylabel('Non-shared reality', rotation=90, fontsize='large')
#                     axs[1][1].set_xticks([])
#                     axs[1][1].set_yticks([])
#                     
#                     axs[0][2].imshow(M3_s[i], cmap='gray')
#                     axs[0][2].title.set_text(r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$")
#                     axs[0][2].set_xlabel(' '+r"$\bf{C3J}$"+'  1st  ('+str(s3[i][-1])+': '+str(np.round(np.sort(V3_s[i])[-1],2))+') '+'\n         2nd ('+str(s3[i][-2])+': '+str(np.round(np.sort(V3_s[i])[-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     axs[0][2].set_ylabel('Shared reality', rotation=90, fontsize='large')
#                     axs[0][2].set_xticks([])
#                     axs[0][2].set_yticks([])
#                     #axarr[0].axis('off')  
#                     axs[1][2].imshow(M3_sn[i], cmap='gray')
#                     axs[1][2].title.set_text(r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$")
#                     axs[1][2].set_xlabel(' '+r"$\bf{C3J}$"+'  1st  ('+str(s3n[i][-1])+': '+str(np.round(np.sort(V3_sn[i])[-1],2))+') '+'\n         2nd ('+str(s3n[i][-2])+': '+str(np.round(np.sort(V3_sn[i])[-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     axs[1][2].set_ylabel('Non-shared reality', rotation=90, fontsize='large')
#                     axs[1][2].set_xticks([])
#                     axs[1][2].set_yticks([])
#                     
#                     axs[2][0].imshow(train[i], cmap='gray')
#                     axs[2][0].title.set_text(r"$\bf{M1}$")
#                     axs[2][0].set_xlabel('    '+r"$\bf{M1J}$"+'  strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n             weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     axs[2][0].set_xticks([])
#                     axs[2][0].set_yticks([])
#                     axs[2][1].axis('off')
#                     axs[2][2].axis('off')
#                     
#                     for ax, col in zip(axs[0], cols):
#                         ax.set_title(col, fontsize='large')
#                     
#                     fig_w_all.suptitle(r"$\bf{Positive}$", fontsize='14')#, y=1.05)  # You can adjust the `y` parameter to position the title
# 
#                     fig_w_all.tight_layout()
#                     fig_w_all.show()
#                     fig_w_all.savefig('./' + date + '_figures/all_positive' + str(i) + '.png', bbox_inches='tight')
# 
# =============================================================================
img_70=[70]
img_228=[228]
def plot_all_phases_positive_corrected():
    s2 = np.argpartition(V2_s,-2, axis=1)[:,-2:]
    s2n = np.argpartition(V2_sn,-2, axis=1)[:,-2:]
    s3 = np.argpartition(V3_s,-2, axis=1)[:,-2:]
    s3n = np.argpartition(V3_sn,-2, axis=1)[:,-2:]
    w2 = np.argpartition(V2_w,-2, axis=1)[:,-2:]
    w2n = np.argpartition(V2_wn,-2, axis=1)[:,-2:]
    w3 = np.argpartition(V3_w,-2, axis=1)[:,-2:]
    w3n = np.argpartition(V3_wn,-2, axis=1)[:,-2:]         
    for i in range(8,10,1):#filtered_idxs[50:80]:
        #if (v1_sorted[i][-1]< 0.65):
# =============================================================================
#             if(labels[i,1]==eval2_s[i]==eval2_sn[i]==eval3_s[i]==eval3_sn[i]):
#                 #if(V2_s[i,labels[i,1]]>V3_s[i,labels[i,1]]) and (V2_sn[i,labels[i,1]]>V3_sn[i,labels[i,1]]):
#                     if(labels[i,0]==eval2_w[i]==eval2_wn[i]):# and w3[i][-1]<0.6 and w3n[i][-1]<0.6):  
# =============================================================================
                        fig_w_all, axs = plt.subplots(3,3 ,layout="constrained", figsize=(8 ,8.3))
                        img=decoder.predict(masked_train_X[i].reshape((1,7,7,64)))
                        axs[0][0].imshow(img.reshape(28,28) , cmap='gray')
                        axs[0][0].title.set_text('Decoded '+r"$\bf{C1I}$")
                        axs[0][0].set_xlabel('50% masking', rotation=0, labelpad=5, loc='center', fontsize='large')
                        #axs[0][0].set_ylabel(r"$\bf{C1I}$", rotation=0, labelpad=-60, loc='center', fontsize='large')
                        axs[0][0].set_xticks([])
                        axs[0][0].set_yticks([])
    
                        img=decoder.predict(train_X[i].reshape((1,7,7,64)))
                        axs[1][0].imshow(img.reshape(28,28), cmap='gray')
                        axs[1][0].title.set_text('Decoded '+r"$\bf{Zq}$"+' '+r"$\bf{(C1)}$")
                        #axw[1][0].set_xlabel('Decoded '+r"$\bf{Zq(C1)}$", rotation=0, labelpad=5, loc='center', fontsize='large')
                        #axs[1][0].set_ylabel(r"$\bf{M1J}$" +'                    '+r"$\bf{M1}$"+' \n strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=124, loc='bottom',fontsize ='large')#+str(np.flip(labels[1])))
                        axs[1][0].set_xticks([])
                        axs[1][0].set_yticks([])
    
                        axs[0][1].imshow(M2_s[i], cmap='gray')
                        axs[0][1].title.set_text(r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$")
                        axs[0][1].set_xlabel(r"$\bf{C2J}$"+' ('+str(labels[i,1])+': '+str(np.round(V2_s[i,labels[i,1]],2))+')'+'\n       ('+str(labels[i,0])+': '+str(np.round(V2_s[i,labels[i,0]],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axs[0][1].set_ylabel('Shared reality', rotation=90, fontsize='large')
                        axs[0][1].set_xticks([])
                        axs[0][1].set_yticks([])
    
                        axs[1][1].imshow(M2_sn[i], cmap='gray')
                        axs[1][1].title.set_text(r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$")
                        axs[1][1].set_xlabel(r"$\bf{C2J}$"+' ('+str(labels[i,1])+': '+str(np.round(V2_sn[i,labels[i,1]],2))+')'+'\n       ('+str(labels[i,0])+': '+str(np.round(V2_sn[i,labels[i,0]],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axs[1][1].set_ylabel('Non-shared reality', rotation=90, fontsize='large')
                        axs[1][1].set_xticks([])
                        axs[1][1].set_yticks([])
                        
                        axs[0][2].imshow(M3_s[i], cmap='gray')
                        axs[0][2].title.set_text(r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$")
                        axs[0][2].set_xlabel(r"$\bf{C3J}$"+' ('+str(labels[i,1])+': '+str(np.round(V3_s[i,labels[i,1]],2))+')'+'\n        ('+str(labels[i,0])+': '+str(np.round(V3_s[i,labels[i,0]],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axs[0][2].set_ylabel('Shared reality', rotation=90, fontsize='large')
                        axs[0][2].set_xticks([])
                        axs[0][2].set_yticks([])
                        #axarr[0].axis('off')  
                        axs[1][2].imshow(M3_sn[i], cmap='gray')
                        axs[1][2].title.set_text(r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$")
                        axs[1][2].set_xlabel(r"$\bf{C3J}$"+' ('+str(labels[i,1])+': '+str(np.round(V3_sn[i,labels[i,1]],2))+')'+'\n       ('+str(labels[i,0])+': '+str(np.round(V3_sn[i,labels[i,0]],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axs[1][2].set_ylabel('Non-shared reality', rotation=90, fontsize='large')
                        axs[1][2].set_xticks([])
                        axs[1][2].set_yticks([])
                        
                        axs[2][0].imshow(train[i], cmap='gray')
                        axs[2][0].title.set_text(r"$\bf{M1}$")
                        axs[2][0].set_xlabel(r"$\bf{M1J}$"+' strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n       weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axs[2][0].set_xticks([])
                        axs[2][0].set_yticks([])
                        axs[2][1].axis('off')
                        axs[2][2].axis('off')
                        
                        for ax, col in zip(axs[0], cols):
                            ax.set_title(col, fontsize='large')
                        
                        fig_w_all.suptitle(r"$\bf{Positive}$", fontsize='16')#, y=1.05)  # You can adjust the `y` parameter to position the title
    
                        fig_w_all.tight_layout()
                        fig_w_all.show()
                        fig_w_all.savefig('./' + date + '_figures/' + str(i) + 'all_positive.png', bbox_inches='tight')

plot_all_phases_positive_corrected()





# =============================================================================
# def plot_all_phases_negative():
#     w2 = np.argpartition(V2_w,-2, axis=1)[:,-2:]
#     w2n = np.argpartition(V2_wn,-2, axis=1)[:,-2:]
#     w3 = np.argpartition(V3_w,-2, axis=1)[:,-2:]
#     w3n = np.argpartition(V3_wn,-2, axis=1)[:,-2:]      
#     for i in range(1):#len(train)):
#             if (v1_sorted[i][-1]< 0.65):
#                 if((V2_s_sorted[i][-1]+V2_s_sorted[i][-2])>0.8 and (V2_sn_sorted[i][-1]+V2_sn_sorted[i][-2])>0.8 and (V3_s_sorted[i][-1]+V3_s_sorted[i][-2])>0.8 and (V3_sn_sorted[i][-1]+V3_sn_sorted[i][-2])>0.8):
#                     #if((V2_w_sorted[i][-1]+V2_w_sorted[i][-2])>0.8 and (V2_wn_sorted[i][-1]+V2_wn_sorted[i][-2])>0.8 and (V3_w_sorted[i][-1]+V3_w_sorted[i][-2])>0.8 and (V3_wn_sorted[i][-1]+V3_wn_sorted[i][-2])>0.8):
#     
#                         
#                         fig_w_all, axw = plt.subplots(3,3 ,layout="constrained", figsize=(8 ,8.3))
#                         img=decoder.predict(masked_train_X[i].reshape((1,7,7,64)))
#                         axw[0][0].imshow(img.reshape(28,28) , cmap='gray')
#                         axw[0][0].title.set_text('Decoded '+r"$\bf{C1I}$")
#                         axw[0][0].set_xlabel('50% masking', rotation=0, labelpad=5, loc='center', fontsize='large')
#                         #axs[0][0].set_ylabel(r"$\bf{C1I}$", rotation=0, labelpad=-60, loc='center', fontsize='large')
#                         axw[0][0].set_xticks([])
#                         axw[0][0].set_yticks([])
#     
#                         img=decoder.predict(train_X[i].reshape((1,7,7,64)))
#                         axw[1][0].imshow(img.reshape(28,28), cmap='gray')
#                         axw[1][0].title.set_text('Decoded '+r"$\bf{Zq(C1)}$")
#                         #axw[1][0].set_xlabel('Decoded '+r"$\bf{Zq(C1)}$", rotation=0, labelpad=5, loc='center', fontsize='large')
#                         #axs[1][0].set_ylabel(r"$\bf{M1J}$" +'                    '+r"$\bf{M1}$"+' \n strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=124, loc='bottom',fontsize ='large')#+str(np.flip(labels[1])))
#                         axw[1][0].set_xticks([])
#                         axw[1][0].set_yticks([])
#     
#                         axw[0][1].imshow(M2_w[i], cmap='gray')
#                         axw[0][1].title.set_text(r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$")
#                         axw[0][1].set_xlabel('     '+r"$\bf{C2J}$"+'  1st  ('+str(w2[i][-1])+': '+str(np.round(np.sort(V2_w[i])[-1],2))+')'+'\n             2nd ('+str(w2[i][-2])+': '+str(np.round(np.sort(V2_w[i])[-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                         axw[0][1].set_ylabel('Shared reality', rotation=90, fontsize='large')
#                         axw[0][1].set_xticks([])
#                         axw[0][1].set_yticks([])
#     
#                         axw[1][1].imshow(M2_wn[i], cmap='gray')
#                         axw[1][1].title.set_text(r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$")
#                         axw[1][1].set_xlabel('     '+r"$\bf{C2J}$"+'  1st  ('+str(w2n[i][-1])+': '+str(np.round(np.sort(V2_wn[i])[-1],2))+') '+'\n            2nd ('+str(w2n[i][-2])+': '+str(np.round(np.sort(V2_wn[i])[-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                         axw[1][1].set_ylabel('Non-shared reality', rotation=90, fontsize='large')
#                         axw[1][1].set_xticks([])
#                         axw[1][1].set_yticks([])
#                         
#                         axw[0][2].imshow(M3_w[i], cmap='gray')
#                         axw[0][2].title.set_text(r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$")
#                         axw[0][2].set_xlabel('     '+r"$\bf{C3J}$"+'  1st  ('+str(w3[i][-1])+': '+str(np.round(np.sort(V3_w[i])[-1],2))+') '+'\n            2nd ('+str(w3[i][-2])+': '+str(np.round(np.sort(V3_w[i])[-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                         axw[0][2].set_ylabel('Shared reality', rotation=90, fontsize='large')
#                         axw[0][2].set_xticks([])
#                         axw[0][2].set_yticks([])
#                         #axarr[0].axis('off')  
#                         axw[1][2].imshow(M3_wn[i], cmap='gray')
#                         axw[1][2].title.set_text(r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$")
#                         axw[1][2].set_xlabel('     '+r"$\bf{C3J}$"+'  1st  ('+str(w3n[i][-1])+': '+str(np.round(np.sort(V3_wn[i])[-1],2))+') '+'\n            2nd ('+str(w3n[i][-2])+': '+str(np.round(np.sort(V3_wn[i])[-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                         axw[1][2].set_ylabel('Non-shared reality', rotation=90, fontsize='large')
#                         axw[1][2].set_xticks([])
#                         axw[1][2].set_yticks([])
#                         
#                         axw[2][0].imshow(train[i], cmap='gray')
#                         axw[2][0].title.set_text(r"$\bf{M1}$")
#                         axw[2][0].set_xlabel('        '+r"$\bf{M1J}$"+'  strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n                weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
#                         axw[2][0].set_xticks([])
#                         axw[2][0].set_yticks([])
#                         axw[2][1].axis('off')
#                         axw[2][2].axis('off')
#                         
#                         for ax, col in zip(axw[0], cols):
#                             ax.set_title(col, fontsize='large')
#                         
#                         fig_w_all.suptitle(r"$\bf{Negative}$", fontsize='16')#, y=1.05)  # You can adjust the `y` parameter to position the title
#     
#                         fig_w_all.tight_layout()
#                         fig_w_all.show()
#                         fig_w_all.savefig('./' + date + '_figures/all_negative' + str(i) + '.png', bbox_inches='tight')
# 
# =============================================================================
def plot_all_phases_negative_corrected():
    w2 = np.argpartition(V2_w,-2, axis=1)[:,-2:]
    w2n = np.argpartition(V2_wn,-2, axis=1)[:,-2:]
    w3 = np.argpartition(V3_w,-2, axis=1)[:,-2:]
    w3n = np.argpartition(V3_wn,-2, axis=1)[:,-2:]      
    for i in range(8,10,1):#filtered_idxs[50:80]:
        #if (v1_sorted[i][-1]< 0.65):
# =============================================================================
#            # if(labels[i,1]==eval2_s[i]==eval2_sn[i]==eval3_s[i]==eval3_sn[i]):
#                 #if(labels[i,0]==eval2_w[i]==eval2_wn[i]==eval3_w[i] and labels[i,1]==eval3_wn[i]):    
#                 if(V2_w[i,labels[i,0]]>V3_w[i,labels[i,0]] and V2_wn[i,labels[i,0]]>V3_wn[i,labels[i,0]]):
#                         
#                     if(labels[i,0]==eval2_w[i]==eval2_wn[i]):# and w3[i][-1]<0.6 and w3n[i][-1]<0.6):    
#              
# =============================================================================
                        fig_w_all, axw = plt.subplots(3,3 ,layout="constrained", figsize=(8 ,8.3))
                        img=decoder.predict(masked_train_X[i].reshape((1,7,7,64)))
                        axw[0][0].imshow(img.reshape(28,28) , cmap='gray')
                        axw[0][0].title.set_text('Decoded '+r"$\bf{C1I}$")
                        axw[0][0].set_xlabel('50% masking', rotation=0, labelpad=5, loc='center', fontsize='large')
                        #axs[0][0].set_ylabel(r"$\bf{C1I}$", rotation=0, labelpad=-60, loc='center', fontsize='large')
                        axw[0][0].set_xticks([])
                        axw[0][0].set_yticks([])
    
                        img=decoder.predict(train_X[i].reshape((1,7,7,64)))
                        axw[1][0].imshow(img.reshape(28,28), cmap='gray')
                        axw[1][0].title.set_text('Decoded '+r"$\bf{Zq}$"+' '+r"$\bf{(C1)}$")
                        #axw[1][0].set_xlabel('Decoded '+r"$\bf{Zq(C1)}$", rotation=0, labelpad=5, loc='center', fontsize='large')
                        #axs[1][0].set_ylabel(r"$\bf{M1J}$" +'                    '+r"$\bf{M1}$"+' \n strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=124, loc='bottom',fontsize ='large')#+str(np.flip(labels[1])))
                        axw[1][0].set_xticks([])
                        axw[1][0].set_yticks([])
    
                        axw[0][1].imshow(M2_w[i], cmap='gray')
                        axw[0][1].title.set_text(r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$")
                        axw[0][1].set_xlabel(r"$\bf{C2J}$"+' ('+str(labels[i,1])+': '+str(np.round(V2_w[i,labels[i,1]],2))+')'+'\n       ('+str(labels[i,0])+': '+str(np.round(V2_w[i,labels[i,0]],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axw[0][1].set_ylabel('Shared reality', rotation=90, fontsize='large')
                        axw[0][1].set_xticks([])
                        axw[0][1].set_yticks([])
    
                        axw[1][1].imshow(M2_wn[i], cmap='gray')
                        axw[1][1].title.set_text(r"$\bf{C2I}$" +' / '+ r"$\bf{M2}$")
                        axw[1][1].set_xlabel(r"$\bf{C2J}$"+' ('+str(labels[i,1])+': '+str(np.round(V2_wn[i,labels[i,1]],2))+')'+'\n       ('+str(labels[i,0])+': '+str(np.round(V2_wn[i,labels[i,0]],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axw[1][1].set_ylabel('Non-shared reality', rotation=90, fontsize='large')
                        axw[1][1].set_xticks([])
                        axw[1][1].set_yticks([])
                        
                        axw[0][2].imshow(M3_w[i], cmap='gray')
                        axw[0][2].title.set_text(r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$")
                        axw[0][2].set_xlabel(r"$\bf{C3J}$"+' ('+str(labels[i,1])+': '+str(np.round(V3_w[i,labels[i,1]],2))+')'+'\n         ('+str(labels[i,0])+': '+str(np.round(V3_w[i,labels[i,0]],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axw[0][2].set_ylabel('Shared reality', rotation=90, fontsize='large')
                        axw[0][2].set_xticks([])
                        axw[0][2].set_yticks([])
                        #axarr[0].axis('off')  
                        axw[1][2].imshow(M3_wn[i], cmap='gray')
                        axw[1][2].title.set_text(r"$\bf{C3I}$" +' / '+ r"$\bf{M3}$")
                        axw[1][2].set_xlabel(r"$\bf{C3J}$"+' ('+str(labels[i,1])+': '+str(np.round(V3_wn[i,labels[i,1]],2))+')'+'\n        ('+str(labels[i,0])+': '+str(np.round(V3_wn[i,labels[i,0]],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axw[1][2].set_ylabel('Non-shared reality', rotation=90, fontsize='large')
                        axw[1][2].set_xticks([])
                        axw[1][2].set_yticks([])
                        
                        axw[2][0].imshow(train[i], cmap='gray')
                        axw[2][0].title.set_text(r"$\bf{M1}$")
                        axw[2][0].set_xlabel(r"$\bf{M1J}$"+' strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n       weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=5, loc='center', fontsize='large')
                        axw[2][0].set_xticks([])
                        axw[2][0].set_yticks([])
                        axw[2][1].axis('off')
                        axw[2][2].axis('off')
                        
                        for ax, col in zip(axw[0], cols):
                            ax.set_title(col, fontsize='large')
                        
                        fig_w_all.suptitle(r"$\bf{Negative}$", fontsize='16')#, y=1.05)  # You can adjust the `y` parameter to position the title
    
                        fig_w_all.tight_layout()
                        fig_w_all.show()
                        fig_w_all.savefig('./' + date + '_figures/' + str(i) + 'all_negative.png', bbox_inches='tight')

    
plot_all_phases_negative_corrected()












def plot_input_phase():
    for i in range(len(train)):
            if (v1_sorted[i][-1]< 0.65):
                #if (v1_v2_s[i]==evals1[i,0] and v1_v2_sn[i]==evals1[i,1]):
                    fig1, ax = plt.subplots(3,1, figsize=(5, 5))
                    img=decoder.predict(masked_train_X[i].reshape((1,7,7,64)))
                    ax[0].imshow(img.reshape(28,28) , cmap='gray')
                    #ax[0].title.set_text(r"$\bf{A2J}$" +' (Audience) \n     (' +str(evals1[i][0])+': '+str('1.0) \n \n'+'Masking: 0.5 \n'))#('VA1 ('+str(VA1[i][1])+': '+str(VM1_percentage[-1])+') '+'('+str(VA1[i][0])+': '+str(VM1_percentage[-2])+')\n \n'+'EC1 \n VC1 ('+str(VC1[i][1])+': '+str(VC1_EC1_percentage[-1])+') '+'('+str(VC1[i][0])+': '+str(VC1_EC1_percentage[-2])+')')
                    ax[0].set_xlabel('Decoded communicator trace', rotation=0, labelpad=5, loc='center', fontsize='large')
                    ax[0].set_ylabel(r"$\bf{C1I}$", rotation=0, labelpad=-85, loc='center', fontsize='large')
                    #ax[0].set_ylabel('\n \n \n \n'+ r"$\bf{VA1}$" +' (Audience) \n weak (' +str(VC1[i][0])+': '+str(VC1_weak[i][VC1[i][0]]).replace(' [', '').replace('[', '').replace(']', '')+')', rotation=0, labelpad=124, loc='top')
                    ax[0].set_xticks([])
                    ax[0].set_yticks([])
                    #ax[0].axis('off')
                    img=decoder.predict(train_X[i].reshape((1,7,7,64)))
                    ax[1].imshow(img.reshape(28,28), cmap='gray')#train_X[i], cmap='gray')
                    ax[1].set_title('Decoded cortical representation', pad=0, y=-0.2, loc='center', fontsize='large')
                    ax[1].set_ylabel(r"$\bf{Zq(C1)}$", rotation=0, labelpad=-97, loc='center', fontsize='large')
                    ax[1].set_xticks([])
                    ax[1].set_yticks([])
                    #ax[1].axis('off')
# =============================================================================
#                     ax[2].imshow(M2_s[303], cmap='gray')
#                     ax[2].set_title('Message', pad=0, y=-0.2, loc='center')
#                     ax[2].set_ylabel(r"$\bf{M2J}$"+' (image classifier) \n 1st  ('+str(7)+': '+str(np.round(np.sort(V2_s[303])[-1],2))+') '+'\n 2nd ('+str(1)+': '+str(np.round(np.sort(V2_s[303])[-2],2))+')', rotation=0, labelpad=124, loc='bottom')
#                     ax[2].set_xticks([])
#                     ax[2].set_yticks([])
# =============================================================================
                    ax[2].imshow(train[i], cmap='gray')
                    ax[2].set_title('Message', pad=0, y=-0.2, loc='center')
                    ax[2].set_ylabel(r"$\bf{M1J}$" +'                                              '+r"$\bf{M1}$"+' \n strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=124, loc='bottom',fontsize ='large')#+str(np.flip(labels[1])))
                    #ax[2].set_ylabel(r"$\bf{M1}$", rotation=0, labelpad=-85, loc='center')
                    ax[2].set_xticks([])
                    ax[2].set_yticks([])
                    #ax[2].axis('off')
                    #fig1.tight_layout()
                    #fig1.supxlabel('VM1 ('+str(VM1[i][1])+': '+str(VM1_percentage[-1])+') '+'('+str(VM1[i][0])+': '+str(VM1_percentage[-2])+')')#+str(np.flip(labels[1])))
                    fig1.subplots_adjust(hspace=0.5)
                    fig1.show()
                    fig1.savefig('./'+date+'_figures/input_img'+str(i)+'.png', bbox_inches='tight')
# =============================================================================
#                 if (v1_v2_sn[i]==evals1[i,1]):
#                     fig1, ax = plt.subplots(3,1, figsize=(5, 5))
#                     img=decoder.predict(masked_train_X[i].reshape((1,7,7,64)))
#                     ax[0].imshow(img.reshape(28,28) , cmap='gray')
#                     ax[0].title.set_text(r"$\bf{A2J}$" +' (Audience) \n     (' +str(evals1[i][0])+': '+str('1.0) \n \n'+'Masking: 0.5 \n'))#('VA1 ('+str(VA1[i][1])+': '+str(VM1_percentage[-1])+') '+'('+str(VA1[i][0])+': '+str(VM1_percentage[-2])+')\n \n'+'EC1 \n VC1 ('+str(VC1[i][1])+': '+str(VC1_EC1_percentage[-1])+') '+'('+str(VC1[i][0])+': '+str(VC1_EC1_percentage[-2])+')')
#                     ax[0].set_xlabel('Communicator trace', rotation=0, labelpad=5, loc='center', fontsize='large')
#                     ax[0].set_ylabel(r"$\bf{C1I}$", rotation=0, labelpad=-85, loc='center')
#                     #ax[0].set_ylabel('\n \n \n \n'+ r"$\bf{VA1}$" +' (Audience) \n weak (' +str(VC1[i][0])+': '+str(VC1_weak[i][VC1[i][0]]).replace(' [', '').replace('[', '').replace(']', '')+')', rotation=0, labelpad=124, loc='top')
#                     ax[0].set_xticks([])
#                     ax[0].set_yticks([])
#                     #ax[0].axis('off')
#                     img=decoder.predict(train_X[i].reshape((1,7,7,64)))
#                     ax[1].imshow(img.reshape(28,28), cmap='gray')#train_X[i], cmap='gray')
#                     ax[1].set_title('Cortical representation', pad=0, y=-0.2, loc='center')
#                     ax[1].set_xticks([])
#                     ax[1].set_yticks([])
#                     #ax[1].axis('off')
#                     ax[2].imshow(train[i], cmap='gray')
#                     ax[2].set_title('Message', pad=0, y=-0.2, loc='center')
#                     ax[2].set_ylabel(r"$\bf{M1J}$" +' (image classifier)                            '+r"$\bf{M1}$"+' \n strong ('+str(evals1[i][1])+': '+str(np.round(v1_sorted[i][-1],2))+') '+'\n weak   ('+str(evals1[i][0])+': '+str(np.round(v1_sorted[i][-2],2))+')', rotation=0, labelpad=124, loc='bottom')#+str(np.flip(labels[1])))
#                     #ax[2].set_ylabel(r"$\bf{M1}$", rotation=0, labelpad=-85, loc='center')
#                     ax[2].set_xticks([])
#                     ax[2].set_yticks([])
#                     #ax[2].axis('off')
#                     #fig1.tight_layout()
#                     #fig1.supxlabel('VM1 ('+str(VM1[i][1])+': '+str(VM1_percentage[-1])+') '+'('+str(VM1[i][0])+': '+str(VM1_percentage[-2])+')')#+str(np.flip(labels[1])))
#                     fig1.subplots_adjust(hspace=0.5)
#                     fig1.show()
#                     fig1.savefig('./'+date+'_figures/non-shared-reality/img'+str(i)+'.png', bbox_inches='tight')
# 
# =============================================================================
    
#plot input phase
#plot_input_phase()
  


def plot_message_reconstruction_phase():
    s = np.argpartition(V2_s,-2, axis=1)[:,-2:]
    sn = np.argpartition(V2_sn,-2, axis=1)[:,-2:]        
    for i in range(len(train)):
        if (v1_sorted[i][-1]< 0.65):
            #if (v1_v2_s[i]==evals1[i,0] and v1_v2_sn[i]==evals1[i,1]):

                fig, axarr = plt.subplots(2,1)
                axarr[0].imshow(M2_s[i], cmap='gray')
                axarr[0].title.set_text(r"$\bf{A2J}$" +' (Audience) \n     (' +str(evals1[i][0])+': '+str('1.0)'))
                axarr[0].set_title('Shared-reality', pad=0, y=-0.15, loc='center')
                axarr[0].set_ylabel(r"$\bf{C2J}$"+'                                                    '+r"$\bf{C2I}$"+' \n 1st  ('+str(s[i][-1])+': '+str(np.round(np.sort(V2_s[i])[-1],2))+') '+'\n 2nd ('+str(s[i][-2])+': '+str(np.round(np.sort(V2_s[i])[-2],2))+')', rotation=0, labelpad=124, loc='bottom', fontsize='large')
                axarr[0].set_xticks([])
                axarr[0].set_yticks([])
                #axarr[0].axis('off')
                fig.subplots_adjust(hspace=0.5)
                fig.show()
                #fig.savefig('./'+date+'_figures/shared-reality/img'+str(i)+'.png', bbox_inches='tight')                

                axarr[1].imshow(M2_sn[i], cmap='gray')
                axarr[1].set_title('Non-shared reality', pad=0, y=-0.15, loc='center')
                axarr[1].set_ylabel(r"$\bf{C2J}$"+'                                                    '+r"$\bf{C2I}$"+' \n 1st  ('+str(sn[i][-1])+': '+str(np.round(np.sort(V2_sn[i])[-1],2))+') '+'\n 2nd ('+str(sn[i][-2])+': '+str(np.round(np.sort(V2_sn[i])[-2],2))+')', rotation=0, labelpad=124, loc='bottom', fontsize='large')
                axarr[1].set_xticks([])
                axarr[1].set_yticks([])
                #axarr[0].axis('off')
                fig.subplots_adjust(hspace=0.5)
                fig.show()
                fig.savefig('./'+date+'_figures/message-reconstruction_img'+str(i)+'.png', bbox_inches='tight')                
    
# =============================================================================
# plt.imshow(M2_s[70], cmap='gray')#[303]
# plt.title('Message', pad=0, y=-0.2, loc='center')
# plt.ylabel(r"$\bf{M2J}$"+' (image classifier) \n 1st  ('+str(7)+': '+str(np.round(np.sort(V2_s[303])[-1],2))+') '+'\n 2nd ('+str(1)+': '+str(np.round(np.sort(V2_s[303])[-2],2))+')', rotation=0, labelpad=124, loc='bottom')
# plt.xticks([])
# plt.yticks([])
# 
# =============================================================================

#plot message reconstruction phase
#plot_message_reconstruction_phase()


def plot_recall():
    s = np.argpartition(V3_s,-2, axis=1)[:,-2:]
    sn = np.argpartition(V3_sn,-2, axis=1)[:,-2:]   
    for i in range(len(train)):
            if (v1_sorted[i][-1]< 0.65):
                #if (v1_v2_s[i]==evals1[i,0] and v1_v2_sn[i]==evals1[i,1]):
                    #if(s[i][1]==evals1[i][0] and sn[i][1]==evals1[i][1]):
                        fig3, axarr = plt.subplots(2,1)
                        axarr[0].imshow(M3_s[i], cmap='gray')
                        axarr[0].set_title('Shared-reality', pad=0, y=-0.15, loc='center')
                        axarr[0].set_ylabel(r"$\bf{C3J}$"+'                                                    '+r"$\bf{C3I}$"+' \n 1st  ('+str(s[i][-1])+': '+str(np.round(np.sort(V3_s[i])[-1],2))+') '+'\n 2nd ('+str(s[i][-2])+': '+str(np.round(np.sort(V3_s[i])[-2],2))+')', rotation=0, labelpad=124, loc='bottom', fontsize='large')
                        axarr[0].set_xticks([])
                        axarr[0].set_yticks([])
                        #axarr[0].axis('off')  
                        axarr[1].imshow(M3_sn[i], cmap='gray')
                        axarr[1].set_title('Non-shared reality', pad=0, y=-0.15, loc='center')
                        axarr[1].set_ylabel(r"$\bf{C3J}$"+'                                                    '+r"$\bf{C3I}$"+' \n 1st  ('+str(sn[i][-1])+': '+str(np.round(np.sort(V3_sn[i])[-1],2))+') '+'\n 2nd ('+str(sn[i][-2])+': '+str(np.round(np.sort(V3_sn[i])[-2],2))+')', rotation=0, labelpad=124, loc='bottom', fontsize='large')
                        axarr[1].set_xticks([])
                        axarr[1].set_yticks([])
                        #axarr[0].axis('off')
                        fig3.subplots_adjust(hspace=0.5)
                        fig3.show()
                        fig3.savefig('./'+date+'_figures/recall_img'+str(i)+'.png', bbox_inches='tight')                
            

#plot_recall()


# Parameters
num_rows = 679  # Number of rows in the array
num_classes = 10  # Number of classes

# Create an array where the first class decreases from 100% to 0%
# and the last class increases from 0% to 100%
logits = np.zeros((num_rows, num_classes))

# Gradual change from 100% to 0% for the first class, and 0% to 100% for the last class
for i in range(num_rows):
    logits[i, 0] = 100 - (100 * i / (num_rows - 1))  # Decrease for class 1
    logits[i, -1] = 100 * i / (num_rows - 1)  # Increase for class 10

# Softmax function to convert logits to probabilities
def softmax(logits):
    # Subtracting the max for numerical stability
    logits_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)
    return probabilities

# Apply the softmax function to the logits
softmax_output = softmax(logits)

softmax_output[:, [0, 2]] = softmax_output[:, [2, 0]]
softmax_output[:, [8, 9]] = softmax_output[:, [9, 8]]

images=[]
acc=[]
num=[]
acc_all=[]
for i in range(310,370,2):#(320,360,2):
#for i in range(softmax_output.shape[0]):
    trace,_=recons(train_X[255], train_y, 0.7, softmax_output[i], 1)#masked_train_X[390]
    image=retrieve(trace)
    images.append(image)
    #num.append(np.argmax(softmax_output[i]))
    #acc.append(softmax_output[i][np.argmax(softmax_output[i])])
    #acc_all.append(softmax_output[i][np.argmax(softmax_output[i])])
    num.append(np.argmax(classifier.predict(image)))
    acc_all.append(np.max(classifier.predict(image)))

fig, axarr = plt.subplots(1, len(images), figsize=(len(images) * 2, 2)) 
for i in range(len(images)):
    if (i==0):
        axarr[i].imshow(np.reshape(train[255], (28,28)), cmap='gray')
        axarr[i].axis('off') 
    else:
        axarr[i].imshow(np.reshape(images[i], (28,28)), cmap='gray')
        axarr[i].axis('off')  # Hide axis for better visualization

# Adjust layout and show the figure
plt.subplots_adjust(wspace=0.1)  # Adjust horizontal space between subplots
plt.show()


#plt.bar(range(len(acc)),acc)
plt.bar(range(len(acc_all)),acc_all)
plt.plot(softmax_output[:,2], acc_all)
plt.plot(softmax_output[:,8], acc_all)
plt.xticks(ticks=range(len(num)), labels=num)
plt.ylabel('Probability')
plt.xlabel('Class of highest propability')
#plt.title('Audience tuning effect on a number image')

plt.bar(range(len(result)), result, color='lightblue')

# Plot a line above the bars
plt.plot(range(len(result)), result, color='red', marker='o', linestyle='-', linewidth=2, label="Line Above Bars")


def moving_average(data, window_size):
    """
    Compute the moving average of a given list/array with a specific window size.

    Parameters:
    data (array-like): Input data (list or numpy array).
    window_size (int): The size of the moving window.

    Returns:
    numpy array: Array containing the moving averages.
    """
    # Ensure the data is a numpy array
    data = np.array(data)
    
    # Check if window_size is valid
    if window_size <= 0 or window_size > len(data):
        raise ValueError("Window size must be between 1 and the length of the data.")
    
    # Compute the moving average using numpy's convolve function
    weights = np.ones(window_size) / window_size
    moving_avg = np.convolve(data, weights, mode='valid')  # 'valid' ensures no boundary effects

    return moving_avg


result = moving_average(acc_all,150)
print(result)
x = np.linspace(0, 10, 430)
plt.plot(result)
plt.fill_between(x, result, color='blue', alpha=0.3)  # Adjust alpha for transparency

# =============================================================================
# import seaborn as sns 
# sns.kdeplot(acc, shade=True, color='blue')
# 
# # Adding labels and title
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.title('Continuous Probability Distribution (KDE)')
# 
# # Show the plot
# plt.show()
# =============================================================================
# =============================================================================
# mask_perc_list = [0.3,0.5,0.7,0.8,0.95]
# mask_token = 0  # does 0 make sense?
# error=[]
# 
# for mask_perc in mask_perc_list:
#     mask_test = np.random.default_rng().choice([True, False], size=(n_test_samples, n_tokens), p=[mask_perc, 1 - mask_perc])
#     masked_test_X = np.copy(test_X)
#     masked_test_X[mask_test] = mask_token
#     #np.save(f'mask_test_{mask_perc}.npy', mask_test)
#     
#     #conditional data
#     #masked_exp_test_X = np.concatenate((test_expanded,masked_test_X ),axis=1)
#     #exp_test_y = np.concatenate(( test_onehot, test_y),axis=1)
#     
#     #wrong condition
#     w_masked_exp_test_X = np.concatenate((train_expanded[:10000],masked_test_X),axis=1)
#     w_exp_test_y = np.concatenate((train_onehot[:10000], test_y),axis=1)
#     
#     # see https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.MaskedLMOutput
#     reconstructions = model.predict({'inputs_embeds': w_masked_exp_test_X[:n_rec], 'labels': w_exp_test_y[:n_rec]}, batch_size=256)
#     
#     logits = reconstructions.logits
#     most_probable = logits.argmax(axis=-1)
#     ###
#     #np.save(f'reconstructions_test_{n_rec}_samples_{mask_perc}.npy', most_probable)
#         
#     
#     
#     tzq= most_probable[:,10:].reshape([n_rec,7,7])
#     
#     #decode(tzq,codes_sampler,x_test, 100,10,10,title)
#     decode(tzq,codes_sampler,x_test, 100,10,10,mask_perc)
#     
#     ### Calculate statistics    
#     
#     
#     
#     zq = codes_sampler(tzq)
#     decoded = decoder.predict(zq, steps=1)
#     pred=classifier.predict(decoded)
#     preds=np.argmax(pred, axis=1)
#     error05_reverse_wrongcond.append((preds != y_test[:n_rec]).mean())
# =============================================================================
    


