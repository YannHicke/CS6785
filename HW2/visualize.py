from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.models.gmvae import GMVAE
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F



model_name= "model=vae_z=10_run=0000"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(z_dim=10, name=model_name).to(device)

ut.load_model_by_name(vae, global_step=10000)

z = ut.sample_gaussian(torch.zeros((200, 10), device=device),
                       torch.ones((200, 10), device=device))
logits = vae.dec.decode(z)
probabilities = torch.sigmoid(logits)
predictions = ((probabilities > 0.5).cpu().detach().numpy()).astype(np.int32)

fig, axarr = plt.subplots(nrows=10, ncols=20, dpi=100, figsize=(10, 5))
for i, img in enumerate(predictions):
    img = img.reshape(28, 28)
    r, c = i // 20, i % 20
    axarr[r, c].imshow(img)
    axarr[r, c].set_axis_off()

fig.savefig('p1-4.png')

# GMVAE
model_name= "model=gmvae_z=10_k=500_run=0000"
gmvae = GMVAE(z_dim=10, k=500, name=model_name).to(device)

ut.load_model_by_name(gmvae, global_step=10000)

z = ut.sample_gaussian(torch.zeros((200, 10), device=device),
                       torch.ones((200, 10), device=device))
logits = gmvae.dec.decode(z)
probabilities = torch.sigmoid(logits)
predictions = ((probabilities > 0.5).cpu().detach().numpy()).astype(np.int32)

fig, axarr = plt.subplots(nrows=10, ncols=20, dpi=100, figsize=(10, 5))
for i, img in enumerate(predictions):
    img = img.reshape(28, 28)
    r, c = i // 20, i % 20
    axarr[r, c].imshow(img)
    axarr[r, c].set_axis_off()

fig.savefig('p1-4_gmvae.png')