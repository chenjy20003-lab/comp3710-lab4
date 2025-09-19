"""
part4-vae.py — Simple VAE on OASIS
 - Dataset: lazy loader for preprocessed OASIS (npz/png)
 - Model: small conv VAE (z=32), MSE recon + KL
 - Output: logs/ep*_in.png, ep*_recon.png, samples.png
"""

# part4-vae.py — 简化 VAE on OASIS
import os, time, glob, math
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---- config ----
DATA_DIR = "/home/groups/comp3710/OASIS_preprocessed"
OUT_DIR, IMG_SIZE, LATENT, BATCH, EPOCHS = "logs", 128, 32, 64, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- dataset ----
class MRIDataset(Dataset):
    def __init__(self, root, size=128):
        self.files = glob.glob(os.path.join(root,"**/*.*"), recursive=True)
        self.size=size
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        f=self.files[i]
        if f.endswith((".npy",".npz")):
            x=np.load(f);
            if isinstance(x, np.lib.npyio.NpzFile): x=x[x.files[0]]
            if x.ndim==3: x=x[:,:,x.shape[2]//2]
        else: x=np.array(Image.open(f).convert("L"))
        x=(x-x.min())/(x.max()-x.min()+1e-8)
        x=Image.fromarray((x*255).astype(np.uint8)).resize((self.size,self.size))
        return torch.tensor(np.array(x)/255.,dtype=torch.float32)[None]

# ---- model ----
class VAE(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(1,32,4,2,1),nn.ReLU(),nn.Conv2d(32,64,4,2,1),nn.ReLU(),nn.Flatten())
        self.fc_mu=nn.Linear(64*32*32,zdim); self.fc_log=nn.Linear(64*32*32,zdim)
        self.fc=nn.Linear(zdim,64*32*32)
        self.dec=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,1,4,2,1),nn.Sigmoid())
    def forward(self,x):
        h=self.enc(x); mu,log=self.fc_mu(h),self.fc_log(h)
        z=mu+torch.randn_like(mu)*torch.exp(0.5*log)
        out=self.dec(self.fc(z).view(-1,64,32,32))
        return out,mu,log

def vae_loss(x,xh,mu,log):
    rec=nn.functional.mse_loss(xh,x); kl=-0.5*torch.mean(1+log-mu**2-log.exp())
    return rec+kl,rec,kl

# ---- train ----
def save_img(t,path):
    t=t.detach().cpu().numpy()[:16,0]; n=4; h,w=t.shape[1:]; grid=np.zeros((n*h,n*w))
    for i in range(n):
        for j in range(n): grid[i*h:(i+1)*h,j*w:(j+1)*w]=t[i*n+j]
    plt.imshow(grid,cmap="gray"); plt.axis("off"); plt.savefig(path); plt.close()

def main():
    os.makedirs(OUT_DIR,exist_ok=True)
    ds=MRIDataset(DATA_DIR,IMG_SIZE); n=len(ds)//10
    train,val=torch.utils.data.random_split(ds,[len(ds)-n,n])
    tl=DataLoader(train,BATCH,True); vl=DataLoader(val,BATCH)
    model=VAE(LATENT).to(device); opt=torch.optim.Adam(model.parameters(),1e-3)

    for ep in range(1,EPOCHS+1):
        model.train();
        for x in tl:
            x=x.to(device); opt.zero_grad()
            xh,mu,log=model(x); loss,_,_=vae_loss(x,xh,mu,log)
            loss.backward(); opt.step()
        model.eval();
        with torch.no_grad():
            x=next(iter(vl)).to(device); xh,mu,log=model(x)
            save_img(x,f"{OUT_DIR}/ep{ep}_in.png"); save_img(xh,f"{OUT_DIR}/ep{ep}_recon.png")
    # 随机采样
    with torch.no_grad():
        z=torch.randn(16,LATENT,device=device); s=model.dec(model.fc(z).view(-1,64,32,32))
        save_img(s,f"{OUT_DIR}/samples.png")

if __name__=="__main__": main()
