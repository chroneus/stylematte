import torch
import random
import dnnlib
import legacy
from PIL import Image
import numpy as np
from torchvision import transforms
normalize =transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
import tqdm
from typing import List, Optional, Tuple, Union


import sys

device = torch.device('cuda')
network_pkl="./animals.pkl" #"./humans.pkl"


def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


if __name__ == "__main__":



    with dnnlib.util.open_url(network_pkl) as f:
        dict_networks=legacy.load_network_pkl(f)
        G3 = dict_networks['G_ema'].to(device) # type: ignore

    batch=1
    idx=0
    for i in tqdm.tqdm(range(idx*10000,(idx+1)*10000)):

        label=torch.zeros([1, G3.c_dim], device=device)

        z = torch.randn((batch,512)).to(device)
        zsave=z
        truncation = int(10*(random.random()*2-1))/10
        
        m = make_transform(((random.random()-0.5)/10,(random.random()-0.5)/10), random.random()*10)
        m = np.linalg.inv(m)
        G3.synthesis.input.transform.copy_(torch.from_numpy(m))
        with torch.no_grad():
            img = G3(zsave,label,  truncation_psi=truncation)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()
        for j in range(batch):
            Image.fromarray(img[j].numpy(), 'RGBA').save(f"~/datasets/synthetic_psi/{i*batch+j}_{truncation}.png")
