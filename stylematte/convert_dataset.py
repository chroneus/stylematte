import torch
import torchvision.transforms
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from models import *
from dataset import *
import tqdm
import glob
import sys
device  = "cuda"

config = OmegaConf.load(os.path.join(os.path.dirname("."),"config/base.yaml"))
print(config)
model =  StyleMatte()
model =  model.to(device)
checkpoint = "checkpoints/animalz.pth"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
state_dict = torch.load(checkpoint, map_location=f'{device}')
print("loaded",checkpoint)
model.load_state_dict(state_dict)
model.eval()
def process(inputpath):
    input = Image.open(inputpath)
    input_tensor=normalize(transforms.ToTensor()(input)).unsqueeze(0).to(device)
    with torch.no_grad():
        out=model(input_tensor)
        mask=np.uint8(out[0][0].cpu().detach().numpy()*255)
    Image.fromarray(np.dstack([np.array(input),mask]),mode="RGBA").save("/home/jovyan/datasets/afhq2/stylematte/"+sys.argv[1]+"_"+inputpath.split("/")[-1])

for inputpath in tqdm.tqdm(glob.glob(f"/home/jovyan/datasets/afhq2/v2/000{sys.argv[1]}/*.png")):
    process(inputpath)
