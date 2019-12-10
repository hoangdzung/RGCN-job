from model import HeteroRGCN
from utils import load_data
from tqdm import tqdm 
import torch

G = load_data() 
model = HeteroRGCN(G, 512, 128, 64)
model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

for _ in tqdm(range(10), desc='Training'):
    logits = model(G)
