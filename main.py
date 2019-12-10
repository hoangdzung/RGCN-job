import torch
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from model import HeteroRGCN
from utils import SAGEDataset, EarlyStopping, load_data
from loss import unsup_loss
import numpy as np
from tqdm import tqdm 
import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser()
    ### Model stuff
    parser.add_argument('--dataset', default="flickr")
    parser.add_argument('--hidden_dim', default = 64, type = int)
    parser.add_argument('--out_dim', default = 32, type = int)

    ### Optimizer stuff
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--wc', default = 5e-4, type = float)
    parser.add_argument('--patience', default = 5, type = int)

    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--batch_size', default = 1024, type = int)
    parser.add_argument('--neg_size', default = 20, type = int)
    parser.add_argument('--swap', help="Swap src and dst nodes in edge iterator", action = "store_true")
    parser.add_argument('--weighted_loss', action = "store_true")

    parser.add_argument('--model_dir', default='model')
    parser.add_argument('--load_pretrained', action = "store_true")

    parser.add_argument('--seed', default= 42, type = int)
    parser.add_argument('--cuda', action = "store_true")

    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)

G, edge_test = load_data()
dataset = SAGEDataset(G, args.batch_size, args.neg_size, args.swap)

model = HeteroRGCN(G, args.hidden_dim, args.out_dim)

if os.path.isfile(os.path.join(args.model_dir, 'model.pt')) and args.load_pretrained:
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt')))

if args.cuda and torch.cuda.is_available:
    model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wc)

best_embeddings = None 
best_loss = 1e20
early_stopping = EarlyStopping(patience=args.patience, verbose=True, datadir=args.model_dir)
for epoch in range(args.epochs):
    total_loss = 0
    for batch_edges, batch_weights, neg in tqdm(dataset, desc='Training'):
        pos0 = batch_edges[:,0]
        pos1 = batch_edges[:,1]

        logits = model(G)
        all_embeddings = torch.cat([logits[ntype] for ntype in G.ntypes])

        embedings0 = F.normalize(all_embeddings[pos0], dim=1)
        embedings1 = F.normalize(all_embeddings[pos1], dim=1)
        neg_embedings = F.normalize(all_embeddings[neg], dim=1)
        if args.weighted_loss:
            batch_weights = torch.FloatTensor(batch_weights)
        else:
            batch_weights = torch.ones((batch_edges.shape[0],))
        
        if args.cuda and torch.cuda.is_available:
            batch_weights = batch_weights.cuda()

        loss = unsup_loss(embedings0, embedings1, neg_embedings, batch_weights)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
    if total_loss != total_loss:
        print("nan???, break")
        break
    early_stopping(total_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

    if early_stopping.is_best:
        best_embeddings = all_embeddings.detach().cpu().numpy()
        np.save(os.path.join(args.model_dir, 'all_embeddings.npy'), best_embeddings)
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pt'))


