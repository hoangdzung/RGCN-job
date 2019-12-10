import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_cross_entropy_with_logits( labels, logits):
    sig_aff = torch.sigmoid(logits)
    loss = labels * -torch.log(1e-10+sig_aff) + (1 - labels) * -torch.log(1e-10+1 - sig_aff)
    return loss

def unsup_loss(embedings0, embedings1, neg_embedings, weights):
    cuda = embedings0.is_cuda
    true_aff = F.cosine_similarity(embedings0, embedings1)
    neg_aff = embedings0.mm(neg_embedings.t())    
    true_labels = torch.ones(true_aff.shape)
    if cuda:
        true_labels = true_labels.cuda()
    true_xent = weights*sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
    neg_labels = torch.zeros(neg_aff.shape)
    if cuda:
        neg_labels = neg_labels.cuda()
    neg_xent = weights.unsqueeze(-1)*sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
    loss = true_xent.sum() + neg_xent.sum()
    return loss
