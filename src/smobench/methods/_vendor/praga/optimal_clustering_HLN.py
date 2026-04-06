import  torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from sklearn.neighbors import kneighbors_graph
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.mixture import GaussianMixture
from .clustering_utils import split_and_merge_op, pairwise_distance

class R5(nn.Module):
    def __init__(self, data_type=None, arg=None):
        super(R5, self).__init__()
        self.tau = arg.tau
        self.N = 2
        self.begin = False
        self.datatype = data_type
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.datatype == 'SPOTS':
            self.weight = 1
            self.arg = Arg(init_K=10)

        elif self.datatype == 'Stereo-CITE-seq':
            self.weight = 1
            self.arg = Arg(init_K=10)

        elif self.datatype == '10x':
            self.weight = arg.cl_weight
            self.arg = Arg(init_K=arg.init_k)

        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.weight =2
            self.arg = Arg(init_K=14)
        else:
            self.weight = 5
            self.arg = Arg(init_K=5)

    def forward(self, feat, epoch):

        if epoch>99:
            if (self.begin == False) or (epoch % 50 == 0):
                print('updating clustring...')
                self.begin = True
                self.centroids = split_and_merge_op(feat, self.arg).to(self.device)
                dist = pairwise_distance(feat, self.centroids)
                value, pred = torch.min(dist, dim=1)
            else:
                dist = pairwise_distance(feat, self.centroids)
                value, pred = torch.min(dist, dim=1)

            cl_loss = self.CL(feat, pred, self.centroids)
        else:
            cl_loss = 0
        return cl_loss * self.weight

    def get_pred(self, x, class_num):
        gmm = GaussianMixture(n_components=class_num, random_state=0)
        gmm.fit(x)
        labels = gmm.predict(x)
        return labels

    def CL(self, feat, pred, centers):
        loss = 0
        valid_centers = 0
        
        for label, center in enumerate(centers):
            center = center.float()
            feat = feat.float()
            if feat[pred == label].shape[0] > 0 and feat[pred != label].shape[0] > 0:

                pos = torch.div(
                    torch.matmul(feat[pred == label], center.unsqueeze(0).T),
                    self.tau)
                neg = torch.div(
                    torch.matmul(feat, center.unsqueeze(0).T),
                    self.tau)

                # Clamp values to prevent overflow in exp
                pos = torch.clamp(pos.squeeze(), min=-10, max=10)
                neg = torch.clamp(neg.squeeze(), min=-10, max=10)
                
                pos = torch.mean(torch.exp(pos))
                neg = torch.mean(torch.exp(neg))

                # Add small epsilon to prevent division by zero
                eps = 1e-8
                if pos > eps and neg > eps:
                    loss_term = -torch.log(pos / (neg + eps))
                    if not (torch.isnan(loss_term) or torch.isinf(loss_term)):
                        loss = loss + loss_term
                        valid_centers += 1

        # Return average loss or 0 if no valid centers
        if valid_centers > 0:
            return loss / valid_centers
        else:
            return torch.tensor(0.0, device=feat.device)


class Arg:
    def __init__(self, init_K):
        self.pi_prior = 'uniform'
        self.prior_dir_counts = 0.1
        self.prior_mu_0 = 'data_mean'
        self.prior_sigma_choice = 'isotropic'
        self.prior_sigma_scale = .005
        self.prior_kappa = 0.0001
        self.prior_nu = init_K * 3
        self.class_num = init_K
        self.temperature = 2


class uniform_loss(nn.Module):
    def __init__(self, t=0.07):
        super(uniform_loss, self).__init__()
        self.t = t

    def forward(self, x):
        return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()








