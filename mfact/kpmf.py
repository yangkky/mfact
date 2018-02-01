import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np

class KPMF(nn.Module):

    def __init__(self, rank, sigma, K_U, K_V):
        super(KPMF, self).__init__()
        self.K_U = K_U
        self.K_V = K_V
        self.S_U = K_U.inverse()
        self.S_V = K_V.inverse()
        self.rank = rank
        self.n_rows = K_U.size()[0]
        self.n_cols = K_V.size()[0]
        self.U = Parameter(torch.Tensor(self.n_rows, rank))
        self.V = Parameter(torch.Tensor(rank, self.n_cols))
        self.sigma = Parameter(torch.Tensor([sigma]))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = torch.rsqrt(torch.Tensor([self.rank]))[0]
        self.U.data.uniform_(-stdv, stdv)
        self.V.data.uniform_(-stdv, stdv)

    def forward(self, target):
        E = Variable(torch.Tensor([0.0]))
        for t, ind in target:
            i, j = ind
            u = self.U[i, :].view(1, self.rank)
            v = self.V[:, j].unsqueeze(1)
            E = E + (t - torch.mm(u, v)) ** 2
        E /= (self.sigma ** 2 * 2.0)
        for ud in self.U.split(1, dim=1):
            E += 0.5 * (ud.t().mm(self.S_U)).mm(ud)
        for vd in self.V.t().split(1, dim=1):
            E += 0.5 * (vd.t().mm(self.S_V)).mm(vd)
        E += len(target) * torch.log(self.sigma ** 2)
        return E

    def get_trainable_parameters(self):
        return (p for p in self.parameters())



class KPMFTrainer(object):

    def __init__(self, rank, K_U, K_V, sigma=0.1, lr=0.1):
        super(KPMFTrainer, self).__init__()
        self.model = KPMF(rank, sigma, K_U, K_V)
        self.U_optimizer = torch.optim.SGD([self.model.U], lr=lr)
        self.V_optimizer = torch.optim.SGD([self.model.V], lr=lr)


    def predict(self):
        # add ability to do just one item
        return self.model.U.mm(self.model.V)

    def RMSE(self, targets):
        SE = 0.0
        n = len(targets)
        for t, ind in targets:
            i, j = ind
            u = self.model.U[i, :].view(1, self.model.rank)
            v = self.model.V[:, j].unsqueeze(1)
            SE += ((t - torch.mm(u, v)) ** 2)
        return torch.sqrt(SE / n)

    def train(self, train_data, val_data, batch_size, min_its, max_its, early=True):
        # Memory, more sophisticated termination criteria
        # everything able to be CUDAed
        self.history = {
            'loss': [],
            'train_SE': [],
            'valid_SE': []
        }
        n = len(train_data)
        inds = np.arange(len(train_data))
        batches = int(np.ceil(n / batch_size))

        for it in range(max_its):
            np.random.shuffle(inds)
            for idx in range(batches):
                if idx == batches - 1:
                    batch_idxs = inds[idx * batch_size:]
                else:
                    batch_idxs = inds[idx * batch_size: (idx + 1) * batch_size]
                loss = self.train_batch([train_data[i] for i in batch_idxs])
            self.history['loss'].append(self.model(train_data).data.numpy()[0][0])
            self.history['train_SE'].append(self.RMSE(train_data).data.numpy()[0][0])
            self.history['valid_SE'].append(self.RMSE(val_data).data.numpy()[0][0])

            print('\rIteration %d\tloss = %.4f\ttrain SE = %.3f\tvalid SE = %.3f'
                  %(it+1, self.history['loss'][-1], self.history['train_SE'][-1],
                    self.history['valid_SE'][-1]),
                 end='')
            if early and it > min_its and self.history['valid_SE'][-1] > self.history['valid_SE'][-2]:
                print('\nTerminating based on validation')
                break
        return self.history

    def train_batch(self, train_data):
        self.U_optimizer.zero_grad()
        loss = self.model(train_data)
        loss.backward(retain_graph=True)
        self.U_optimizer.step()
        self.V_optimizer.zero_grad()
        loss = self.model(train_data)
        loss.backward(retain_graph=True)
        self.V_optimizer.step()
        return loss
