import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np

class SEKernel(nn.Module):
    
    def __init__(self, ell=1.0, sigma=1.0):
        super(SEKernel, self).__init__()
        self.ell = Parameter(torch.Tensor([ell]))
        self.sigma = Parameter(torch.Tensor([sigma]))
        
    def forward(self, X1, X2, inds=None):
        A = torch.sum(X1 ** 2, dim=1, keepdim=True)
        B = torch.sum(X2 ** 2, dim=1, keepdim=True)
        B = torch.t(B)
        C = 2 * torch.matmul(X1, torch.t(X2))
        D = A + B - C
        return self.sigma ** 2 * torch.exp(-0.5 * D / self.ell ** 2)  
    
class FactorModel(nn.Module):

    def __init__(self, n_rows, n_cols, rank):
        super(FactorModel, self).__init__()
        self.rank = rank
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.U = nn.Embedding(self.n_rows, rank)
        self.V = nn.Embedding(self.n_cols, rank)
        stdv = torch.rsqrt(torch.Tensor([self.rank]))[0]
        self.U._parameters['weight'].data.uniform_(-stdv, stdv)
        self.V._parameters['weight'].data.uniform_(-stdv, stdv)


    def forward(self, row_ids, col_ids):
        rows = self.U(row_ids)
        cols = self.V(col_ids)
        return torch.sum(rows * cols, dim=2).squeeze()


class KPMFLoss(nn.Module):

    def __init__(self, sigma, K_U, K_V):
        super(KPMFLoss, self).__init__()
        self.K_U = K_U
        self.K_V = K_V
        self.S_U = K_U.inverse()
        self.S_V = K_V.inverse()
        self.sigma = Parameter(torch.Tensor([sigma]))

    def forward(self, preds, targets, U, V):
        se = torch.sum((preds - targets) ** 2)
        se = se / self.sigma ** 2 / 2.0
        U_loss = U.t() @ self.S_U
        U_loss = 0.5 * torch.sum(U_loss.t() * U)
        V_loss = V.t() @ self.S_V
        V_loss = 0.5 * torch.sum(V_loss.t() * V)
        return (se, U_loss, V_loss)


class KPMF(object):

    def __init__(self, rank, K_U, K_V, sigma=0.1, lr=0.1):
        super(KPMF, self).__init__()
        n_rows = K_U.size()[0]
        n_cols = K_V.size()[0]
        self.lr = lr
        self.model = FactorModel(n_rows, n_cols, rank)
        self.loss_function = KPMFLoss(sigma, K_U, K_V)
        self.U_optimizer = torch.optim.SGD([self.model.U._parameters['weight']], lr=lr)
        self.V_optimizer = torch.optim.SGD([self.model.V._parameters['weight']], lr=lr)

    def predict(self, rows, cols):
        return self.model(rows, cols)

    def RMSE(self, rows, cols, targets):
        preds = self.predict(rows, cols)
        n = len(rows)
        return torch.sqrt(torch.sum((preds - targets) ** 2) / n)

    def fit(self, train_data, val_data, batch_size, min_its, max_its, early=True):
        self.history = {
            'loss': [],
            'train_SE': [],
            'valid_SE': []
        }
        rows_t, cols_t, tars_t = train_data
        rows_v, cols_v, tars_v = val_data
        n = len(tars_t)
        inds = np.arange(n)
        batches = int(np.ceil(n / batch_size))

        for it in range(max_its):
            np.random.shuffle(inds)
            total_loss = 0
            for idx in range(batches):
                if idx == batches - 1:
                    batch_idxs = inds[idx * batch_size:]
                else:
                    batch_idxs = inds[idx * batch_size: (idx + 1) * batch_size]
                batch_idxs = torch.LongTensor(batch_idxs)
                rows_b = rows_t[batch_idxs]
                cols_b = cols_t[batch_idxs]
                tars_b = tars_t[batch_idxs]
                loss = self.train_batch((rows_b, cols_b, tars_b))
                total_loss += loss.data.numpy()[0]
            self.history['loss'].append(total_loss / batches)
            self.history['train_SE'].append(
                self.RMSE(rows_t, cols_t, tars_t).data.numpy()[0])
            self.history['valid_SE'].append(
                self.RMSE(rows_v, cols_v, tars_v).data.numpy()[0])

            print('\rIteration %d\tloss = %.4f\ttrain SE = %.3f\tvalid SE = %.3f'
                  %(it+1, self.history['loss'][-1], self.history['train_SE'][-1],
                    self.history['valid_SE'][-1]),
                 end='')
            if early:
                if it > min_its:
                    if self.history['valid_SE'][-1] > self.history['valid_SE'][-2]:
                        print('\nTerminating based on validation')
                        break
        return self.history

    def train_batch(self, train_data):
        rows, cols, targets = train_data
        self.U_optimizer.zero_grad()
        preds = self.model(rows, cols)
        L = self.loss_function(preds, targets, self.model.U._parameters['weight'],
                      self.model.V._parameters['weight'])
        L.backward(retain_graph=True)
        self.U_optimizer.step()
        self.V_optimizer.zero_grad()
        preds = self.model(rows, cols)
        L = self.loss_function(preds, targets, self.model.U._parameters['weight'],
                      self.model.V._parameters['weight'])
        self.V_optimizer.step()
        return L
