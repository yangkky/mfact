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
        stdv = torch.rsqrt(torch.Tensor([self.rank]))[0] * 2
        self.U._parameters['weight'].data.uniform_(-stdv, stdv)
        self.V._parameters['weight'].data.uniform_(-stdv, stdv)


    def forward(self, row_ids, col_ids):
        rows = self.U(row_ids)
        cols = self.V(col_ids)
        return torch.sum(rows * cols, dim=2).squeeze()


class DeepFactorModel(nn.Module):

    def __init__(self, n_rows, n_cols, row_rank, col_rank,
                 hidden_sizes, non_linear, dropout=0.1):
        super(DeepFactorModel, self).__init__()
        self.row_rank = row_rank
        self.col_rank = col_rank
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.U = nn.Embedding(self.n_rows, row_rank)
        self.V = nn.Embedding(self.n_cols, col_rank)
        self.non_linear = non_linear
        hidden_sizes = [row_rank + col_rank] + hidden_sizes
        self.stack = [nn.Linear(prev, curr) for prev, curr
                      in zip(hidden_sizes[:-1], hidden_sizes[1:])]
        self.stack = nn.ModuleList(self.stack)
        self.last_layer = nn.Linear(self.stack[-1].weight.size()[0], 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, rows_ids, col_ids):
        rows = self.U(rows_ids)
        cols = self.V(col_ids)
        h = torch.cat([rows, cols], dim=-1)
        h = self.dropout(h)
        for layer in self.stack:
            h = layer(h)
            h = self.non_linear(h)
            h = self.dropout(h)
        h = self.last_layer(h)
        return h.squeeze()


class KPMFLoss(nn.Module):

    def __init__(self, lambdas, K_U, K_V):
        super(KPMFLoss, self).__init__()
        self.K_U = K_U
        self.K_V = K_V
        self.S_U = K_U.inverse()
        self.S_V = K_V.inverse()
        self.lambdas = lambdas

    def forward(self, preds, targets, U, V):
        se, U_loss, V_loss = self.parts(preds, targets, U, V)
        return se + U_loss + V_loss

    def parts(self, preds, targets, U, V):
        n = len(preds)
        se = torch.sum((preds - targets) ** 2) / n * self.lambdas[0]
        U_loss = U.t() @ self.S_U
        U_loss = torch.sum(U_loss.t() * U) * self.lambdas[1]
        V_loss = V.t() @ self.S_V
        V_loss = torch.sum(V_loss.t() * V) * self.lambdas[2]
        return se, U_loss, V_loss

class KPMF(object):

    def __init__(self, rank, K_U, K_V, lambdas, lr=0.1):
        super(KPMF, self).__init__()
        n_rows = K_U.size()[0]
        n_cols = K_V.size()[0]
        self.lr = lr
        self.model = FactorModel(n_rows, n_cols, rank)
        self.loss_function = KPMFLoss(lambdas, K_U, K_V)
        self.U_optimizer = torch.optim.SGD([self.model.U._parameters['weight']], lr=lr)
        self.V_optimizer = torch.optim.SGD([self.model.V._parameters['weight']], lr=lr)

    def predict(self, rows, cols):
        return self.model(rows, cols)

    def MSE(self, rows, cols, targets):
        preds = self.predict(rows, cols)
        n = len(rows)
        return torch.sum((preds - targets) ** 2) / n

    def parts(self, rows, cols, targets):
        preds = self.model(rows, cols)
        U = self.model.U._parameters['weight']
        V = self.model.V._parameters['weight']
        return self.loss_function.parts(preds, targets, U, V)

    def fit(self, train_data, val_data, batch_size, min_its, max_its, patience=np.inf):
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
            total_n = 0
            total_mse = 0
            for idx in range(batches):
                if idx == batches - 1:
                    batch_idxs = inds[idx * batch_size:]
                else:
                    batch_idxs = inds[idx * batch_size: (idx + 1) * batch_size]
                batch_idxs = torch.LongTensor(batch_idxs)
                rows_b = rows_t[batch_idxs]
                cols_b = cols_t[batch_idxs]
                tars_b = tars_t[batch_idxs]
                self.model.train()
                loss, mse = self.train_batch((rows_b, cols_b, tars_b))
                total_loss += loss.cpu().data.numpy()[0] * len(batch_idxs)
                total_n += len(batch_idxs)
                total_mse += mse.cpu().data.numpy()[0] * len(batch_idxs)
            self.history['loss'].append(total_loss / total_n)
            self.history['train_SE'].append(total_mse / total_n)
            self.model.eval()
            self.history['valid_SE'].append(
                self.MSE(rows_v, cols_v, tars_v).data.numpy()[0])

            # Save checkpoint
            if np.argmin(self.history['valid_SE']) == it:
                torch.save(self.model.state_dict(), 'tmp.chkpt.pkl')


            print('\rEpoch %d\tloss = %.4f\ttrain SE = %.3f\tvalid SE = %.3f'
                  %(it+1, self.history['loss'][-1], self.history['train_SE'][-1],
                    self.history['valid_SE'][-1]),
                 end='')
            if len(self.history['valid_SE']) - np.argmin(self.history['valid_SE']) > patience:
                if it > min_its:
                    print('\nTerminating based on validation')
                    break
        # Set model to best parameters
        self.model.load_state_dict(torch.load('tmp.chkpt.pkl'))

        return self.history

    def train_batch(self, train_data):
        rows, cols, targets = train_data

        self.U_optimizer.zero_grad()
        preds = self.model(rows, cols)
        L = self.loss_function(preds, targets, self.model.U._parameters['weight'],
                      self.model.V._parameters['weight'])
        L.backward()
        self.U_optimizer.step()

        self.V_optimizer.zero_grad()
        preds = self.model(rows, cols)
        L = self.loss_function(preds, targets, self.model.U._parameters['weight'],
                      self.model.V._parameters['weight'])
        L.backward()
        self.V_optimizer.step()
        n = len(targets)
        mse = torch.sum((preds - targets) ** 2) / n
        return L, mse


class DeepKPMF(KPMF):

    def __init__(self, row_rank, col_rank, K_U, K_V, hidden_sizes, non_linear,
                 lambdas, lr=0.1, dropout=0.1, steps=50, factor=0.8):
        n_rows = K_U.size()[0]
        n_cols = K_V.size()[0]
        self.lr = lr
        self.model = DeepFactorModel(n_rows, n_cols, row_rank, col_rank,
                                     hidden_sizes, non_linear, dropout=dropout)
        self.loss_function = KPMFLoss(lambdas, K_U, K_V)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         steps, factor)

    def train_batch(self, train_data):
        rows, cols, targets = train_data
        self.optimizer.zero_grad()
        preds = self.model(rows, cols)
        L = self.loss_function(preds, targets, self.model.U._parameters['weight'],
                      self.model.V._parameters['weight'])
        L.backward(retain_graph=True)
        self.scheduler.step()
        self.optimizer.step()
        n = len(targets)
        mse = torch.sum((preds - targets) ** 2) / n
        return L, mse
