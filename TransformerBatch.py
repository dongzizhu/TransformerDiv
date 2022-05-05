import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import normalize

import copy
import time

# torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigma = 10


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        self.encoded_output = output
        output = self.decoder(output)
        return output
    
    def get_attention_maps(self, src: Tensor, src_mask: Tensor):
        x = self.encoder(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        attention_mats = []
        attention_embs = []
        for l in self.transformer_encoder.layers:
            x2, attn_map = l.self_attn(x, x, x,
                                    attn_mask=src_mask,
                                    key_padding_mask=None,
                                    need_weights=True,
                                    average_attn_weights=False)
            attention_mats.append(attn_map)
            attention_embs.append(x2)
            x = l.norm1(x + l.dropout1(x2))
            x = l.norm2(x + l._ff_block(x))
        return attention_mats, attention_embs
    
    
    def get_attention_params(self, src: Tensor, src_mask: Tensor):
        x = self.encoder(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        attention_params = []
        for l in self.transformer_encoder.layers:
            attention_params.append(x)
            x = l(x, src_mask=src_mask)
        return attention_params


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def kernel(type='cos'):
    if type == 'softmax':
        kernel = softmaxKernel()
    elif type == 'gaussian':
        kernel = gaussianKernel()
    else:
        kernel = nn.CosineSimilarity(eps=1e-6)
    
    return kernel

class gaussianKernel(nn.Module):
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self) -> None:
        super(gaussianKernel, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        mid = torch.norm(x1-x2, dim=1)
        return torch.exp(-(mid)**2/(2*sigma**2))


class softmaxKernel(nn.Module):
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self) -> None:
        super(softmaxKernel, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        mid1 = F.normalize(x1).mul(F.normalize(x2))
        mid = torch.sum(mid1, dim=1)
        return torch.exp(mid)

class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


km = 'gaussian'
kp = 'softmax'
ke = 'HSIC'
cuda_cka = CudaCKA(device)

if km != 'HSIC':
    simKernelM = kernel(km)
else:
    simKernelM = cuda_cka

if kp != 'HSIC':
    simKernelP = kernel(kp)
else:
    simKernelP = cuda_cka

if ke != 'HSIC':
    simKernelE = kernel(ke)
else:
    simKernelE = cuda_cka



def main(i):
    def data_process(raw_text_iter: dataset.IterableDataset, vocab) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


    def batchify(data: Tensor, bsz: int) -> Tensor:
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            data: Tensor, shape [N]
            bsz: int, batch size

        Returns:
            Tensor of shape [N // bsz, bsz]
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device)


    def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            source: Tensor, shape [full_seq_len, batch_size]
            i: int

        Returns:
            tuple (data, target), where data has shape [seq_len, batch_size] and
            target has shape [seq_len * batch_size]
        """
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target
    
    def train(model: nn.Module, alpha: float, beta: float, gamma: float, ifDiverseMatrix: bool, ifSepP: bool, ifSepM: bool, ifDiverseParam: bool, ifembs: bool) -> None:
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(bptt).to(device)

        num_batches = len(train_data) // bptt
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:  # only on last batch
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            dp = 0
            dm = 0
            de = 0
            if ifDiverseParam:
                attn_params = model.get_attention_params(data, src_mask)
                head_shape = int(emsize / nhead)
                for params in attn_params:
                    for h1 in range(nhead):
                        for h2 in range(h1+1, nhead):
                            p1 = params[:, :, h1*head_shape:(h1+1)*head_shape]
                            p2 = params[:, :, h2*head_shape:(h2+1)*head_shape]
                            if ifSepP:
                                for b in range(p1.shape[1]):
                                    if kp!= 'HSIC':
                                        r = simKernelP(p1[:, b, :], p2[:, b, :])
                                    else:
                                        r = simKernelP.linear_CKA(p1[:, b, :], p2[:, b, :])
                                    dp += torch.mean(r)
                            else:
                                dp += simKernelP(p1.reshape(1, -1), p2.reshape(1, -1))
                if ifSepP:
                    dp = dp / p1.shape[1]
            # print(dp)
            
            if ifDiverseMatrix or ifembs:
                attn_mats, attn_embs = model.get_attention_maps(data, src_mask)
                
                if ifDiverseMatrix:
                    for t in attn_mats:
                        for i in range(t.shape[1]):
                            for j in range(i+1, t.shape[1]):
                                t1 = t[:, i, :, :]
                                t2 = t[:, j, :, :]
                                if ifSepM:
                                    for b in range(t1.shape[0]):
                                        if km!= 'HSIC':
                                            r = simKernelM(t1[b].T, t2[b].T)
                                        else:
                                            r = simKernelM.linear_CKA(t1[b].T, t2[b].T)
                                        dm += torch.mean(r)
                                else:
                                    dm += simKernelM(t1.reshape(1, -1), t2.reshape(1, -1))
                    if ifSepM:
                        dm = dm / t1.shape[0]
                
                
                if ifembs:
                    head_shape = int(emsize / nhead)
                    for embs in attn_embs:
                        for h1 in range(nhead):
                            for h2 in range(h1+1, nhead):
                                p1 = embs[:, :, h1*head_shape:(h1+1)*head_shape]
                                p2 = embs[:, :, h2*head_shape:(h2+1)*head_shape]
                                for b in range(p1.shape[1]):
                                    if ke!= 'HSIC':
                                        r = simKernelE(p1[:, b, :], p2[:, b, :])
                                    else:
                                        r = simKernelE.linear_CKA(p1[:, b, :], p2[:, b, :])
                                    de += torch.mean(r)
                    de = de/p1.shape[1]
            # print(dm)
            # print(de)
            
            
            loss = criterion(output.view(-1, ntokens), targets) + alpha * dp + beta * dm + gamma * de
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()


    def evaluate(model: nn.Module, eval_data: Tensor) -> float:
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(bptt).to(device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, bptt):
                data, targets = get_batch(eval_data, i)
                batch_size = data.size(0)
                if batch_size != bptt:
                    src_mask = src_mask[:batch_size, :batch_size]
                output = model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += batch_size * criterion(output_flat, targets).item()
        
        simKernel = kernel(k)
        attn_weights, _ = model.get_attention_maps(data, src_mask)
        d = 0
        for t in attn_weights:
            t1 = t[:, 0, :, :].reshape(1, -1)
            t2 = t[:, 1, :, :].reshape(1, -1)
            d += simKernel(t1, t2)
        print('kernal similarity: {}'.format(d))
        return total_loss / (len(eval_data) - 1)

    print('=' * 89)
    print(i)
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])


    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab)
    val_data = data_process(val_iter, vocab)
    test_data = data_process(test_iter, vocab)



    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    bptt = 35

    ntokens = len(vocab)  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)



    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        

    best_val_loss = float('inf')
    epochs = 8
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, alpha=0.01, beta=0.01, gamma=0.01, ifDiverseMatrix=True, ifSepM=False, ifSepP=False, ifDiverseParam=True, ifembs=True)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        scheduler.step()
        

    test_loss = evaluate(best_model, test_data)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
          f'test ppl {test_ppl:8.2f}')
    print('=' * 89)
    
    return test_ppl


if __name__ == '__main__':
    all_r = []
    for i in range(3):
        r = main(i)
        all_r.append(r)
    print(sum(all_r)/len(all_r))