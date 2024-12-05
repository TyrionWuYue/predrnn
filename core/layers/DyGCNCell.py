import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from core.layers.ConvLayers import DoubleConv


class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super(GatedFusion, self).__init__()
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(embed_dim)
        self.w = nn.Linear(embed_dim, embed_dim)
        self.trans = nn.Parameter(torch.zeros(embed_dim, embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.trans, gain=1.414)
        self.w_r = nn.Linear(embed_dim, embed_dim)
        self.u_r = nn.Linear(embed_dim, embed_dim)
        self.w_h = nn.Linear(embed_dim, embed_dim)
        self.w_u = nn.Linear(embed_dim, embed_dim)

    def forward(self, node_embeddings, dyn_embeddings):
        #node_embeddings shaped [N, D], x shaped [B, N, D]
        #output shaped [B, N, D]
        batch_size = dyn_embeddings.shape[0]
        
        node_embeddings = self.norm(node_embeddings)
        node_embeddings_res = self.w(node_embeddings) + node_embeddings
        node_embeddings_res = node_embeddings_res.repeat(batch_size,  1, 1)

        et_res = dyn_embeddings + torch.einsum('bnd,dd->bnd', dyn_embeddings, self.trans)

        z = torch.sigmoid(node_embeddings_res + et_res)
        r = torch.sigmoid(self.w_r(dyn_embeddings) + self.u_r(node_embeddings).repeat(batch_size, 1, 1))
        h = torch.tanh(self.w_h(dyn_embeddings) + r * self.w_u(node_embeddings).repeat(batch_size, 1, 1))
        res = torch.add(z * node_embeddings, torch.mul(torch.ones(z.size()).to(dyn_embeddings.device) - z, h))

        return res


class ConditionalPositionalEncoding(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(ConditionalPositionalEncoding, self).__init__()
        self.pe = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, bias=True, groups=in_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        num_node = x.shape[1]
        w = d = int(np.sqrt(num_node))
        x = x.reshape(batch_size, -1, w, d)
        x = x + self.pe(x)
        x = x.reshape(batch_size, num_node, -1)
        return x


class DynAGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(DynAGCN, self).__init__()
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.pe = ConditionalPositionalEncoding(dim_in, kernel_size=5)

        # Dynamic Graph Embedding
        self.fc = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_in//2),
            nn.Sigmoid(),
            nn.Linear(self.dim_in//2, self.dim_in//2),
            nn.Sigmoid(),
            nn.Linear(self.dim_in//2, self.dim_in)
        )
        self.Wq = nn.Parameter(torch.randn(self.dim_in, self.embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.Wq)
        self.gated_fusion = GatedFusion(embed_dim)

        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool)
        nn.init.constant_(self.bias_pool, 0)

    def forward(self, x, node_embeddings, Mem):
        batch_size, num_nodes, _ = x.shape
        stc_E = node_embeddings

        x = self.pe(x)
        x_e = self.fc(x)
        query = torch.einsum('bnd,de->bne', x_e, self.Wq)
        att_score = F.softmax(torch.matmul(query, Mem.transpose(0, 1)), dim=-1)
        dyn_E = torch.matmul(att_score, Mem)

        comb_E = self.gated_fusion(stc_E, dyn_E)

        supports = F.softmax(F.relu(torch.matmul(comb_E, comb_E.transpose(-1, -2))), dim=-1)
        support_set = [torch.eye(num_nodes).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        x_g = torch.einsum("kbnm,bmc->bknc", supports, x)
        weights = torch.einsum('nd,dkio->nkio', stc_E, self.weights_pool)
        bias = torch.matmul(stc_E, self.bias_pool)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv


class DyGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, hidden_dim, cheb_k, mem_num, embed_dim):
        super(DyGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim

        self.conv = DoubleConv(dim_in, dim_in)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim_in, dim_in*4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_in*4),
            nn.GELU(),
            nn.Conv2d(dim_in*4, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim)
        )

        # Memory
        self.Mem = nn.Parameter(torch.randn(mem_num, embed_dim), requires_grad=True)
        # Static Graph Embedding
        self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.Mem)
        nn.init.xavier_normal_(self.node_embeddings)
        
        self.gate = DynAGCN(dim_in+self.hidden_dim, 2*self.hidden_dim, cheb_k, embed_dim)
        self.update = DynAGCN(dim_in+self.hidden_dim, self.hidden_dim, cheb_k, embed_dim)

    def forward(self, x, state):
        x = self.conv(x)
        x = self.ffn(x)
        assert x.shape[-2:] == state.shape[-2:]
        assert x.shape[-2] * x.shape[-1] == self.node_num
        batch_size = x.shape[0]
        width = x.shape[-1]
        height = x.shape[-2]
        x = x.reshape(batch_size, self.node_num, -1)
        state = state.reshape(batch_size, self.node_num, -1)
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, self.node_embeddings, self.Mem))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, self.node_embeddings, self.Mem))
        h = r*state + (1-r)*hc
        h = h.reshape(batch_size, -1, width, height)
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)