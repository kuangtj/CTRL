import dgl
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from functools import partial
import copy

import math

# dgl graph utils
def reverse_edge(tensor):
    n = tensor.size(0)

    if n <= 1:
        return tensor

    assert n % 2 == 0, f"reverse_edge expects even number of edges, got {n}"

    device = tensor.device
    delta = torch.ones(n, dtype=torch.long, device=device)

    idx = torch.arange(1, n, 2, dtype=torch.long, device=device)
    delta[idx] = -1

    base = torch.arange(n, dtype=torch.long, device=device)
    return tensor[delta + base]


def del_reverse_message(edge,field):
    """for g.apply_edges"""
    return {'m': edge.src[field]-edge.data['rev_h']}

def add_attn(node,field,attn):
        feat = node.data[field].unsqueeze(1)
        return {field: (attn(feat,node.mailbox['m'],node.mailbox['m'])+feat).squeeze(1)}

# nn modules

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    # p_attn = F.softmax(scores, dim = -1).masked_fill(mask, 0)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

       
class MVMP(nn.Module):
    def __init__(self,msg_func=add_attn,hid_dim=300,depth=3,view='aba',suffix='h',act=nn.ReLU()):
        """
        MultiViewMassagePassing
        view: a, ap, apj
        suffix: filed to save the nodes' hidden state in dgl.graph. 
                e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
        """
        super(MVMP,self).__init__()
        self.view = view
        self.depth = depth
        self.suffix = suffix
        self.msg_func = msg_func
        self.act = act
        self.homo_etypes = [('a','b','a')]
        self.hetero_etypes = []
        self.node_types = ['a','p']
        if 'p' in view:
            self.homo_etypes.append(('p','r','p'))
        if 'j' in view:
            self.node_types.append('junc')
            self.hetero_etypes=[('a','j','p'),('p','j','a')] # don't have feature

        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(4,hid_dim)

        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes:
            self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim,hid_dim) for i in range(depth-1)])

        self.node_last_layer = nn.ModuleDict()
        for ntype in self.node_types:
            self.node_last_layer[ntype] = nn.Linear(3*hid_dim,hid_dim)

    def update_edge(self,edge,layer):
        return {'h':self.act(edge.data['x']+layer(edge.data['m']))}
    
    def update_node(self,node,field,layer):
        return {field:layer(torch.cat([node.mailbox['mail'].sum(dim=1),
                                       node.data[field],
                                       node.data['f']],1))}
    def init_node(self,node):
        return {f'f_{self.suffix}':node.data['f'].clone()}

    def init_edge(self,edge):
        return {'h':edge.data['x'].clone()}


    def forward(self,bg):
        suffix = self.suffix
        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node,ntype=ntype)
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge,etype=etype)

        if 'j' in self.view:
            bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
            bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()

        update_funcs = {e:(fn.copy_e('h','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in self.homo_etypes }
        update_funcs.update({e:(fn.copy_src(f'f_junc_{suffix}','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})
        # message passing
        for i in range(self.depth-1):
            bg.multi_update_all(update_funcs,cross_reducer='sum')
            for edge_type in self.homo_etypes:
                bg.edges[edge_type].data['rev_h']=reverse_edge(bg.edges[edge_type].data['h'])
                bg.apply_edges(partial(del_reverse_message,field=f'f_{suffix}'),etype=edge_type)
                bg.apply_edges(partial(self.update_edge,layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)

        # last update of node feature
        update_funcs = {e:(fn.copy_e('h','mail'),partial(self.update_node,field=f'f_{suffix}',layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
        bg.multi_update_all(update_funcs,cross_reducer='sum')

        # last update of junc feature
        bg.multi_update_all({e:(fn.copy_src(f'f_junc_{suffix}','mail'),
                                 partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
                                 cross_reducer='sum')


class PharmHGT_AtomFuse(nn.Module):
    def __init__(self, args={'hid_dim': 256, 'depth': 3, 'act': "ReLU", 'atom_dim': 42, 'bond_dim': 14, 'pharm_dim': 194, 'reac_dim': 34}):
        super().__init__()
        hid_dim = args['hid_dim']
        self.act = nn.LeakyReLU()
        self.depth = args['depth']

        # ---- init linear ----
        self.w_atom = nn.Linear(args['atom_dim'], hid_dim)
        self.w_bond = nn.Linear(args['bond_dim'], hid_dim)

        self.w_pharm = nn.Linear(args['pharm_dim'], hid_dim)
        self.w_reac  = nn.Linear(args['reac_dim'],  hid_dim)

        self.w_junc  = nn.Linear(args['atom_dim'] + args['pharm_dim'], hid_dim)

        # ---- MP modules ----
        # 1) atom-only interaction (a-b-a)
        self.mp_atom  = MVMP(msg_func=add_attn, hid_dim=hid_dim, depth=self.depth,
                             view='a', suffix='a', act=self.act)
        # 2) pharm-level interaction (p-r-p) + (a-b-a)
        self.mp_pharm = MVMP(msg_func=add_attn, hid_dim=hid_dim, depth=self.depth,
                             view='ap', suffix='p', act=self.act)

        # ---- fuse (atom <- pharm_msg) ----
        self.gate = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.Sigmoid()
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            self.act,
            nn.Linear(hid_dim, hid_dim)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def init_feature(self, bg):
        # node
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.nodes['p'].data['f'] = self.act(self.w_pharm(bg.nodes['p'].data['f']))

        # edge
        bg.edges[('a','b','a')].data['x'] = self.act(self.w_bond(bg.edges[('a','b','a')].data['x']))
        bg.edges[('p','r','p')].data['x'] = self.act(self.w_reac(bg.edges[('p','r','p')].data['x']))

        # junc feature
        bg.nodes['a'].data['f_junc'] = self.act(self.w_junc(bg.nodes['a'].data['f_junc']))
        bg.nodes['p'].data['f_junc'] = self.act(self.w_junc(bg.nodes['p'].data['f_junc']))

    @staticmethod
    def _aggregate_pharm_to_atom(bg, pharm_field: str, out_field: str = "pharm_msg"):
        device = bg.device
        num_a = bg.num_nodes('a')

        dim = bg.nodes['p'].data[pharm_field].shape[-1]
        bg.nodes['a'].data[out_field] = torch.zeros((num_a, dim), device=device)

        etype = ('p', 'j', 'a')
        if bg.num_edges(etype) == 0:
            return

        bg.update_all(
            message_func=fn.copy_src(pharm_field, 'm'),
            reduce_func=fn.mean('m', out_field),
            etype=etype
        )

    def forward(self, bg):
        bg = bg.local_var()
        self.init_feature(bg)
        self.mp_atom(bg)
        h_atom = bg.nodes['a'].data['f_a']   # [N_atom, hid]

        # 2) pharm-level MP: 得到 f_p（p-r-p）
        self.mp_pharm(bg)
        h_pharm = bg.nodes['p'].data['f_p']  # [N_pharm, hid]

        # 3) pharm -> atom 聚合（通过 p-j-a 边）
        self._aggregate_pharm_to_atom(bg, pharm_field='f_p', out_field='pharm_msg')
        pharm_msg = bg.nodes['a'].data['pharm_msg']  # [N_atom, hid]

        # 4) fuse: atom + gate * pharm_msg
        g = self.gate(torch.cat([h_atom, pharm_msg], dim=-1))      # [N_atom, hid]
        h_fused = h_atom + g * pharm_msg
        h_fused = self.fuse_mlp(h_fused)

        bg.nodes['a'].data['h_fused'] = h_fused

        # 输出 atom-level 表征 + 每图 atom 数，方便你拆回 batch
        atom_num = bg.batch_num_nodes('a')
        return h_fused, atom_num


