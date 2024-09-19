import esm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def gather_edges(edges, neighbor_idx):
    # features [B,N,N,C] at neighbor indices [B,N,K] => obtain the neighbor features [B,N,K,C] for the nodes
    
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # features [B,N,C] at neighbor indices [B,N,K] => obtain the neighbor features [B,N,K,C]
    
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))     # [B, N'K]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))     # [B, N'K, nodes.size(2)]
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


class PositionalEncoding(nn.Module):
    '''"Positional encoding based on sequence distance"'''
    
    def __init__(self, pos_emb_dims, seq_neighbors, period_range=[2,1000]):
        super(PositionalEncoding, self).__init__()
        self.pos_emb_dims = pos_emb_dims
        self.seq_neighbors = seq_neighbors
        self.period_range = period_range
    
    def forward(self, E_idx, mask):
        # E_idx [B,C*N,K] mask [B,C*N]
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).cuda()     # [1,C*N',1]
        d = (E_idx.float() - ii).unsqueeze(-1)      # d: [B,C*N',K,1]
        d = d * mask.unsqueeze(-1).unsqueeze(-1)
        d[torch.abs(d) > self.seq_neighbors] = 0

        # Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.pos_emb_dims, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.pos_emb_dims)
        ).cuda()
        angles = d * frequency.view((1,1,1,-1))

        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        emask = (d.ne(0)).float().expand(-1,-1,-1,E.size(-1))

        E = E * emask

        return E


class Normalize(nn.Module):

    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):   
        # x:   [B,N',hidden_size]
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)   [B,N', 1]
        gain = self.gain
        bias = self.bias

        return gain * (x - mu) / (sigma + self.epsilon) + bias


class MPNNLayer(nn.Module):

    def __init__(self, num_hidden, num_in, dropout):
        super(MPNNLayer, self).__init__()

        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalize(num_hidden)

        self.W = nn.Sequential(
                nn.Linear(num_hidden + num_in, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
        )
        self.Ws = nn.Sequential(
            nn.Linear(num_hidden*2, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
        )
            
    def forward(self, h_V, h_E, mask_attend):
        # h_V: [B, C*N', hidden_size]
        # h_E: [B, C*N, K, hidden_size*2]
        # mask_attend: [B, C*N', K]/[B, C*N]
        
        if h_V.dim() != h_E.dim():
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)    # [B, C*N', K, hidden_size]
            h_EV = torch.cat([h_V_expand, h_E], dim=-1)
            h_message = self.W(h_EV) * mask_attend.unsqueeze(-1)    # [B, C*N', K, hidden_size]
            dh = torch.mean(h_message, dim=-2)
            h_V = self.norm(h_V + self.dropout(dh))     # [B, C*N', hidden_size]
        else:
            h_EV = torch.cat([h_V, h_E], dim=-1)    # [B,C*N,hidden_size*2]
            h_message = self.Ws(h_EV) * mask_attend.unsqueeze(-1)
            h_V = self.norm(h_V + self.dropout(h_message))

        return h_V


class Encoder(nn.Module):

    def __init__(self, args, node_in, edge_in, type):
        super(Encoder, self).__init__()
        self.node_in, self.edge_in = node_in, edge_in
        self.type = type
        self.depth = args.depth

        self.W_v = nn.Sequential(
                nn.Linear(self.node_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.W_e = nn.Sequential(
                nn.Linear(self.edge_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )

        self.coord_layers = nn.ModuleList([
                MPNNLayer(args.hidden_size, args.hidden_size * 2, dropout=args.dropout)
                for _ in range(self.depth)
        ])
        self.seq_layers = nn.ModuleList([
                MPNNLayer(args.hidden_size, args.hidden_size * 2, dropout=args.dropout)
                for _ in range(self.depth)
        ])

        for param in self.parameters(): 
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, V, E, E_idx, mask):

        # V [B,C*N,6], E [B,C*N,K,39]/(hS) [B,C*N,hidden_size], E_idx [B,C*N,K], mask [B,C*N]
        
        vmask = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)     # [B, C*N, K]

        if self.type == 'coord':

            h_v = self.W_v(V)
            h_e = self.W_e(E)

            # [B, N, 1] -> [B, N, K, 1] -> [B, N, K]
            h = h_v
            for layer in self.coord_layers:
                nei_v =  gather_nodes(h, E_idx)
                nei_h = torch.cat([nei_v, h_e], dim=-1)
                h = layer(h, nei_h, mask_attend=vmask)  # [B, C*N', hidden_size]
                h = h * mask.unsqueeze(-1)  # [B, N', hidden_size]

        elif self.type == 'seq':    # 第二步序列编码
            h = V   # [B, C*N, hidden_size]
            nei_s = gather_nodes(E, E_idx)  # 提取最近邻的序列特征表示  [B, C*N, K, hidden_size]
            for i,layer in enumerate(self.seq_layers):
                if i < int(self.depth/2):   # 前两层仅编码自身残基类型
                    nei_h = E.clone()       # [B,C*N,hidden_size]
                    h = layer(h, nei_h, mask_attend=mask)  # [B, C*N', hidden_size]
                else:   # 融合邻居节点的残基类型
                    nei_v =  gather_nodes(h, E_idx)
                    nei_h = torch.cat([nei_v, nei_s], dim=-1)   # 同情况coord 考虑邻居残基类型信息
                    h = layer(h, nei_h, mask_attend=vmask)

                h = h * mask.unsqueeze(-1)  # [B, N', hidden_size]


        return h
    


class ComplexFeatures(nn.Module):

    def __init__(self, top_k = 9, num_rbf = 16, pos_emb_dims= 16, seq_neighbors = 30):
        super(ComplexFeatures, self).__init__()

        self.num_rbf = num_rbf
        self.top_k = top_k
        self.feature_dimensions = (6, pos_emb_dims + num_rbf + 7)
        # self.feature_dimensions = (6, pos_emb_dims + num_rbf)
        self.embeddings = PositionalEncoding(pos_emb_dims, seq_neighbors)
    
    def _dihedrals(self, X, mask, eps=1e-7):
        '''计算二面角特征，用作节点表示'''
        # X: [B*C,N',4,3] mask [B*C,N']
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)    # 把前3个原子堆在一起，维度变为 [B*C, 3*N', 3]
        mask_expand = mask.unsqueeze(-1).repeat(1,1,3).view(mask.shape[0], -1)    # [B*C,3*N]
        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]     # 相邻坐标之间作差    [B*C, 3*N'-1, 3]
        U = F.normalize(dX*mask_expand[:,1:].unsqueeze(-1), dim=-1)     # 标准化    [B*C, 3*N'-1, 3]
        u_2 = U[:,:-2,:]        # 从N-CA开始  [B*C, 3*N'-3, 3]    
        u_1 = U[:,1:-1,:]       # 从CA-C开始  [B*C, 3*N'-3, 3]
        u_0 = U[:,2:,:]     # 从C-N开始  [B*C, 3*N'-3, 3]

        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)    # 计算法向量    [B*C, 3*N'-3, 3]
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)    # 计算法向量    [B*C, 3*N'-3, 3]

        # Angle between normals
        # 由于前面已经标准化过，所以这里可以直接用点积计算余弦值
        cosD = (n_2 * n_1).sum(-1)      # 计算两个法向量的余弦值，1表示两个向量的方向完全一致，-1表示完全相反，0表示垂直； [B*C, 3*N'-3]
        cosD = torch.clamp(cosD, -1+eps, 1-eps)     # 对cosD进行截断操作，防止超过余弦值的合法范围[-1, 1]，防止舍入误差等原因导致的计算异常； [B*C, 3*N'-3]
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)      # torch.sign确定角度的符号，torch.acos计算角度的弧度； [B*C, 3*N'-3]

        D = F.pad(D, (1,2), 'constant', 0)      # 补在最末尾3个 [B*C, 3*N']
        D = D.view((D.size(0), int(D.size(1)/3), 3))    # 将第二维度重新调整为原始的长度  [B*C, N', 3]

        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)     # 把每个夹角分解成余弦值和正弦值； [B*C, N', 6]

        return D_features*mask.unsqueeze(-1)
    
    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # X [B,C*N,3], mask [B,C*N]
        N = X.size(1)
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)     # 出于计算需要生成二维掩码  [B, C*N', C*N']
        mask_2D = mask_2D - torch.eye(N).unsqueeze(0).cuda()  # remove self    [B, C*N', C*N']
        # mask_2D = mask_2D - torch.eye(N).unsqueeze(0)
        mask_2D = mask_2D.clamp(min=0)  # 对张量进行截断，使张量中的元素不小于指定的最小值

        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)    # 第二维对应的是残基index，第三维对应的是所有残基和这个残基的坐标差    [B, C*N', C*N', 3]
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)     # 残基间欧氏距离 [B, C*N', C*N']

        # Identify k nearest neighbors (not including self)
        D_adjust = D + (1. - mask_2D) * 10000       # 给非标准AA处加上极大值    [B, C*N', C*N']
        top_k = min(self.top_k, N)
        D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)     #  每个残基各取出距离自己最近的9个邻居
        # D_neighbors: [B, C*N', K]
        # E_idx: [B, C*N', K]

        # mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)     # [B, N', K, 1]

        return D_neighbors, E_idx

    def _rbf(self, D):
        
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf   # D_min-D_max为RBF的距离范围，self.num_rbf为RBF函数的中心点数量
        D_mu = torch.linspace(D_min, D_max, D_count).cuda()     # [num_rbf]     生成RBF函数的中心点
        # D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view([1,1,1,-1])        # [1,1,1,num_rbf]
        D_sigma = (D_max - D_min) / D_count     # 计算RBF函数中心点之间的间隔大小
        D_expand = torch.unsqueeze(D, -1)       # [B,N',K,1]
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)      # [B,N',K,num_rbf]

        return RBF
    
    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        diag = torch.diagonal(R, dim1=-2, dim2=-1)  # 取[-2,-1]维构成矩阵的左对角线 [B,N',K,3]，最后一个3是对角线的长度
        Rxx, Ryy, Rzz = diag.unbind(-1)     # [B,N',K]  分别表示旋转矩阵的xx,yy,zz元素
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))    # 计算四元数的幅值部分
        _R = lambda i,j: R[:,:,:,i,j]       # 定义一个辅助函数_R用于获取旋转矩阵的元素
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))     # 计算四元数的符号部分
        xyz = signs * magnitudes    # 计算四元数的xyz部分
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.     # 计算四元数的 w 部分，使用 relu 函数确保非负迹
        Q = torch.cat((xyz, w), -1)     # 将四元数的 xyz 和 w 部分拼接得到四元数
        Q = F.normalize(Q, dim=-1)      # 对四元数进行归一化，确保其模为1

        return Q
    
    def _orientations_coarse(self, X, mask, E_idx):

        # X 为CA原子坐标 [B,C,N',3], mask [B,C,N'], E_idx [B,C*N',K]
        B, N = X.size(0), X.size(2)
        dX = X.view(-1,N,3)[:,1:,:] - X.view(-1,N,3)[:,:-1,:]     # 相邻残基坐标差    [B*C, N'-1, 3]
        U = F.normalize(dX*mask.view(-1,N)[:,1:].unsqueeze(-1), dim=-1)     # 标准化    [B*C, N'-1, 3]
        u_2 = U[:,:-2,:]    # [B*C, N'-3, 3]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)    # [B*C, N'-3, 3]  计算相邻残基之间的法向量 
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)    # [B*C, N'-3, 3]  计算相邻残基之间的相对方向
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)   # [B*C, N'-3, 3, 3]  将相对方向和法向量堆叠起来
        O = O.view(list(O.shape[:2]) + [9])     # [B*C, N'-3, 9]  这里的9不是K，是3*3；将旋转矩阵的信息展平成3*3的表示
        O = F.pad(O, (0,0,1,2), 'constant', 0)      # [B*C, N', 9]

        O_neighbors = gather_nodes(O.view(B,-1,9), E_idx)    # 提取最近邻的方向信息 [B,C*N', K, 9]
        O_neighbors = O_neighbors * mask.reshape(B,-1).unsqueeze(-1).unsqueeze(-1)
        X_neighbors = gather_nodes(X.reshape(B,-1,3), E_idx)    # 提取最近邻的坐标信息 [B,C*N', K, 3]
        X_neighbors = X_neighbors * mask.reshape(B,-1).unsqueeze(-1).unsqueeze(-1)
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])       # 调整旋转矩阵形状 [B*C, N', 3, 3]
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])     # K邻旋转矩阵形状 [B,C*N',K,3,3]

        # Rotate into local reference frames
        dX = X_neighbors - X.reshape(B,-1,3).unsqueeze(-2)      # 计算最近邻坐标和原坐标的差值 [B,C*N',K,3]
        dX = dX * (mask.reshape(B,-1).unsqueeze(-1).unsqueeze(-1))
        dU = torch.matmul(O.reshape(B,-1,3,3).unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)     # 通过矩阵相乘，将坐标变换到局部参考框架  [B,C*N', K, 3]
        dU = F.normalize(dU, dim=-1)        # [B,C*N',K,3]
        R = torch.matmul(O.reshape(B,-1,3,3).unsqueeze(2).transpose(-1,-2), O_neighbors)      # 构建旋转矩阵  [B,C*N', K, 3, 3]
        Q = self._quaternions(R)    # 把旋转矩阵转换成四元数   [B,C*N',K,4]

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)      # 将变换后的局部坐标和四元数连接在一起，得到空间旋转特征    [B,C*N',K,7]
        O_features = O_features * mask.reshape(B,-1).unsqueeze(-1).unsqueeze(-1)

        return O_features
    
    def forward(self, X, mask):
        '''结构图编码前向传播'''
        # X [B, C, N, 4, 3], mask [B, C, N]
        B, N = X.size(0), X.size(2)
        # 空间距离特征
        X_ca = X[:,:,:,1,:]   # [B, C, N, 3]
        D_neighbors, E_idx = self._dist(X_ca.reshape(B, -1, 3), mask.reshape(B, -1))     # 返回每个残基最近的K个欧式距离
        # D_neighbors [B, C*N, K], E_idx [B, C*N, K]
        RBF = self._rbf(D_neighbors)    # [B, C*N, K, num_rbf]
        # 序列距离特征
        E_positional = self.embeddings(E_idx, mask.reshape(B, -1))   # [B, C*N, K, 16]
        # 空间旋转特征
        O_features = self._orientations_coarse(X_ca, mask, E_idx)     # [B,C*N',K,7]
        # 残基构象特征/节点特征
        V = self._dihedrals(X.view(-1, N, 4, 3), mask.view(-1, N))    # [B*C, N, 6]
        # 边特征    
        E = torch.cat((E_positional, RBF, O_features), -1)      # [B,C*N,K,39]
        # E = torch.cat((E_positional, RBF), -1)        # [B,C*N,K,num_rbf+7]
        
        return V.view(B,-1,6), E, E_idx


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()

        self.esm = args.esm
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)      # 初始化序列编码层
        self.k_neighbors = args.k_neighbors
        self.block_size = args.block_size
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.L_max = args.L_max

        if self.esm != None:
            torch.hub.set_dir('/home/gongchen/model/esm') 
            self.model_esm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model_esm.eval()
            self.W_esm = nn.Sequential(
                    nn.Linear(320, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
            )
        self.rnn = nn.GRU(
                args.hidden_size, args.hidden_size//2, batch_first=True, 
                num_layers=1, bidirectional=True
        )
        self.features = ComplexFeatures(top_k=self.k_neighbors, num_rbf=args.num_rbf, pos_emb_dims=args.pos_dims)
        self.node_in, self.edge_in = self.features.feature_dimensions
        self.coord_mpn = Encoder(args, self.node_in, self.edge_in, type = 'coord')
        self.seq_mpn = Encoder(args, self.node_in, self.edge_in, type = 'seq')
        self.W_seq = nn.Sequential(
                nn.Linear(args.hidden_size*2, args.hidden_size),
                nn.ReLU()
        )
        self.mlp = nn.Sequential(
                        nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
                        nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.project = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size//2, self.hidden_size//8), nn.ReLU(),
                        nn.Linear(self.hidden_size//8, self.hidden_size//16), nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size//16, 1)
        )
        for param in self.parameters():     # 去掉这一段会默认随机初始化，可能导致训练不稳定，产生梯度消失或爆炸等问题
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


    def mask_mean(self, X, mask, i):
        '''从X中取block_size个数据[B, block_size,  ...]，返回其平均值[B, 1, ...]'''
        
        if X.dim() == 3:
            X = X[i:i+self.block_size, :]    # X: [block_size, 4, 3]    对于坐标数据
            mask = mask[i:i+self.block_size]      # [block_size]
            return torch.sum(X, dim=0, keepdims=True) / (mask.sum(dim=0, keepdims=True) + 1e-8)     #  [1, 4, 3]/[1]
        else:       
            X = X[i:i+self.block_size]      # 对于序列数据 [block_size, 256]
            mask = mask[i:i+self.block_size].unsqueeze(-1)       # [block_size, 1]

            return torch.sum(X * mask, dim=0, keepdims=True) / (mask.sum(dim=0, keepdims=True) + 1e-8)      # 序列已经编码过了，所以需要乘一个掩码

    def make_X_blocks(self, x, l, r, mask, M=False):
        '''把有效AA和interface之间的普通AA粗粒化'''
        # x [N,4,3]/[N,hidden_size]/[N]
        end = (mask != 0.0).nonzero(as_tuple=False).max().item()
        if not M:
            lblocks = [self.mask_mean(x, mask, i) for i in range(0,l,self.block_size)]
            rblocks = [self.mask_mean(x, mask, i) for i in range(r + 1, end, self.block_size)]
            bX = torch.cat(lblocks + [x[l:r+1,:]] + rblocks, dim=0)
        else:   # 对于两个掩码
            lblocks = [x[i:i+self.block_size].amax(dim=0, keepdims=True) for i in range(0,l,self.block_size)]
            rblocks = [x[i:i+self.block_size].amax(dim=0, keepdims=True) for i in range(r+1,end,self.block_size)]
            bX = torch.cat(lblocks + [x[l:r+1]] + rblocks, dim=0)

        if bX.dim() == 2: return bX     # 序列已经被嵌入了，需要计算梯度
        else: return bX.detach()    # 坐标和掩码不需要计算梯度
            
    def make_chain_blocks(self, x, os, ms, l, mask):
        '''粗粒化每一条链并返回链长度'''
        # x [N,4,3]
        # os, ms [N,hidden_size]
        # l, mask [N]
        if torch.all(l == 0).item():
            end = (mask != 0.0).nonzero(as_tuple=False).max().item()
            hx = torch.cat([self.mask_mean(x, mask, i) for i in range(0, end, self.block_size)])
            hos =  torch.cat([self.mask_mean(os, mask, i) for i in range(0, end, self.block_size)])
            hms =  torch.cat([self.mask_mean(ms, mask, i) for i in range(0, end, self.block_size)])
            mask = torch.cat([mask[i:i+self.block_size].amax(dim=0, keepdims=True) for i in range(0, end, self.block_size)])
            l = l[:len(mask)]
        else:
            # 确定粗粒化的范围
            start = (l != 0.0).nonzero()[0].item()
            end = (l != 0.0).nonzero(as_tuple=False).max().item()
            hx = self.make_X_blocks(x, start, end, mask)    # [N',4,3]
            hos = self.make_X_blocks(os, start, end, mask)
            hms = self.make_X_blocks(ms, start, end, mask)
            l = self.make_X_blocks(l, start, end, mask, M=True)
            mask = self.make_X_blocks(mask, start, end, mask, M=True)

        return [hx, hos, hms, l, mask], mask.size(0)
    
    def make_proteins(self, X, oS, mS, L, mask):
        '''编码序列并粗粒化整个张量'''
        # X [B,C,L,4,3], oS/mS/L/mask [B,C,L]
        B, C, N = mask.size(0), mask.size(1), mask.size(2)
        X = X.view(-1, N, 4, 3)     # [B*C,N,4,3]
        L = L.view(-1, N)
        mask = mask.view(-1, N)
        # print('mask:', mask.shape)
        if self.esm == None:
            hoS = self.W_s(oS).view(-1, N, self.hidden_size)      # [B*C,N,hidden_size]
            hmS = self.W_s(mS).view(-1, N, self.hidden_size)
        else:
            seq_lens = (oS != self.alphabet.padding_idx).sum(2).view(-1)
            # print('oS/mS:', oS[:,:,0], mS[:,:,-1])
            repre_oS = torch.zeros([oS.size(0)*oS.size(1), oS.size(2), 320], dtype=torch.float32)
            repre_mS = torch.zeros([mS.size(0)*mS.size(1), mS.size(2), 320], dtype=torch.float32)
            # print('repre_oS:', repre_oS.shape)
            for b in range(repre_oS.size(0)):
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    repre_oS[b,:] = self.model_esm(oS.view(-1,oS.size(2))[b,:].unsqueeze(0), repr_layers=[6], return_contacts=True)["representations"][6]
                    repre_mS[b,:] = self.model_esm(mS.view(-1,mS.size(2))[b,:].unsqueeze(0), repr_layers=[6], return_contacts=True)["representations"][6]

            hoS = torch.zeros([seq_lens.size(0), mask.size(-1), self.hidden_size], dtype=torch.float32).cuda()
            hmS = torch.zeros([seq_lens.size(0), mask.size(-1), self.hidden_size], dtype=torch.float32).cuda()
            for i, chain_len in enumerate(seq_lens):
                if chain_len == 1: continue
                hoS[i, : chain_len - 2] = self.W_esm(repre_oS[i, 1:chain_len-1].cuda())
                hmS[i, : chain_len - 2] = self.W_esm(repre_mS[i, 1:chain_len-1].cuda())

        # 创建填充张量
        bX = torch.zeros(B*C,self.L_max,4,3).cuda()
        boS = torch.zeros(B*C,self.L_max,self.hidden_size).cuda()
        bmS = torch.zeros(B*C,self.L_max,self.hidden_size).cuda()
        bL = torch.zeros(B*C,self.L_max).cuda()
        bmask = torch.zeros(B*C,self.L_max).cuda()

        for i in range(B*C):
            if not torch.all( mask[i,:] == 0.0).item():
                results, n = self.make_chain_blocks(X[i,:], hoS[i,:], hmS[i,:], L[i,:], mask[i,:])
                bX[i,:n,:], boS[i,:n,:], bmS[i,:n,:], bL[i,:n], bmask[i,:n] = results
        bX = bX.reshape(B,C,-1,4,3)      # [B,C,N',4,3]
        boS = boS.reshape(B,-1,self.hidden_size)    # [B,C*N',hidden_size]
        bmS = bmS.reshape(B,-1,self.hidden_size)    # [B,C*N',hidden_size]
        bL = bL.reshape(B,-1)    # [B,C*N']
        bmask = bmask.reshape(B,C,-1)      # [B,C,N']

        return bX, boS, bmS, bL, bmask

    def attention(self, Q, context, cmask, lmask, W):
        '''注意力机制编码'''
        # Q [B,C*N,hidden_size] context [B, C*N', hidden_size] cmask [B, C*N'] lmask [B, C*N']
        att = torch.bmm(Q, context.transpose(1,2))  # 计算点积得注意力分数 [B,C*N,C*N]
        # 核心区得分
        latt = att - 1e6 * (1-lmask.unsqueeze(1))   # 核心区掩码处理
        latt = F.softmax(latt, dim=-1)    # 对注意力分数进行softmax，得到权重分布
        lout = torch.bmm(latt, context)  # 根据权重对context进行加权平均，得到注意力池化后的表示    [B,N',hidden_size]
        lout = lout * lmask.unsqueeze(-1)

        # 普通区得分
        catt = att - 1e6 * (1 - cmask.unsqueeze(1))
        catt = F.softmax(catt, dim=-1)
        cout = torch.bmm(catt, context)
        cout = cout * cmask.unsqueeze(-1)

        out = torch.cat([Q, (lout+cout)], dim=-1)   # [B,C*N,hidden_size*2]
        # out = torch.cat([Q, cout], dim=-1)      # [B,C*N,hidden_size*2]

        return W(out) * cmask.unsqueeze(-1)
    
    def forward(self, X, oS, mS, L, mask, ddg_true=None):
        '''模型前向传播部分'''
        # 将链粗粒化后拼接为完整的复合物张量
        # X [B,C,L,4,3], oS/mS [B,C,L,hidden_size], L/mask [B,C,L]
        X, oS,  mS, L, mask = self.make_proteins(X, oS, mS, L, mask)    
        # X [B,C,N,4,3], oS [B,C*N,hidden_size], mask [B,C,N]
        B = mask.size(0)
        # 结构编码
        V, E, E_idx = self.features(X, mask)      # V: [B,C*N,6]   E: [B,C*N,K,39]   E_idx: [B,C*N,K]

        # 消息传递图编码
        hoS, _ = self.rnn(oS)   # [B,C*N,hidden_size]
        hmS, _ = self.rnn(mS)   # [B,C*N,hidden_size]

        # 消融
        # h_ori = self.coord_mpn(V, E, oS, E_idx, mask.reshape(B,-1))
        # h_mut = self.coord_mpn(V, E, mS, E_idx, mask.reshape(B,-1))

        h_coord = self.coord_mpn(V, E, E_idx, mask.reshape(B,-1))
        h_ori = self.seq_mpn(h_coord, oS, E_idx, mask.reshape(B,-1))
        # h_ori [B, C*N', hidden_size]
        h_ori = self.attention(h_ori, hoS, mask.reshape(B,-1), L.reshape(B,-1), self.W_seq)
        h_mut = self.seq_mpn(h_coord, mS, E_idx, mask.reshape(B,-1))
        h_mut = self.attention(h_mut, hmS, mask.reshape(B,-1), L.reshape(B,-1), self.W_seq)     
        # h_ori [B, C*N', hidden_size]
        
        # 送入模型传播
        feat_wm = torch.cat([h_ori, h_mut], dim=-1).cuda()  # [B, C*N', hidden_size*2]
        feat_mw = torch.cat([h_mut, h_ori], dim=-1).cuda()

        feat_diff = self.mlp(feat_wm) - self.mlp(feat_mw)   #  [B, C*N', hidden_size]
        
        per_residue_ddg = self.project(feat_diff).squeeze(-1)*mask.reshape(B,-1)    # [B,C*N']
        ddg_pre = per_residue_ddg.sum(dim=1)    # [B]

        if ddg_true is None:
            return ddg_pre
        else:
            losses = F.mse_loss(ddg_pre, ddg_true)      # tensor(1.5493, device='cuda:0', grad_fn=<MseLossBackward0>)
            return losses, ddg_pre
        
