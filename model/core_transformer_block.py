import torch 
import torch.nn as nn
import torch.nn.functional as F

class CoreTransformerEncoder(nn.Module):

    """
     Implementation of Core Transformer Block 

     Args:
        d_model (int): Dimension of the model (latent dimension).
        nhead (int): Num of heads 
        dropout (float): droput rate 
        mask (torch.Tensor, optional): Attention mask to prevent attention to certain places. Default to None. 

    Inputs:
        nodes (torch.Tensor): Node features of shape (batch_size, num_nodes, d_model/latent_dim)
        edges (torch.Tensor): Edge features of shape (batch_size, num_nodes, num_nodes, d_model/latent_dim)

    """

    def __init__(self, d_model, nhead, dropout, clip_value = 10):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout 
        self.clip_value = clip_value 
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = GeneralizedAttention(d_model, nhead, clip_value)
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "latent_dim must be divisible by num of heads"

        self.feedforward = nn.Sequential (
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), 
            nn.Dropout(dropout)
        )

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, mask = None):
        
        #Pre norm attention block
        nodes = self.layer_norm(nodes)
        z = self.attn(nodes, edges, mask)
        nodes = nodes + z[0]  #Do residuals 

        #Feedforward network block
        nodes = self.layer_norm(nodes)
        ff_o = self.feedforward(nodes)
        nodes = ff_o + nodes # Do residuals 
        
        return nodes
    

class GeneralizedAttention(nn.Module):

    """
     Implementation of Generalized Attention Mechanism for Core Transformer Block. 

     Args:
        d_model (int): Dimension of the model (latent dimension).
        nhead (int): Num of heads 
        mask (torch.Tensor, optional): Attention mask to prevent attention to certain places. Default to None. 

    Inputs:
        nodes (torch.Tensor): Node features of shape (batch_size, num_nodes, d_model/latent_dim)
        edges (torch.Tensor): Edge features of shape (batch_size, num_nodes, num_nodes, d_model/latent_dim)

    Consider this: https://medium.com/@pacharjee/maximizing-attention-how-attention-mechanisms-improve-deep-learning-models-55ced0fd545e
    https://github.com/naver/goal-co/blob/main/model/attention.py

    """

    def __init__(self, d_model, nhead, clip_value = 10):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.clip_value = clip_value
        assert d_model % nhead == 0, "d_model must be divisible by num of heads"
        self.node_query_proj = nn.Linear(d_model, d_model)
        self.node_key_proj = nn.Linear(d_model, d_model)
        self.node_value_proj = nn.Linear(d_model, d_model)

        self.edge_query_proj = nn.Linear(d_model, d_model)
        self.edge_key_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def MultiHeadNodeProjections(self, nodes: torch.Tensor):
        
        queries = self.node_query_proj(nodes) #(b, n, d)
        keys = self.node_key_proj(nodes)  #(b, n, d)
        values = self.node_value_proj(nodes) #(b, n, d)
        batch_size, num_nodes = nodes.size(0), nodes.size(1) # check this 

        queries = queries.view(batch_size, num_nodes, self.nhead, self.head_dim) # For splitting matrices between heads. 
        keys = keys.view(batch_size, num_nodes, self.nhead, self.head_dim)
        values = values.view(batch_size, num_nodes, self.nhead, self.head_dim)

        return queries, keys, values 

    def MultiHeadEdgeProjections(self, edges: torch.Tensor, batch_size:int, num_nodes: int):
        
        queries = self.edge_query_proj(edges) #(b, n, n, d)
        keys = self.edge_key_proj(edges) #(b, n, n, d)

        queries = queries.view(batch_size, num_nodes, num_nodes, self.nhead, self.head_dim)
        keys = keys.view(batch_size, num_nodes, num_nodes, self.nhead, self.head_dim)

        return queries, keys 

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, mask= None):

        batch_size, num_nodes = nodes.size(0), nodes.size(1)
                
        nodes_queries, nodes_keys, nodes_values = self.MultiHeadNodeProjections(nodes) # these will already come split for different heads 
        edges_queries, edges_keys = self.MultiHeadEdgeProjections(edges, batch_size, num_nodes)

        # to perform generalized attention in a batched and efficient way, we add a node to all corresponding edges and peform attention
        # between edges 
        query = edges_queries + nodes_queries [:, :, None, :, :] # (b, n, 1, h, d) #first broadcast then add
        key = edges_keys + nodes_keys [:, None, :, :, :] # (b, 1, n, h, d)
        scores = torch.einsum('bijhd, bijhd -> bhij', query, key) / (self.head_dim ** 0.5)  # (b, h, n, n)

        if self.clip_value is not None:
            scores = self.clip_value * torch.tanh(scores)

        # Apply mask, if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # add a dimension so that mask is from (b, n, n) to (b, h, n, n)
            scores = scores + mask # additive mask

        # Compute the attention weights
        attn_weights = F.softmax(scores, dim=-1) # softmax over the last dimension (over keys)

        # Compute the output
        output = torch.matmul(attn_weights, nodes_values.transpose(1, 2)) #(b,h,n,n) @ (b,h,n,d) → (b,h,n,d)

        # Combine heads and project output
        output = output.transpose(1, 2).contiguous() #(b,h,n,d) -> (b,n,h, d)
        output = output.view(batch_size, num_nodes, self.d_model)
        output = self.output_proj(output)

        return output, attn_weights






