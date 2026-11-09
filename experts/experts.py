from torch import nn
import torch 
from model.core_transformer_block import CoreTransformerEncoder 


# Implementation of all unit expert 
# normalization for interaction components in Flow Expert? 

class DistillationColumn(nn.Module):
    
    def __init__(self, gen_config, env_config):
        super(DistillationColumn, self).__init__()
        self.name = "distillation_column"
        self.gen_config = gen_config 
        self.env_config = env_config
        self.latent_dim = self.gen_config.latent_dim
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.df_categorical = nn.Linear(in_features = self.latent_dim , out_features = len(self.env_config.DF_distillation_map))
        self.df_and_type_embed = nn.Linear(in_features = 1, out_features=self.latent_dim, bias = True)
        

    def embed(self, node_data: dict):
        df = node_data["params"]["df"]

        # convert into a tensor
        df_tensor = torch.tensor([float(df)], dtype=torch.float32, device= self.gen_config.training_device)

        return self.df_and_type_embed(df_tensor)
        
    def predict(self, x: torch.FloatTensor):
        # x is the embedding of shape (batch, num_nodes, latent_dim)
        return {
            "picked_logit": self.logit_linear(x), 
            "distillate_fraction_categorical": self.df_categorical(x)
            }

class Decanter(nn.Module):
    def __init__(self, config):
        super(Decanter, self).__init__()
        self.name = "decanter"
        self.config = config 
        self.latent_dim = self.config.latent_dim 
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.type_embed = nn.Embedding(num_embeddings= 1, embedding_dim= self.latent_dim)

    def embed(self, batch_size: int):
        return self.type_embed(
            torch.tensor([0] * batch_size, dtype = torch.long, device=self.config.training_device) #(batch_size, latent_dim)
        )
        
    def predict(self, x: torch.FloatTensor):

        # x is the embedding of shape (batch, num_nodes, latent_dim)
        
        return {
            "picked_logit": self.logit_linear(x), 
            }
    
class Mixer(nn.Module):
    def __init__(self, config):
        super(Mixer, self).__init__()
        self.name = "mixer"
        self.config = config 
        self.latent_dim = self.config.latent_dim 
        self.query_linear_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.key_linear_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.scale_constant = 10
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.type_embed = nn.Embedding(num_embeddings= 1, embedding_dim= self.latent_dim)
        self.choose_destination_outlet = nn.Linear(in_features = self.latent_dim , out_features = 2) #max_num of outlets allowed but idk if this is right 

    def embed(self, batch_size: int):
        return self.type_embed(
            torch.tensor([0] * batch_size, dtype = torch.long, device=self.config.training_device) #(batch_size, latent_dim)
        )
        
    def predict(self, x: torch.FloatTensor, mixer_mask = None):

        # x is the embedding of shape (batch_size, num_nodes, latent_dim)
        query = self.query_linear_proj(x)
        key = self.key_linear_proj(x)
        scores = torch.einsum('bnd, bmd -> bnm', query, key)
        scores = self.scale_constant * torch.tanh(scores)

        # mask out not allowed nodes (e.g self connections, system source nodes)
        # Apply mask only if it exists

        if mixer_mask is not None:
            mixer_mask = torch.as_tensor(mixer_mask, device=self.config.training_device, dtype=torch.bool)
            scores = scores.masked_fill(~mixer_mask, -1e9) 

        return {
        "picked_logit": self.logit_linear(x), # (batch_size, num_nodes, 1)
        "target_scores": scores, # (batch_size, num_nodes, num_nodes)
        "destinate_node_outlets": self.choose_destination_outlet(x) # (batch_size, num_nodes, max_potential_outlets)
        } 
    

class Split(nn.Module):
    def __init__(self, gen_config, env_config):
        super(Split, self).__init__()
        self.name = "split"
        self.gen_config = gen_config 
        self.env_config = env_config
        self.latent_dim = self.gen_config.latent_dim 
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.split_ratio_categorical = nn.Linear(in_features = self.latent_dim, out_features = len(self.env_config.split_ratio_map))
        self.split_ratio_and_type_embed = nn.Linear(in_features= 1, out_features= self.latent_dim, bias = True)

    def embed(self, node_data: dict):
        sr = node_data["params"]["split_ratio"]
        sr_tensor = torch.tensor([float(sr)], dtype=torch.float32, device=self.gen_config.training_device)
        return self.split_ratio_and_type_embed(sr_tensor)
        
    def predict(self, x: torch.FloatTensor):

        # x is the embedding of shape (batch, num_nodes, latent_dim)
        
        return {
            "picked_logit": self.logit_linear(x), 
            "split_ratio_categorical": self.split_ratio_categorical(x)
            }
    
class Recycler(nn.Module):
    def __init__(self, config):
        super(Recycler, self).__init__()
        self.name = "recycler"
        self.config = config 
        self.latent_dim = self.config.latent_dim 
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.query_linear_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.key_linear_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.scale_constant = 10

    def predict(self, x: torch.FloatTensor, recycler_mask = None):

        # x is the embedding of shape (batch_size, num_nodes, latent_dim)
        # recycler_mask: (B, N, N) 

        query = self.query_linear_proj(x)
        key = self.key_linear_proj(x)
        scores = torch.einsum('bnd, bmd -> bnm', query, key)
        scores = self.scale_constant * torch.tanh(scores)

        # mask out not allowed nodes (e.g self connections, system source nodes)
        if recycler_mask is not None:
            recycler_mask = torch.as_tensor(recycler_mask, device=self.config.training_device, dtype=torch.bool)
            scores = scores.masked_fill(~recycler_mask, -1e9) 

        return {
        "picked_logit": self.logit_linear(x), # (batch_size, num_nodes, 1)
        "target_scores": scores # (batch_size, num_nodes, num_nodes)
        } 
    
class AddSolvent(nn.Module):
    def __init__(self, gen_config, env_config):
        super(AddSolvent, self).__init__()
        self.name = "add_solvent"
        self.gen_config = gen_config
        self.env_config = env_config
        self.latent_dim = gen_config.latent_dim 
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.type_embed = nn.Linear(in_features= 3, out_features = self.latent_dim, bias= True)


        component_data = env_config.components_tensor.float()
        component_mean = component_data.mean(dim=0)
        component_std = component_data.std(dim=0).clamp_min(1e-6)

        self.norm_component = {}
        self.norm_component["component_mean"] = component_mean
        self.norm_component["component_std"] = component_std
        self.norm_component["components_properites"] = component_data

        self.prediction_mlp = nn.Sequential(
            nn.Linear(self.latent_dim + 3, 2 * self.latent_dim),
            nn.SiLU(),
            nn.Linear(2 * self.latent_dim, 1 + len(self.env_config._amount_grid)) # for logit and amount discretized to 100 categories
        )

    def embed(self, x):
        component_tuple = x["output_flows"]["out0"]
        component_tuple_tensor = torch.from_numpy(component_tuple).to(dtype = torch.float32, device = self.gen_config.training_device)
        return self.type_embed(component_tuple_tensor)

    def predict(self, x: torch.FloatTensor, valid_avail_solvent_mask= None):

        # x is the embedding of shape (batch_size, num_nodes, latent_dim)
        # components is of shape (num_possible_components, 3)

        # all possible components 
        global_components = self.env_config.components_tensor.to(device=self.gen_config.training_device)
        components = (global_components - self.norm_component["component_mean"].to(x.device, x.dtype)) / self.norm_component["component_std"].to(x.device, x.dtype)

        batch_size, num_nodes, _ = x.size()
        num_components, _ = components.size()

        components = components[None, ...].repeat(num_nodes, 1, 1)
        components = components[None, ...].repeat(batch_size, 1, 1, 1) # (batch_size, num_nodes, num_possible_components, 3)

        mlp_in = x[:, :, None, :].repeat(1, 1, num_components, 1)
        mlp_in = torch.cat([mlp_in, components], dim = -1) #(batch_size, num_nodes, num_componets, 4 + latent_dim)
        mlp_out = self.prediction_mlp(mlp_in) #(batch_size, num_nodes, num_components, 1 + 100 )
        choose_solvent_logits = mlp_out[:, :, :, 0]
        choose_solvent_logits = choose_solvent_logits.masked_fill(~valid_avail_solvent_mask[:, None, :], float("-inf"))

        choose_solvent_amount_logits = mlp_out[:, :, :, 1:]
        choose_solvent_amount_logits = choose_solvent_amount_logits.masked_fill(~valid_avail_solvent_mask[:, None, :, None], float("-inf"))

        return {
        "picked_logit": self.logit_linear(x), # (batch_size, num_nodes, 1)
        "component_logit": choose_solvent_logits.squeeze(-1), # (batch_size. num_nodes, num_components) <- distribution over which component to add 
        "component_amount": choose_solvent_amount_logits # (batch_size, num_nodes, num_components, 100)
        } 

class ComponentsExpert(nn.Module):
    def __init__(self, gen_config, env_config):
        super(ComponentsExpert, self).__init__()
        self.gen_config = gen_config 
        self.env_config = env_config
        self.flow_latent_dim = gen_config.flow_latent_dim

        self.component_linear = nn.Linear(in_features=self.env_config.max_number_of_components, out_features = self.flow_latent_dim)
        self.edge_linear = nn.Linear(in_features = 1, out_features= self.flow_latent_dim)

    def flat_gamma_to_matrix(self, system_gamma_inf, num_components):
        flat = torch.tensor(system_gamma_inf, dtype=torch.float32, device=self.gen_config.training_device)
        gamma_local = torch.zeros((num_components, num_components), device=self.gen_config.training_device)
        idx = 0
        for i in range(num_components):
            for j in range(i + 1, num_components):
                if i != j:
                    gamma_local[i, j] = flat[idx]
                    gamma_local[j, i] = flat[idx + 1]
                    idx += 2
        return gamma_local
    
    def forward(self, system_pure_crit, system_gammas_inf, x):

        # component_params: (batch_size, num_components, 3)
        # interaction_params: (batch_size, num_components, num_components, 1)
        compon_tensor = torch.tensor(system_pure_crit).reshape(self.env_config.max_number_of_components, 3).to(self.gen_config.training_device) #convert to tensor and add dim for batch

        gamma = self.flat_gamma_to_matrix(system_gammas_inf, self.env_config.max_number_of_components)
        interaction_params = gamma.unsqueeze(-1).to(device=self.gen_config.training_device)

        return self.component_linear(compon_tensor), self.edge_linear(interaction_params)

    def forward_raw(self, system_pure_crit: torch.Tensor, interaction_params: torch.Tensor):
        """Encode an already padded batch of physical property inputs."""
        return self.component_linear(system_pure_crit), self.edge_linear(interaction_params)
    
class FlowExpert(nn.Module):
    def __init__(self, gen_config, env_config):
        super(FlowExpert, self).__init__()
        self.gen_config = gen_config 
        self.env_config = env_config
        self.flow_latent_dim = gen_config.flow_latent_dim
        self.num_trf_blocks = gen_config.num_trf_flow_blocks
        self.flow_latent_upscale = nn.Linear(self.flow_latent_dim, self.gen_config.latent_dim, bias = True)

        self.blocks = nn.ModuleList([
            CoreTransformerEncoder(d_model= self.flow_latent_dim, nhead=4, dropout=self.gen_config.dropout)
            for _ in range(self.gen_config.num_trf_flow_blocks)
        ])

    def forward(self, x, component_emb, interaction_emb):

        # component_emb: from ComponentExpert of (batch_size, num_components, k)
        # interaction_emb: from ComponentExpert of (batch_size, num_components, num_components, k)
        # x: amount (batch_size, num_components, 1)

        amount_outlets = torch.stack([torch.tensor(
                x.get("output_flows", {}).get(k, [0.0, 0.0, 0.0]),
                device=self.gen_config.training_device, dtype=torch.float32
            )
            for k in self.env_config.outlet_to_idx.keys()])# (batch_size, 2, num_comp)

        amount_outlets = amount_outlets.unsqueeze(-1) # (num_outlets, num_componets, 1)
        component_emb = component_emb.unsqueeze(0) # (1, num_componets, flow_latent_dim)

        nodes = component_emb * amount_outlets # (num_outlets, num_components, flow_latent_dim)
        edges = interaction_emb.unsqueeze(0).repeat(amount_outlets.shape[0], 1, 1, 1) # duplicate twice (num_outlets, num_components, num_components, flow_latent_dim)

        for block in self.blocks:
            nodes = block(nodes, edges)
        
        flow_embedding = nodes.mean(dim=1) # (num_outlets, flow_latent_dim)
        return self.flow_latent_upscale(flow_embedding) # (num_outlets, latent_dim)

    def forward_raw(self, output_flows: torch.Tensor, component_emb: torch.Tensor,
                    interaction_emb: torch.Tensor) -> torch.Tensor:
        """Encode padded node/outlet flow amounts in one batched operation."""
        batch_size, num_nodes, num_outlets, num_components = output_flows.shape
        amount_outlets = output_flows.reshape(
            batch_size * num_nodes * num_outlets, num_components, 1
        )
        component_emb = component_emb[:, None, None, :, :].expand(batch_size, num_nodes, num_outlets, -1, -1).reshape(batch_size * num_nodes * num_outlets, num_components, -1)
        interaction_emb = interaction_emb[:, None, None, :, :, :].expand(batch_size, num_nodes, num_outlets, -1, -1, -1).reshape(batch_size * num_nodes * num_outlets, num_components, num_components, -1)

        nodes = component_emb * amount_outlets
        edges = interaction_emb
        for block in self.blocks:
            nodes = block(nodes, edges)

        flow_embedding = nodes.mean(dim=1)
        return self.flow_latent_upscale(flow_embedding).reshape(
            batch_size, num_nodes, num_outlets, -1
        )
    
class EdgeFlowExpert(nn.Module):
    def __init__(self, config, flow_expert: FlowExpert):
        super(EdgeFlowExpert, self).__init__()
        object.__setattr__(self, "_flow_expert_ref", flow_expert)
        self.config = config
        self.latent_dim = self.config.latent_dim
        self.is_recycle_emb = nn.Embedding(num_embeddings = 2, embedding_dim = self.latent_dim) # 0 for no, 1 for yes

    def forward(self, edge_exists: bool, is_recycle: bool, edge=None, feed_emb = None):

        # if an edge exists (in cases of a single open stream or virtual node)
        #edge_idx = torch.tensor(1, dtype=torch.long, device=self.config.training_device)

        # if its a recycle 
        recycle_idx = torch.tensor(1 if is_recycle else 0, dtype=torch.long, device=self.config.training_device)
        #edge_emb = self.no_edge_emb(edge_idx)
        recycle_emb = self.is_recycle_emb(recycle_idx)

        combine_edge_embed = recycle_emb + feed_emb

        return combine_edge_embed
    

class OpenStreamExpert(nn.Module):
    def __init__(self, gen_config, env_config, flow_expert: FlowExpert):
        super(OpenStreamExpert, self).__init__()
        object.__setattr__(self, "_flow_expert_ref", flow_expert)
        self.gen_config = gen_config
        self.env_config = env_config
        self.latent_dim = self.gen_config.latent_dim 
        self.linear_transform_open_stream = nn.Linear(self.latent_dim, self.latent_dim, bias = True)

    def forward(self, x, compon_emb, interaction_emb):
        latent_flow = self._flow_expert_ref(x, compon_emb, interaction_emb)
        open_stream_embed = self.linear_transform_open_stream(latent_flow)

        return latent_flow, open_stream_embed
