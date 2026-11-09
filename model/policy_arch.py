import torch
from torch import nn
from model.core_transformer_block import CoreTransformerEncoder
from experts.experts import (
    AddSolvent,
    ComponentsExpert,
    Decanter,
    DistillationColumn,
    EdgeFlowExpert,
    FlowExpert,
    Mixer,
    OpenStreamExpert,
    Recycler,
    Split,
)


class StateEncoder(nn.Module):
    
    """Encode raw simulator state for the shared policy.
    """

    def __init__(self, gen_config, env_config):
        super().__init__()
        self.gen_config = gen_config
        self.env_config = env_config
        self.latent_dim = gen_config.latent_dim 

        self.component_expert = ComponentsExpert(gen_config, env_config)
        self.flow_expert = FlowExpert(gen_config, env_config)
        self.edge_expert = EdgeFlowExpert(gen_config, self.flow_expert)
        self.open_stream_expert = OpenStreamExpert(gen_config, env_config, self.flow_expert)

        #self.component_identity_embedding = nn.Linear(len(env_config.component_names), self.latent_dim)
        #self.feed_composition_projection = nn.Linear(len(env_config.component_names), self.latent_dim)
        
        self.unit_experts = nn.ModuleDict({
            "distillation_column": DistillationColumn(gen_config, env_config),
            "decanter": Decanter(gen_config),
            "mixer": Mixer(gen_config),
            "split": Split(gen_config, env_config),
            "recycle": Recycler(gen_config),
            "add_solvent": AddSolvent(gen_config, env_config),
        })

    def forward(self, batch):
        properties = batch["batch_component_properties"]
        interactions = batch["batch_component_interactions"]
        node_flows = batch["batch_node_output_flows"]
        node_types = batch["batch_node_unit_types"]
        node_params = batch["batch_node_continuous_params"]
        edge_exists = batch["batch_edge_exists"]
        edge_recycle = batch["batch_edge_recycle"]
        edge_outlets = batch["batch_edge_outlet_indices"]

        component_emb, interaction_emb = self.component_expert.forward_raw(properties, interactions)
        flow_embeds = self.flow_expert.forward_raw(node_flows, component_emb, interaction_emb)
        #open_stream_embeds = self.open_stream_expert.linear_transform_open_stream(flow_embeds)

        batch_size, num_nodes, _, _ = flow_embeds.shape
        node_embeds = torch.zeros(batch_size, num_nodes, self.latent_dim, dtype=flow_embeds.dtype, device=flow_embeds.device)

        # Feed nodes retain the existing physical-flow representation.  Unit
        # nodes retain their existing type/parameter-specific embedding paths.
        feed_mask = node_types < 0
        node_embeds = torch.where(feed_mask[..., None], flow_embeds[:, :, 0, :], node_embeds)
        for unit_name, unit_idx in zip(self.env_config.units_map_indices_type, range(self.env_config.num_units)):
            mask = node_types == unit_idx
            if unit_name == "distillation_column":
                encoded = self.unit_experts[unit_name].df_and_type_embed(node_params[..., 0, None])
            elif unit_name == "split":
                encoded = self.unit_experts[unit_name].split_ratio_and_type_embed(node_params[..., 1, None])
            elif unit_name in ("decanter", "mixer"):
                encoded = self.unit_experts[unit_name].type_embed.weight[0].view(1, 1, -1).expand(batch_size, num_nodes, -1)
            elif unit_name == "add_solvent":
                encoded = self.unit_experts[unit_name].type_embed(node_flows[:, :, 0, :])
            else:
                continue
            node_embeds = torch.where(mask[..., None], encoded, node_embeds)

        # Edge features use the source outlet's physical representation and
        # preserve the original recycle embedding and zero absent-edge path.
        edge_count = edge_exists.shape[2]
        parallel_edge_count = edge_exists.shape[3]
        edge_source_flows = flow_embeds.unsqueeze(2).unsqueeze(3).expand(-1, -1, edge_count, parallel_edge_count, -1, -1)
        source_flow = torch.gather(edge_source_flows,4, edge_outlets.clamp_min(0).unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, -1, 1, self.latent_dim
            ),
        ).squeeze(4)
        recycle_emb = self.edge_expert.is_recycle_emb(edge_recycle.long())
        per_edge_embeds = torch.where(edge_exists[..., None], source_flow + recycle_emb, torch.zeros_like(source_flow))
        parallel_edge_count = edge_exists.sum(dim=3, keepdim=True)
        edge_embeds = per_edge_embeds.sum(dim=3) / parallel_edge_count.clamp_min(1)
        
        edge_embeds_with_virtual = torch.zeros(
            batch_size, num_nodes + 1, num_nodes + 1, self.latent_dim,
            dtype=edge_embeds.dtype, device=edge_embeds.device,
        )
        edge_embeds_with_virtual[:, 1:, 1:] = edge_embeds

        return {
            "batch_latent_nodes_embeds": node_embeds,
            "batch_latent_edges_embeds": edge_embeds_with_virtual,
            "batch_latent_open_streams_embeds": flow_embeds,
        }


class FlowsheetNetwork(nn.Module):
    
    def __init__(self, gen_config, env_config, device: torch.device = None):
        super().__init__()
        
        self.device = torch.device("cpu") if device is None else device
        self.latent_dim = gen_config.latent_dim
        self.num_heads = gen_config.num_heads
        self.num_blocks = gen_config.num_transformer_blocks
        self.env_config = env_config

        # First build up Transformer Encoder using blocks 
        self.core_transformer = nn.ModuleList([])
        for _ in range(self.num_blocks):
            block = CoreTransformerEncoder(d_model = self.latent_dim, nhead = self.num_heads, 
                                           dropout=gen_config.dropout, 
                                           clip_value = 10)
            self.core_transformer.append(block)

        self.virtual_node_embedding = nn.Embedding(num_embeddings = 4, embedding_dim=self.latent_dim) # decisions across 4 lvls 
        self.rewards_projector = nn.Linear(1, self.latent_dim)

        #----heads----#
        self.terminate_head = nn.Linear(self.latent_dim, 1)
        self.open_stream_head = nn.Linear(self.latent_dim, 1)
        
        self.state_encoder = StateEncoder(gen_config, env_config).to(self.device)

    def forward(self, x):
        
        encoded_state = self.state_encoder(x)
        batch_latent_nodes_embed = encoded_state["batch_latent_nodes_embeds"]
        batch_latent_edges_embed = encoded_state["batch_latent_edges_embeds"]
        batch_latent_open_streams_embeds = encoded_state["batch_latent_open_streams_embeds"]
        valid_avail_solvent_mask = x["valid_avail_solvent_mask"]
        valid_nodes = x["valid_nodes"] # for padding for additive attention
        terminate_or_open_stream_logits = {}
        rewards = x["rewards"]

        # Create attentive masks for transformer 
        padding_attn_mask = (~valid_nodes[:, :, None] | ~valid_nodes[:, None, :]) # (B, N, N)
        padding_attn_mask = padding_attn_mask.float() * -1e9
    
        level_embed = self.virtual_node_embedding(x['levels']).unsqueeze(1) # only virtual residue gets level info
        batch_latent_nodes_embed = torch.cat([level_embed, batch_latent_nodes_embed], dim=1) # (B, N + 1, d)
        batch_latent_nodes_embed = batch_latent_nodes_embed.masked_fill(~valid_nodes.unsqueeze(-1), 0.0)

        # Send node and edge embeddings to core transformer 
        for block in self.core_transformer:
            batch_latent_nodes_embed = block(batch_latent_nodes_embed, batch_latent_edges_embed, padding_attn_mask)

        # lvl 0: make predictions whether to terminate or open stream
        latent_virtual_node = batch_latent_nodes_embed[:, 0, :] # (B, d)
        terminate_logits = self.terminate_head(latent_virtual_node) # (B,) 

        latent_nodes_transformed = batch_latent_nodes_embed[:, 1:, :] # (B, N, d) 

        # Now provide stream embedding to open_stream head to get logits   
        latent_nodes_for_streams = latent_nodes_transformed.unsqueeze(2).expand(-1, -1, 2, -1) # add extra dimension and duplicate embed
        open_stream_embeds = batch_latent_open_streams_embeds + latent_nodes_for_streams
        open_stream_logits = self.open_stream_head(open_stream_embeds).squeeze(-1)
        open_stream_logits = open_stream_logits.masked_fill(~x["open_stream_mask"], -1e9) # mask out all non valid streams from padding
        
        terminate_or_open_stream_logits['terminate_logits'] = terminate_logits
        terminate_or_open_stream_logits['open_stream_logits'] = open_stream_logits

        # lvl 1: if open stream, logits for predictions for units 
        reward_embedding = self.rewards_projector(rewards.unsqueeze(-1))
        latent_nodes_transformed = latent_nodes_transformed + reward_embedding
        unit_predictions = {}
        for unit_type, expert in self.state_encoder.unit_experts.items():
            if unit_type == "distillation_column":
                unit_predictions[unit_type] = expert.predict(
                    latent_nodes_transformed)
                
            elif unit_type == "add_solvent":
                unit_predictions[unit_type] = expert.predict(
                    latent_nodes_transformed,
                    valid_avail_solvent_mask
                )
            elif unit_type not in ["mixer", "recycle", "flow_expert"]:
                unit_predictions[unit_type] = expert.predict(latent_nodes_transformed)
            elif unit_type == "recycle":
                unit_predictions[unit_type] = expert.predict(latent_nodes_transformed, x['recycler_masks'])
            elif unit_type == "mixer":
                unit_predictions[unit_type] = expert.predict(latent_nodes_transformed, x['mixer_masks'])
            
        return terminate_or_open_stream_logits, unit_predictions
    
    
    def get_weights(self):
        return dict_to_cpu(self.state_dict())
    
    
def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict
