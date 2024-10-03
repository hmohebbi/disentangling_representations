import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

def unweighted_average(x, m):
    m = m.unsqueeze(-1).expand_as(x)  
    effective_x = x * m  
    sum_effective_x = torch.sum(effective_x, dim=1)
    pooled_x = sum_effective_x / torch.sum(m, dim=1)
    return pooled_x

class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = torch.nn.Parameter(torch.ones(1, embed_dim), requires_grad=True)

    def forward(self, x, attention_mask=None, output_attentions=False):
        """Input shape: Batch x Time x Hidden Dim"""
       
        scores = torch.matmul(x, self.query.unsqueeze(0).transpose(-2, -1)).squeeze(-1)  # (batch_size, seq_len)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == False, float('-inf')) 

        attn_weights = torch.nn.functional.softmax(scores, dim=-1).unsqueeze(-1)  
        
        # weighted average
        pooled_x = torch.sum(attn_weights * x, dim=1) 
        
        outputs = (pooled_x, attn_weights) if output_attentions else (pooled_x,)
        return outputs
    

@dataclass
class VIBConfig(): 
    input_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    num_classes: Optional[int] = None
    stage: Optional[str] = None
    layer_weight_averaging: Optional[bool] = False
    num_layers: Optional[int] = None

class VariationalEncoder(torch.nn.Module):
    def __init__(self, config):
        super(VariationalEncoder, self).__init__()
        self.enc1 = torch.nn.Linear(config.input_dim, config.input_dim)
        self.enc2 = torch.nn.Linear(config.input_dim, config.input_dim)
        self.mu = torch.nn.Linear(config.input_dim, config.latent_dim)
        self.var = torch.nn.Linear(config.input_dim, config.latent_dim)

    def forward(self, h):
        o = F.gelu(self.enc1(h))
        o = F.gelu(self.enc2(o))
        
        mu = self.mu(o)
        var = F.softplus(self.var(o)) # to generate positive values
        
        return mu, var

class FrameDecoder(torch.nn.Module):
    def __init__(self, config):
        super(FrameDecoder, self).__init__()
        self.lm_head = torch.nn.Linear(config.latent_dim, config.num_classes)
    
    def forward(self, z, m=None, cond=None, output_attentions=False):
        logits = self.lm_head(z)
        return (logits,)
    
class UtteranceDecoder(torch.nn.Module):
    def __init__(self, config):
        super(UtteranceDecoder, self).__init__()
        self.cond_attention = SelfAttention(config.latent_dim)
        self.z_attention = SelfAttention(config.latent_dim)
        self.clf = torch.nn.Linear(config.latent_dim*2, config.num_classes) # latent_dim * 2 due to concatenation
    
    def forward(self, z, m, cond, output_attentions=False):
        # pooled_cond = unweighted_average(cond, m)
        # pooled_z = unweighted_average(z, m)
        attn_out_cond = self.cond_attention(cond, m, output_attentions=output_attentions)
        attn_out_z = self.z_attention(z, m, output_attentions=output_attentions)
        pooled_cond, pooled_z = attn_out_cond[0], attn_out_z[0]
        concatenated_pooled = torch.concat([pooled_cond, pooled_z], dim=-1)

        logits = self.clf(concatenated_pooled)
        
        outputs = (logits, attn_out_cond[1], attn_out_z[1]) if output_attentions else (logits,)
        return outputs

class VIB(torch.nn.Module):
    def __init__(self, config):
        super(VIB, self).__init__()
        self.layer_weight_averaging = config.layer_weight_averaging
        if self.layer_weight_averaging:
            self.layer_weights = torch.nn.Parameter(torch.ones(config.num_layers)/config.num_layers, requires_grad=True)

        self.encoder = VariationalEncoder(config)

        if config.stage == "1":
            self.decoder = FrameDecoder(config)
        elif config.stage == "2":
            self.decoder = UtteranceDecoder(config)
        else:
            raise ValueError("Invalid VIB training stage!")

    def forward(self, h, m=None, cond=None, output_attentions=False, noise=True): 
        if self.layer_weight_averaging:
            # compute weighted sum over layers
            w = torch.nn.functional.softmax(self.layer_weights, dim=0)
            h = torch.sum(h * w.view(1, w.shape[0], 1, 1), dim=1)

        mu, var = self.encoder(h)
        std = var ** 0.5
        # # reparameterization trick: introducing epsilon only during training, and use the z = mu during inference
        if self.training and noise:
            eps = torch.randn_like(std) # sample from N(0, 1)
            z = mu + std * eps 
        else:
            z = mu

        # decoing
        outputs = self.decoder(z, m, cond, output_attentions)
        
        return outputs + (mu, var)
