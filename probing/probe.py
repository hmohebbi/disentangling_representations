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
class ProbeConfig(): 
    input_dim: Optional[int] = None
    num_classes: Optional[int] = None
    objective: Optional[str] = None
    layer_weight_averaging: Optional[bool] = False
    num_layers: Optional[int] = None
    
class FrameProbe(torch.nn.Module):
    def __init__(self, config):
        super(FrameProbe, self).__init__()
        self.lm_head = torch.nn.Linear(config.input_dim, config.num_classes)
    
    def forward(self, h, m=None, output_attentions=False):
        logits = self.lm_head(h)
        return (logits,)
    
class UtteranceProbe(torch.nn.Module):
    def __init__(self, config):
        super(UtteranceProbe, self).__init__()
        self.attention = SelfAttention(config.input_dim)
        self.clf = torch.nn.Linear(config.input_dim, config.num_classes)
    
    def forward(self, h, m, output_attentions=False):
        # pooled_h = unweighted_average(h, m)
        attn_out = self.attention(h, m, output_attentions=output_attentions)
        pooled_h = attn_out[0]
        
        logits = self.clf(pooled_h)
        
        outputs = (logits, attn_out[1]) if output_attentions else (logits,)
        return outputs
    

class Probe(torch.nn.Module):
    def __init__(self, config):
        super(Probe, self).__init__()
        self.layer_weight_averaging = config.layer_weight_averaging
        if self.layer_weight_averaging:
            self.layer_weights = torch.nn.Parameter(torch.ones(config.num_layers)/config.num_layers, requires_grad=True)

        if config.objective == "ctc":
            self.clf = FrameProbe(config)
        else:
            self.clf = UtteranceProbe(config)

    def forward(self, h, m=None, output_attentions=False):
        if self.layer_weight_averaging:
            # compute weighted sum over layers
            w = torch.nn.functional.softmax(self.layer_weights, dim=0)
            h = torch.sum(h * w.view(1, w.shape[0], 1, 1), dim=1)
        
        outputs = self.clf(h, m, output_attentions)
        return outputs
    