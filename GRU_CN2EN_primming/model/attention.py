from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class Attention(nn.Module):
    num_hidden: int
    ncontext: int
    natt: int
    
    h2s: nn.Linear 
    s2s: nn.Linear
    a2o: nn.Linear
    
    def __post_init__(self):
        super(Attention, self).__init__()
        self.h2s = nn.Linear(self.num_hidden, self.natt)
        self.s2s = nn.Linear(self.ncontext, self.natt)
        self.a2o = nn.Linear(self.natt, 1)
        
    def forward(self, hidden, mask, context):
        batch_size, context_len, _ = context.size()
    
        attn_h = self.s2s(context.reshape(batch_size * context_len, -1))
        attn_h = attn_h.view(batch_size, context_len, -1)
        attn_h += self.h2s(hidden).unsqueeze(1).expand_as(attn_h)
        
        logit = self.a2o(torch.tanh(attn_h)).squeeze(-1)
        
        if mask is not None:
            logit = logit.masked_fill(~mask, -float("inf"))
            
        softmax = nn.functional.softmax(logit, dim=1)
        output = torch.bmm(softmax.unsqueeze(1), context).squeeze(1)
        
        return output