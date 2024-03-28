from dataclasses import dataclass
import torch
import torch.nn as nn
from attention import Attention

@dataclass
class GRUDecoder(torch.nn.Module):
    num_input: int
    num_hidden: int
    enc_ncontext: int
    natt: int
    nreadout: int
    readout_dropout: float

    gru1: nn.GRUCell
    gru2: nn.GRUCell
    enc_attn: Attention
    embedding2out: nn.Linear
    hidden2out: nn.Linear
    c2o: nn.Linear
    readout_dp: nn.Dropout

    def __post_init__(self):
        super(GRUDecoder, self).__init__()
        self.gru1 = nn.GRUCell(self.num_input, self.num_hidden)
        self.gru2 = nn.GRUCell(self.enc_ncontext, self.num_hidden)
        self.enc_attn = Attention(self.num_hidden, self.enc_ncontext, self.natt)
        self.embedding2out = nn.Linear(self.num_input, self.nreadout)
        self.hidden2out = nn.Linear(self.num_hidden, self.nreadout)
        self.c2o = nn.Linear(self.enc_ncontext, self.nreadout)
        self.readout_dp = nn.Dropout(self.readout_dropout)
    
    def forward(self, emb, hidden, enc_mask, enc_context):
        hidden = self.gru1(emb, hidden)
        attn_enc = self.enc_attn(hidden, enc_mask, enc_context)
        hidden = self.gru2(attn_enc, hidden)
        
        output = self.embedding2out(emb) + self.hidden2out(hidden) + self.c2o(attn_enc)
        output = torch.tanh(output)
        output = self.readout_dp(output)
        
        return output, hidden