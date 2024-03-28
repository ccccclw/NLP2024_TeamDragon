from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class GRUEncoder(torch.nn.Module):

    num_input: int
    num_hidden: int
    num_token: int
    padding_idx: int
    emb_dropout: float
    hid_dropout: float
    
    embedding: nn.Sequential
    gru: nn.GRU
    dropout: nn.ModuleList

    def __post_init__(self):
        super(GRUEncoder, self).__init__()
        
        self.embedding = nn.Sequential(
            nn.Embedding(self.num_token, self.num_input, padding_idx=self.padding_idx),
            nn.Dropout(self.emb_dropout)
        )
        
        self.gru = nn.GRU(self.num_input, self.num_hidden, 1, batch_first=True, bidirectional=True)
        
        self.dropout = nn.ModuleList([
            nn.Dropout(self.emb_dropout),
            nn.Dropout(self.hid_dropout)
        ])


    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        h0 = weight.new_zeros(2, batch_size, self.num_hidden)
        return h0
    
    def forward(self, input, mask):
        batch_size = input.size(0)
        hidden = self.init_hidden(batch_size)
        
        input = self.emb(input)
        input = self.enc_emb_dp(input)
        
        lengths = mask.sum(dim=1).cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, hidden = self.bi_gru(packed_input, hidden)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.enc_hid_dp(output)
        
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        
        return output, hidden