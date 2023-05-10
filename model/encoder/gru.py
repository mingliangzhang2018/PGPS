import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, cfg):
        super(GRU, self).__init__()

        self.is_bidirectional = True
        self.batch_first = True
        self.gru = nn.GRU(
            input_size = cfg.encoder_embedding_size,
            hidden_size = cfg.encoder_hidden_size, # int(hidden_size / num_directions),
            num_layers = cfg.encoder_layers,
            bidirectional = self.is_bidirectional,
            dropout = cfg.dropout_rate,
            batch_first = self.batch_first
        )
        self.hidden_size = cfg.encoder_hidden_size
        self.dropout = nn.Dropout(cfg.dropout_rate)
    
    def forward(self, src_emb, input_lengths, hidden=None):

        input_emb = self.dropout(src_emb)
        # input_emb = src_emb
        packed = nn.utils.rnn.pack_padded_sequence(input_emb, input_lengths.cpu(), \
                                            batch_first=self.batch_first, enforce_sorted=False)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru(packed, pade_hidden)
        pade_outputs, _ = nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=self.batch_first)
        # pade_outputs [B, S, hidden_size*num_directions] 
        # pade_hidden [n_layers*num_directions, B, hidden_size]
        if self.is_bidirectional: 
            pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # B x S x H
            pade_hidden = pade_hidden[0::2, :, :] + pade_hidden[1::2, :, :]

        return pade_outputs, pade_hidden




