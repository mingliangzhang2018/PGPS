import torch.nn as nn


class LSTM(nn.Module):
    
    def __init__(self, cfg):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=cfg.WORD_EMBED_SIZE,
            hidden_size=cfg.HIDDEN_SIZE, # int(hidden_size / num_directions),
            num_layers=cfg.NUM_LAYERS,
            batch_first=cfg.BATCH_FIRST,  # first dim is batch_size or not
            bidirectional=cfg.BIDIRECTIONAL
        )

    def forward(self, input, h0, c0):
        output, (hn, cn) = self.lstm(input, (h0, c0))
        return output, hn, cn




