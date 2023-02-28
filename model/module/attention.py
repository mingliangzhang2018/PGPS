import torch
import torch.nn as nn

class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, candi_embeddings, candi_mask=None):
        '''
        Arguments:
            hidden: B x 1 x 2H
            candi_embeddings: B x candi_size x H
            candi_mask: B x candi_size
        Return:
            score: B x candi_size
        '''
        hidden = hidden.repeat(1, candi_embeddings.size(1), 1)  # B x candi_size x H
        # For each position of encoder outputs
        energy_in = torch.cat((hidden, candi_embeddings), 2)  # B x candi_size x 3H
        score = self.score(torch.tanh(self.attn(energy_in))).squeeze(-1)  # B x candi_size
        if candi_mask is not None:
            score = score.masked_fill_(~candi_mask, -1e12)
        return score

class Attn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        '''
        Arguments:
            hidden: B x 1 x H (q)
            encoder_outputs: B x S x H
            seq_mask: B x S
        Return:
            attn_energies: B x S
        '''
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)  # B x S x H
        energy_in = torch.cat((hidden, encoder_outputs), 2) # B x S x 2H
        score_feature = torch.tanh(self.attn(energy_in)) # B x S x H
        attn_energies = self.score(score_feature).squeeze(-1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(~seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies

class Score_Multi(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score_Multi, self).__init__()
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, candi_embeddings, candi_mask=None):
        '''
        Arguments:
            hidden: B x S x H
            candi_embeddings: B x candi_size x H
            candi_mask: B x candi_size
        Return:
            score: B x S x candi_size
        '''
        hidden = hidden.unsqueeze(2).repeat(1, 1, candi_embeddings.size(1), 1) # B x S x candi_size x H
        candi_embeddings = candi_embeddings.unsqueeze(1).repeat(1, hidden.size(1), 1, 1) # B x S x candi_size x H
        candi_mask = candi_mask.unsqueeze(1).repeat(1, hidden.size(1), 1) # B x S x candi_size
        energy_in = torch.cat((hidden, candi_embeddings), -1)  # B x S x candi_size x 2H
        score = self.score(torch.tanh(self.attn(energy_in))).squeeze(-1)  # B x S x candi_size
        if candi_mask is not None:
            score = score.masked_fill_(~candi_mask, -1e12)
        return score