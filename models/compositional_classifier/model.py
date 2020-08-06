import torch
import torch.nn as nn


class CompositionalClassifier(nn.Module):
    def __init__(self, phrases_as_token_idx_lists,
                 M=400, len_vocab=831, rnn_n_layers=1):
        super(CompositionalClassifier, self).__init__()
        D = 2048
        self.img_encoder = nn.Linear(D, M)  # could be more complicated
        self.phrase_encoder = RNN(M, len_vocab, rnn_n_layers)
        self.phrases_as_token_idx_lists = phrases_as_token_idx_lists

    def forward(self, img_feats, phrase_batch_in, phrase_lengths):
        img_encoding = self.img_encoder(img_feats)  # BxM
        ph_encoding = self.phrase_encoder(self.phrases_as_token_idx_lists, phrase_lengths, phrase_batch_in)  # NxM
        ph_encoding_t = ph_encoding.permute(1, 0)  # MxN
        pred_score = torch.mm(img_encoding, ph_encoding_t)
        return pred_score
