import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import BertTokenizer, BertModel

from data_api.dataset_api import WordEncoder
from models.layers.util import print_tensor_stats


def make_encoder(method, word_emb_dim=0, word_encoder=None):
    if method == 'mean':
        return MeanEncoder(word_emb_dim=word_emb_dim, word_encoder=word_encoder)
    elif method == 'lstm':
        return LSTMEncoder(word_emb_dim=word_emb_dim, word_encoder=word_encoder)
    elif method == 'elmo':
        return ElmoEncoder()
    elif method == 'bert':
        return BertEncoder()
    else:
        raise NotImplementedError


class MeanEncoder(nn.Module):
    def __init__(self, word_emb_dim, word_encoder=None):
        super(MeanEncoder, self).__init__()
        self.word_encoder = word_encoder
        if self.word_encoder is None:
            self.word_encoder = WordEncoder()
        self.embeds = nn.Embedding(num_embeddings=len(self.word_encoder.word_list), embedding_dim=word_emb_dim)
        self.out_dim = word_emb_dim

    def forward(self, sentences):
        """
        sentences: list[str], len of list: B
        output: mean_embed: including embed of <end>, not including <start>
        """
        device = self.embeds.weight.device
        # encoded = list()
        # for sent in sentences:
        #     e = self.word_encoder.encode(sent, max_len=-1)
        #     t = torch.as_tensor(e, dtype=torch.long, device=device)
        #     encoded.append(t)
        encoded = [torch.as_tensor(self.word_encoder.encode(sent, max_len=-1)[0], dtype=torch.long, device=device)
                   for sent in sentences]
        sent_lengths = torch.as_tensor([len(e) for e in encoded], dtype=torch.long, device=device)
        sent_end_ids = torch.cumsum(sent_lengths, dim=0)
        sent_start_ids = torch.empty_like(sent_end_ids)
        sent_start_ids[0] = 0
        sent_start_ids[1:] = sent_end_ids[:-1]

        encoded = torch.cat(encoded)
        embeded = self.embeds(encoded)  # sum_len x E
        sum_embeds = torch.cumsum(embeded, dim=0)  # sum_len x E
        sum_embed = sum_embeds.index_select(dim=0, index=sent_end_ids - 1) - \
                    sum_embeds.index_select(dim=0, index=sent_start_ids)  # exclude <start>
        mean_embed = sum_embed / sent_lengths.unsqueeze(-1).float()  # B x E
        return mean_embed


class LSTMEncoder(nn.Module):
    def __init__(self, word_emb_dim, hidden_dim=256, bi_direct=True, word_encoder=None):
        super(LSTMEncoder, self).__init__()
        self.word_encoder = word_encoder
        if self.word_encoder is None:
            self.word_encoder = WordEncoder()
        self.embeds = nn.Embedding(num_embeddings=len(self.word_encoder.word_list), embedding_dim=word_emb_dim,
                                   padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=word_emb_dim, hidden_size=hidden_dim, bidirectional=bi_direct,
                                  batch_first=True, num_layers=1, bias=True, dropout=0.0)
        self.out_dim = hidden_dim
        if bi_direct:
            self.out_dim = hidden_dim * 2
        # self.lstm = PytorchSeq2VecWrapper(module=lstm_module)
        # self.out_dim = self.lstm.get_output_dim()

    def forward(self, sentences):
        """
        sentences: list[str], len of list: B
        output sent_embs: Tensor B x OUT
        """
        device = self.embeds.weight.device
        encode_padded, lens = self.word_encoder.encode_pad(sentences)
        encode_padded = torch.as_tensor(encode_padded, dtype=torch.long, device=device)  # B x max_len
        word_embs = self.embeds(encode_padded)  # B x max_len x E
        packed = pack_padded_sequence(word_embs, lens, batch_first=True, enforce_sorted=False)
        lstm_embs_packed, _ = self.lstm(packed)
        lstm_embs, lens = pad_packed_sequence(lstm_embs_packed, batch_first=True)  # B x max_len x OUT
        sent_embs = torch.stack([lstm_embs[i, lens[i] - 1] for i in range(len(lens))])
        # print_tensor_stats(sent_embs, 'sent_embs')
        return sent_embs


class ElmoEncoder(nn.Module):
    def __init__(self, options_file="cache/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                 weight_file="cache/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"):
        super(ElmoEncoder, self).__init__()
        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        self.elmo = Elmo(options_file, weight_file, num_output_representations=2, dropout=0)
        self.out_dim = 1024

    def forward(self, sentences, device='cuda'):
        """
        sentences: list[str], len of list: B
        output sent_embs: Tensor B x OUT
        """
        sentences = [WordEncoder.tokenize(s) for s in sentences]
        # sentences = [['First', 'sentence', '.'], ['Another', '.']]
        # use batch_to_ids to convert sentences to character ids
        character_ids = batch_to_ids(sentences).to(device)
        embeddings = self.elmo(character_ids)
        # embeddings['elmo_representations'] is length two list of tensors.
        # Each element contains one layer of ELMo representations with shape
        # (2, 3, 1024).
        #   2    - the batch size
        #   3    - the sequence length of the batch
        #   1024 - the length of each ELMo vector
        sent_embeds = embeddings['elmo_representations'][1]  # B x max_l x 1024
        sent_emb_list = list()
        for si in range(len(sentences)):
            sent_len = len(sentences[si])
            sent_embed = torch.mean(sent_embeds[si, :sent_len, :], dim=0)  # 1024
            sent_emb_list.append(sent_embed)
        sent_embs = torch.stack(sent_emb_list, dim=0)  # B x 1024
        return sent_embs


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.out_dim = 768
        self.eval()

    def forward(self, sentences, device='cuda'):
        """
        sentences: list[str], len of list: B
        output: embeddings
        """
        embeddings = list()
        for sentence in sentences:
            text = '[CLS] ' + sentence + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(indexed_tokens)
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            segments_tensors = torch.tensor([segments_ids]).to(device)
            last_hidden_states, _ = self.model(tokens_tensor, segments_tensors)
            sentence_embedding = torch.mean(last_hidden_states[0], dim=0)
            # del tokens_tensor, segments_tensors, last_hidden_states
            embeddings.append(sentence_embedding)
        embeddings = torch.stack(embeddings)
        return embeddings
