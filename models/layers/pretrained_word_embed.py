import io
import numpy as np

word_emb_fpath = {'fast_text': 'cache/word_emb/wiki-news-300d-1M.vec'}


def fast_text_load_vectors():
    fname = word_emb_fpath['fast_text']
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data, d


def get_word_embed(word_list, embed_name='fast_text'):
    if embed_name == 'fast_text':
        whole_vocab, emb_dim = fast_text_load_vectors()
    else:
        raise NotImplementedError
    embed_matrix = np.random.randn(len(word_list), emb_dim)
    for wi, w in enumerate(word_list):
        if w in whole_vocab.keys():
            embed_matrix[wi] = np.asarray(whole_vocab[w])
    return embed_matrix
