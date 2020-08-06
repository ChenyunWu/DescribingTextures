import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from experiments.utils.caption_dataset import *
from experiments.utils.language_generation_utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys

# Parameters
data_folder = '/mnt/nfs/work1/smaji/mtimm/input_files_dtd'  # folder with data files saved by create_input_files.py
data_name = 'texture_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = './BEST_checkpoint_texture_lstm_no_encoder_finetune_dlr4e-4.pth.tar'  # model checkpoint
word_map_file = '/mnt/nfs/work1/smaji/mtimm/input_files_dtd/WORDMAP_texture_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint, map_location='cuda:0')
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# get scores for all phrases in data/phrase_freq.npy for all images.
# get captions for all test images as well
def evaluate_phrase_scores_and_get_captions(caption_beam_size=5, hijack_phrase_list=[]):
    """
    Evaluation on all frequent phrases.
    """
    # DataLoader
    print("Loading images...")
    loader = torch.utils.data.DataLoader(
        CaptionDatasetNoCaptions(data_folder='./data/', split='test25.txt',
                                 img_location='/mnt/nfs/work1/smaji/mtimm/dtd/dtd/images/',
                                 transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    all_phrases = np.load('data/phrase_freq.npy', allow_pickle=True).item()
    all_phrases_list = all_phrases.keys()
    if hijack_phrase_list:
        print("Evaluating scores for this list of phrases:")
        print(hijack_phrase_list)
        all_phrases_list = hijack_phrase_list

    image_to_phrase_to_score = {}

    # For each image
    for i, (image, filename) in enumerate(
            tqdm(loader, desc="EVALUATING SCORES FOR ALL PHRASES")):
        seqs, seq_scores = get_sequences(beam_size=caption_beam_size, image=image)
        image_to_phrase_to_score[filename[0]] = {}
        image_to_phrase_to_score[filename[0]]['caption_outputs_lstm'] = seqs
        image_to_phrase_to_score[filename[0]]['caption_scores_lstm'] = seq_scores
        #print(seqs)
        #print(seq_scores)
        #exit(1)
        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        #print("Evaluating phrases for image %s" % filename[0])

        for phrase in all_phrases_list:
            current_phrase_score = 0
            # We'll treat the problem as having a batch size of 1 (1 phrase at a time)
            encoder_out = encoder_out.expand(1, num_pixels, encoder_dim)  # (1, num_pixels, encoder_dim)

            # Tensor to store score at each step for each phrase; now they're just <start>
            prev_words = torch.LongTensor([[word_map['<start>']]]).to(device)  # (1, 1)
            h, c = decoder.init_hidden_state(encoder_out)
            current_phrase = phrase
            if "-" in phrase:
                current_phrase = phrase.replace("-", " ")
            #print("Getting score for phrase [%s]" % phrase)
            if len(current_phrase.split()) > 0:
                tokens = current_phrase.split()
                tokens.append("<end>")
            else:
                tokens = [current_phrase, "<end>"]
            index_prev_word = word_map['<start>']

            for word in tokens:
                index_of_word_we_want_score_for = word_map[word]
                embeddings = decoder.embedding(prev_words).squeeze(1)  # (s, embed_dim)

                awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                scores = decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                current_word_score = scores[0][index_of_word_we_want_score_for]
                current_phrase_score += current_word_score

                prev_word_ind = index_prev_word
                next_word_ind = index_of_word_we_want_score_for

                if next_word_ind != word_map['<end>']:
                    # need to keep going through tokens in the phrase
                    encoder_out = encoder_out[0].unsqueeze(0)
                    prev_words = torch.LongTensor([[next_word_ind]]).to(device)

            #print("Finished getting score for phrase %s." % phrase)
            image_to_phrase_to_score[filename[0]][phrase] = current_phrase_score.item()
        del image
        del encoder_out
        del prev_words
        del scores
        del embeddings
        torch.cuda.empty_cache()

    if hijack_phrase_list:
        np.save('eval_scores/test_lstm_scores_hijacked_phrases.npy', image_to_phrase_to_score)
    else:
        np.save("eval_scores/test_image_to_phrase_to_score_with_captions.npy", image_to_phrase_to_score)


def get_sequences(beam_size, image):
    hypotheses = list()
    k = beam_size

    # Move to GPU device, if available
    image = image.to(device)  # (1, 3, 256, 256)

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    # Hypotheses
    hypotheses_scores = list()
    for i, seq in enumerate(complete_seqs):
        hypotheses_scores.append(complete_seqs_scores[i])
        hypotheses.append([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
    sorted_lists = sorted(zip(hypotheses_scores, hypotheses), reverse=True)
    del encoder_out
    del scores
    del embeddings
    torch.cuda.empty_cache()
    hypotheses_sorted = [seq for _, seq in sorted_lists]
    hypotheses_scores_sorted = [score.item() for score, _ in sorted_lists]
    return hypotheses_sorted, hypotheses_scores_sorted


if __name__ == '__main__':
    evaluate_phrase_scores_and_get_captions(caption_beam_size=5, hijack_phrase_list=['pink dots', 'pink', 'white dots',
                                                                                     'white', 'dots'])
