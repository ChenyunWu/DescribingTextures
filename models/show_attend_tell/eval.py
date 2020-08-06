import os
import json
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from data_api.dataset_api import WordEncoder, ImgOnlyDataset, TextureDescriptionData
from data_api.eval_gen_caption import eval_caption
from models.layers.img_encoder import build_transforms


# Parameters
split = 'test'
out_name_pref = 'pred_v2_last'
# checkpoint = 'output/show_attend_tell/checkpoints/BEST_checkpoint_v0_tuneResNetFalse.pth.tar'  # model checkpoint
# checkpoint = 'output/show_attend_tell/checkpoints/BEST_checkpoint_v1_tuneResNetTrue.pth.tar'  # model checkpoint
checkpoint = 'output/show_attend_tell/checkpoints/checkpoint_v2_FastText.pth.tar'  # model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
word_encoder = WordEncoder()


def predict(beam_size, img_dataset=None, split=split):
    # DataLoader
    if img_dataset is None:
        img_dataset = ImgOnlyDataset(split=split, transform=build_transforms(is_train=False))
    loader = torch.utils.data.DataLoader(img_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    predictions = dict()
    # For each image
    with torch.no_grad():
        for bi, (img_name, image) in enumerate(tqdm(loader, desc="PREDICTING AT BEAM SIZE " + str(beam_size))):
            img_name = img_name[0]
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
            k_prev_words = torch.as_tensor([[word_encoder.word_map['<start>']]] * k, dtype=torch.long).to(device) #(k,1)

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
                vocab_size = len(word_encoder.word_list)
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)
                # print("Top k scores")
                # print(top_k_scores)
                # print(top_k_words)

                # Add new words to sequences
                # print("Prev word inds")
                # print(prev_word_inds)
                # print("Next word idns")
                # print(next_word_inds)
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_encoder.word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds].cpu().numpy())
                # k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                # if k == 0:
                #     break
                if len(incomplete_inds) == 0:
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
                # print('seqs:', seqs)
                # print('complete lens:', [len(s) for s in complete_seqs])
                # print("Next step\n")

            # i = complete_seqs_scores.index(max(complete_seqs_scores))
            # seq = complete_seqs[i]
            if len(complete_seqs_scores) == 0:
                complete_seqs_scores = top_k_scores.squeeze().cpu().numpy()
                complete_seqs = seqs.cpu().tolist()
            sorted_idxs = np.argsort(np.asarray(complete_seqs_scores) * -1.0)
            if len(sorted_idxs) > beam_size:
                sorted_idxs = sorted_idxs[: beam_size]
            sorted_seqs = [complete_seqs[i] for i in sorted_idxs]

            if bi == 0:  # debug
                # best_i = complete_seqs_scores.index(max(complete_seqs_scores))
                # best_seq = complete_seqs[best_i]
                # print('best:', best_seq, complete_seqs_scores[best_i])
                print('top k:')
                for i, idx in enumerate(sorted_idxs):
                    ignore_wids = [word_encoder.word_map[w] for w in ['<start>', '<end>', '<pad>']]
                    seq = sorted_seqs[i]
                    tokens = [word_encoder.word_list[wid] for wid in seq if wid not in ignore_wids]
                    caption = word_encoder.detokenize(tokens)
                    print(caption, complete_seqs_scores[idx])

            predictions[img_name] = list()
            ignore_wids = [word_encoder.word_map[w] for w in ['<start>', '<end>', '<pad>']]
            for seq in sorted_seqs:
                tokens = [word_encoder.word_list[wid] for wid in seq if wid not in ignore_wids]
                # caption = ' '.join(tokens)
                caption = word_encoder.detokenize(tokens)
                predictions[img_name].append(caption)
            if len(predictions[img_name]) < beam_size:
                predictions[img_name] += [''] * (beam_size - len(predictions[img_name]))
    return predictions


if __name__ == '__main__':
    result_path = 'output/show_attend_tell/results'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    dataset = TextureDescriptionData(phid_format=None)
    for beam_size in [1, 5, 10, 15]:
        html_path = os.path.join(result_path, '%s_beam%d_%s.html' % (out_name_pref, beam_size, split))
        pred_path = os.path.join(result_path, '%s_beam%d_%s.json' % (out_name_pref, beam_size, split))

        if os.path.exists(pred_path):
            print('EVALUATING AT BEAM SIZE %d:' % beam_size)
            eval_caption(split=split, dataset=dataset, pred_captions_fpath=pred_path, html_path=html_path)
            continue

        predictions = predict(beam_size)
        with open(pred_path, 'w') as f:
            json.dump(predictions, f)
        print('EVALUATING AT BEAM SIZE %d:' % beam_size)
        eval_caption(split=split, dataset=dataset, pred_captions=predictions, html_path=html_path)
