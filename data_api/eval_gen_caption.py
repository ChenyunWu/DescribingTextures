import json
import numpy as np
import time

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge

from data_api.dataset_api import TextureDescriptionData, WordEncoder


def add_space_to_cap_dict(cap_dict):
    new_dict = dict()
    for img_name, caps in cap_dict.items():
        new_dict[img_name] = list()
        for cap in caps:
            tokens = WordEncoder.tokenize(cap)
            if len(tokens) > 0:
                new_cap = ' '.join(tokens)
            else:
                new_cap = cap
            new_dict[img_name].append(new_cap)
    return new_dict


def compute_metrics(gt_caps, pred_caps):
    assert len(gt_caps) == len(pred_caps)
    gt_caps = add_space_to_cap_dict(gt_caps)
    pred_caps = add_space_to_cap_dict(pred_caps)

    ret_scores = {}
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gt_caps, pred_caps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            print("%s: %0.6f" % (method, score))
            ret_scores[method] = score
        if isinstance(scorer, Meteor):
            scorer.close()
    del scorers
    return ret_scores


def eval_caption(split, dataset=None, pred_captions=None, pred_captions_fpath=None, html_path=None,
                 visualize_count=100):
    if pred_captions is None:
        with open(pred_captions_fpath, 'r') as f:
            pred_captions = json.load(f)
    assert type(pred_captions) == dict

    if dataset is None:
        dataset = TextureDescriptionData(phid_format=None)
    gt_captions = dict()
    for img_name in dataset.img_splits[split]:
        img_data = dataset.img_data_dict[img_name]
        gt_captions[img_name] = img_data['descriptions']
        # gt_captions[img_name] = list()
        # for desc in img_data['descriptions']:
        #     cap = ' '.join(WordEncoder.tokenize(desc))
        #     gt_captions[img_name].append(cap)

    pred_k_metrics_list = list()
    pred_per_img = len(list(pred_captions.values())[0])

    for pred_k in range(pred_per_img):
        print('Metrics on %d-th predicted caption:' % (pred_k + 1))
        tic = time.time()
        pred_caps_k = {img_name: [caps[pred_k]] for img_name, caps in pred_captions.items()}
        metrics_k = compute_metrics(gt_captions, pred_caps_k)
        pred_k_metrics_list.append(metrics_k)
        toc = time.time()
        print('time cost: %.1f s' % (toc - tic))

    pred_k_metrics_dict = dict()
    for metric in pred_k_metrics_list[0].keys():
        pred_k_metrics_dict[metric] = [metric_dict[metric] for metric_dict in pred_k_metrics_list]

    if html_path is not None:
        html_str = '<html><body>\n'
        html_str += '<h1>Captioning metrics</h1>\n'

        for pred_k in range(len(pred_k_metrics_list)):
            html_str += '<b>Metrics on %d-th predicted captions:</b><br>\n' % (pred_k + 1)
            for k, v in pred_k_metrics_list[pred_k].items():
                mean = np.mean(pred_k_metrics_dict[k][:pred_k + 1])
                html_str += '%s: %f (mean of top %d: %f)<br>\n' % (k, v, pred_k + 1, mean)

        img_names = dataset.img_splits[split]
        html_str += '<table>\n'
        for img_i, img_name in enumerate(img_names):
            html_str += '<tr style="border-bottom:1px solid black; border-collapse: collapse;">'
            html_str += '<td><img src=https://maxwell.cs.umass.edu/mtimm/images/%s width=300></td>\n' % img_name
            # pred caps
            pred_caps = pred_captions[img_name]
            pred_str = '<b>Predicted captions:</b><br><br>\n'
            for ci, cap in enumerate(pred_caps):
                pred_str += '({ci}) {cap}<br>\n'.format(ci=ci, cap=cap)
            html_str += '<td>' + pred_str + '</td>\n'
            # gt descriptions
            descriptions = dataset.img_data_dict[img_name]['descriptions']
            desc_str = '<b>Ground-truth descriptions:</b><br><br>\n'
            for di, desc in enumerate(descriptions):
                desc_str += '({di}) {desc}<br>\n'.format(di=di, desc=desc)
            html_str += '<td>' + desc_str + '</td>\n'
            html_str += '</tr>\n'

            if img_i >= visualize_count:
                break

        html_str += '</table></body></html>'
        with open(html_path, 'w') as f:
            f.write(html_str)

    return pred_k_metrics_list
