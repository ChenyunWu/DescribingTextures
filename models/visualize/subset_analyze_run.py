import os
import json
import pickle
import numpy as np
from tqdm import tqdm

from data_api.dataset_api import TextureDescriptionData
from data_api.utils.retrieval_metrics import average_precision
from models.visualize.subset_analyzer import SubsetAnalyzer
from models.triplet_match.predictors import RetrieveImgFromDesc
from nlgeval.pycocoevalcap.meteor.meteor import Meteor

model_preds = {'cls': 'output/naive_classify/v1_35_ft2,4_fc512_tuneTrue/eval_visualize_test/pred_scores.npy',
               'tri': 'output/triplet_match/c34_bert_l2_s_lr0.00001/eval_visualize_test/pred_scores.npy'}


def analyze_phrase_retrieval(dataset=None, wc=None, models=('cls', 'tri'), cm_range=(0, 1)):
    metric_name = 'phrase_retrieval_ap'
    if dataset is None:
        dataset = TextureDescriptionData(phid_format='set')
    gt_matrix = dataset.get_img_phrase_match_matrices('test')
    for m in models:
        neg_distances = np.load(model_preds[m])
        # neg_distances = np.load('output/triplet_match/c34_bert_l2_s_lr0.00001/eval_visualize_test/pred_scores.npy')
        match_scores = neg_distances

        analyzer = SubsetAnalyzer(metric_name)

        for img_i, img_name in tqdm(enumerate(dataset.img_splits['test']), total=len(dataset.img_splits['test']),
                                    desc='analyzing %s with %s' % (metric_name, m)):
            img_data = dataset.img_data_dict[img_name]
            phid_set = img_data['phrase_ids']
            phrases = [dataset.phid_to_phrase(i) for i in phid_set]

            phrase_idx_sorted = np.argsort(-match_scores[img_i, :])
            i2p_correct = gt_matrix[img_i, phrase_idx_sorted]
            ap = average_precision(i2p_correct)
            analyzer.update(value=ap, img_names=[img_name], phrases=phrases)

        wc = analyzer.report('visualizations/subset/%s__%s' % (metric_name, m), wc=None, cm_range=cm_range)
    return wc


def analyze_image_retrieval(dataset=None, wc=None, models=('cls', 'tri'), cm_range=(0, 1)):
    metric_name = 'image_retrieval_ap'
    if dataset is None:
        dataset = TextureDescriptionData(phid_format='set')
    gt_matrix = dataset.get_img_phrase_match_matrices('test')  # img_num x phrase_num
    for m in models:
        pred_scores = np.load(model_preds[m])
        # pred_scores = np.load('output/naive_classify/v1_35_ft2,4_fc512_tuneTrue/eval_visualize_test/pred_scores.npy')
        match_scores = pred_scores

        analyzer = SubsetAnalyzer(metric_name)

        for ph_i, ph in tqdm(enumerate(dataset.phrases), total=len(dataset.phrases),
                             desc='analyzing %s with %s' % (metric_name, m)):
            gt_img_names = [dataset.img_splits['test'][i] for i in range(gt_matrix.shape[0]) if gt_matrix[i, ph_i]]

            img_idx_sorted = np.argsort(-match_scores[:, ph_i])
            p2i_correct = gt_matrix[img_idx_sorted, ph_i]
            ap = average_precision(p2i_correct)
            analyzer.update(value=ap, img_names=gt_img_names, phrases=[ph])

        wc = analyzer.report('visualizations/subset/%s__%s' % (metric_name, m), wc=None, cm_range=cm_range)
    return wc


def analyze_image_retrieval_desc(dataset=None, wc=None, cm_range=(0, 1)):
    metric_name = 'image_retrieval_desc_mrr'
    analyzer_cache_path = 'output/triplet_match/da3_bert_lr0.00001/subset_analyze_img_ret_desc.pkl'
    if os.path.exists(analyzer_cache_path):
    # if False:
        with open(analyzer_cache_path, 'rb') as f:
            analyzer = pickle.load(f)
    else:
        if dataset is None:
            dataset = TextureDescriptionData(phid_format=None)

        tri_desc_retriever = RetrieveImgFromDesc(dataset=dataset, split_to_phrases=False,
                                                 trained_path='output/triplet_match/da3_bert_lr0.00001')
        analyzer = SubsetAnalyzer(metric_name)

        for img_i, img_name in tqdm(enumerate(dataset.img_splits['test']), total=len(dataset.img_splits['test']),
                                    desc='analyzing %s over images' % metric_name):
            img_data = dataset.img_data_dict[img_name]
            for desc in img_data['descriptions']:
                pred_scores = tri_desc_retriever(desc, split='test')
                if np.all(pred_scores == 0):
                    v = 0
                else:
                    r = 1
                    for s in pred_scores:
                        if s > pred_scores[img_i]:
                            r += 1
                    v = 1.0 / r
                phrases = dataset.description_to_phrases(desc)
                analyzer.update(value=v, img_names=[img_name], phrases=phrases, desc=desc)
        with open('output/triplet_match/da3_bert_lr0.00001/subset_analyze_img_ret_desc.pkl', 'wb') as f:
            pickle.dump(analyzer, f)

    wc = analyzer.report('visualizations/subset/' + metric_name, wc=wc, cm_range=cm_range)
    return wc


def analyze_caption(dataset=None, wc=None, cm_range=(0, 1)):
    metric_name = 'caption_Meteor'
    if dataset is None:
        dataset = TextureDescriptionData(phid_format='set')
    with open('output/show_attend_tell/results/pred_v2_last_beam5_test.json', 'r') as f:
        pred_captions = json.load(f)
    analyzer = SubsetAnalyzer(metric_name)
    # scorer = Bleu(4)
    scorer = Meteor()

    for img_i, img_name in tqdm(enumerate(dataset.img_splits['test']), total=len(dataset.img_splits['test']),
                                desc='analyzing %s over images' % metric_name):
        img_data = dataset.img_data_dict[img_name]
        phids = img_data['phrase_ids']
        phrases = [dataset.phid_to_phrase(i) for i in phids]
        gt_captions = img_data['descriptions']
        pred_caption = pred_captions[img_name][0]
        score, _ = scorer.compute_score({0: gt_captions}, {0: [pred_caption]})
        # bleu4 = score[3]
        meteor = score
        analyzer.update(value=meteor, img_names=[img_name], phrases=phrases)

    wc = analyzer.report('visualizations/subset/' + metric_name, wc=wc, cm_range=cm_range)
    return wc


if __name__ == '__main__':
    dataset = TextureDescriptionData(phid_format='set')
    wc = None
    # analyze_phrase_retrieval(dataset, cm_range=(0.25, 0.35))
    # wc = analyze_phrase_retrieval(dataset)
    # analyze_image_retrieval(dataset, cm_range=(0, 0.5))
    analyze_image_retrieval_desc(dataset, cm_range=(0, 0.2))
    # analyze_caption(dataset, cm_range=(0.17, 0.24))
