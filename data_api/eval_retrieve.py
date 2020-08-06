import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_api.dataset_api import TextureDescriptionData
from data_api.utils.retrieval_metrics import mean_reciprocal_rank, r_precision, mean_precision_at_k, mean_recall_at_k
from data_api.utils.retrieval_metrics import mean_average_precision

plt.switch_backend('agg')

i2p_mode_names = ['i2p', 'img2phrase', 'img_to_phrase']
p2i_mode_names = ['p2i', 'phrase2img', 'phrase_to_img']


def retrieve_with_desc_eval(pred_fn, dataset=None, split='val'):
    """
    INPUT:
    pred_fn: prediction function, input (desc, split='test'), output scores over images
    dataset: instance of TextureDescriptionData
    split: default is 'val'. match_scores should cover all imgs in this split
    """

    if dataset is None:
        dataset = TextureDescriptionData(phid_format='set')

    rrs = list()
    for img_i, img_name in tqdm(enumerate(dataset.img_splits[split]), total=len(dataset.img_splits[split]),
                                desc='computing mrr in retrieve_with_desc'):
        img_data = dataset.img_data_dict[img_name]
        for desc in img_data['descriptions']:
            pred_scores = pred_fn(desc, split=split)
            if np.all(pred_scores == 0):
                rrs.append(0)
            else:
                r = 1
                for s in pred_scores:
                    if s > pred_scores[img_i]:
                        r += 1
                rrs.append(1.0 / r)
    mrr = np.mean(rrs)
    print('mean reciprocal rank on %s: %f' % (split, mrr))
    return mrr


def retrieve_eval(match_scores, dataset=None, split='val', mode='img2phrase',
                  visualize_path=None, max_visualize_num=100, add_to_summary_name=None, verbose=True):
    """
    INPUT:
    match_scores: [img_num x phrase_num], match_scores[i,j] shows how well img_i and phrase_j matches
    dataset: instance of TextureDescriptionData
    split: default is 'val'. match_scores should cover all imgs in this split
    visualize_path: if None, no visualization
    max_visualize_num: if <= 0, visualize all in this split
    """

    if dataset is None:
        dataset = TextureDescriptionData(phid_format='set')
    gt_matrix = dataset.get_img_phrase_match_matrices(split)
    img_num = gt_matrix.shape[0]
    phrase_num = gt_matrix.shape[1]

    if mode in i2p_mode_names:
        # each row is prediction for one image. phrase sorted by pred scores. values are whether the phrase is correct
        i2p_correct = np.zeros_like(gt_matrix, dtype=bool)  # img_num x phrase_num
        i2p_phrase_idxs = np.zeros_like(i2p_correct, dtype=int)
        for img_i in range(img_num):
            phrase_idx_sorted = np.argsort(-match_scores[img_i, :])
            i2p_phrase_idxs[img_i] = phrase_idx_sorted
            i2p_correct[img_i] = gt_matrix[img_i, phrase_idx_sorted]
        retrieve_binary_lists = i2p_correct
        retrieve_idxs = i2p_phrase_idxs
        # gt_count = dataset.get_gt_phrase_count(split)
    elif mode in p2i_mode_names:
        # each row is prediction for one prhase. images sorted by pred scores. values are whether the image is correct
        p2i_correct = np.zeros_like(gt_matrix, dtype=bool).transpose()  # class_num x img_num
        p2i_img_idxs = np.zeros_like(p2i_correct, dtype=int)
        for pi in range(phrase_num):
            img_idx_sorted = np.argsort(-match_scores[:, pi])
            p2i_img_idxs[pi] = img_idx_sorted
            p2i_correct[pi] = gt_matrix[img_idx_sorted, pi]
        retrieve_binary_lists = p2i_correct
        retrieve_idxs = p2i_img_idxs
        # gt_count = np.sum(gt_matrix, axis=0)
    else:
        raise NotImplementedError

    # calculate metrics
    metrics = dict()
    mean_reciprocal_rank_ = mean_reciprocal_rank(retrieve_binary_lists)
    r_precision_ = r_precision(retrieve_binary_lists)
    mean_average_precision_ = mean_average_precision(retrieve_binary_lists)
    metrics['mean_reciprocal_rank'] = mean_reciprocal_rank_
    metrics['r_precision'] = r_precision_
    metrics['mean_average_precision'] = mean_average_precision_

    for k in [5, 10, 20, 50, 100]:
        precision_at_k_ = mean_precision_at_k(retrieve_binary_lists, k)
        recall_at_k_ = mean_recall_at_k(retrieve_binary_lists, k, gt_count=None)
        metrics['precision_at_%03d' % k] = precision_at_k_
        metrics['recall_at_%03d' % k] = recall_at_k_

    # print metrics
    if verbose:
        print('## retrieve_eval {mode} on {split} ##'.format(mode=mode, split=split))
    for m, v in sorted(metrics.items(), key=lambda mv: mv[0]):
        print('%s: %.4f' % (m, v))

    # add to summary
    if mode in i2p_mode_names:
        log_to_summary(add_to_summary_name, i2p_metrics=metrics)
    elif mode in p2i_mode_names:
        log_to_summary(add_to_summary_name, p2i_metrics=metrics)

    # visualization
    if visualize_path is not None:
        if max_visualize_num <= 0 or max_visualize_num > len(retrieve_binary_lists):
            max_visualize_num = len(retrieve_binary_lists)
        if not os.path.exists(visualize_path):
            os.makedirs(visualize_path)

        precisions = list()
        recalls = list()
        for k in range(1, 101):
            precisions.append(mean_precision_at_k(retrieve_binary_lists, k))
            recalls.append(mean_recall_at_k(retrieve_binary_lists, k, gt_count=None))

        # plot pr curve and topk-recall curve
        plot_precision_recall_curves(mode, precisions, recalls, visualize_path)

        # generate html file
        generate_html(dataset, split, mode, visualize_path, max_visualize_num, img_num, phrase_num, gt_matrix,
                      match_scores, metrics, retrieve_idxs)

    return metrics


def log_to_summary(exp_name=None, p2i_metrics=None, i2p_metrics=None):
    if exp_name is None:
        return
    to_log = list()
    if p2i_metrics is not None:
        summary_path = 'output/summary/retrieve_phrase2img.csv'
        to_log.append((p2i_metrics, summary_path))
    if i2p_metrics is not None:
        summary_path = 'output/summary/retrieve_img2phrase.csv'
        to_log.append((i2p_metrics, summary_path))

    for metrics, summary_path in to_log:
        if not os.path.exists(os.path.dirname(summary_path)):
            os.makedirs(os.path.dirname(summary_path))
        with open(summary_path, 'a') as f:
            to_write = [exp_name]
            for m, v in sorted(metrics.items(), key=lambda mv: mv[0]):
                to_write.append(str(v))
            f.write(','.join(to_write) + '\n')


def plot_precision_recall_curves(mode, precisions, recalls, visualize_path):
    fig, axes = plt.subplots(1, 3, figsize=(8, 2))
    axes[0].plot(recalls, precisions, '-')
    axes[0].set_title('Precision-recall curve')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[1].plot(range(1, 101), recalls)
    axes[1].set_title('Top-k recall')
    axes[1].set_xlabel('Top-k')
    axes[1].set_ylabel('Recall')
    axes[2].plot(range(1, 101), precisions)
    axes[2].set_title('Top-k precision')
    axes[2].set_xlabel('Top-k')
    axes[2].set_ylabel('Precision')
    fig.tight_layout()
    plt.savefig(os.path.join(visualize_path, '%s_precision_recall.jpg' % mode), dpi=250)
    plt.close(fig)


def generate_html(dataset, split, mode, visualize_path, max_visualize_num, img_num, phrase_num, gt_matrix,
                  match_scores, metrics, retrieve_idxs):
    html_str = '<html><body>\n'
    html_str += '<h1>Retrieval {mode} on {split}</h1>\n'.format(mode=mode, split=split)
    html_str += '{img_num} images, {phrase_num} phrases<br><br>\n'.format(img_num=img_num, phrase_num=phrase_num)
    html_str += '<b>Metrices:</b><br>\n'
    for m, v in sorted(metrics.items(), key=lambda mv: mv[0]):
        html_str += '%s: %.4f<br>\n' % (m, v)
    html_str += '<img src="{mode}_precision_recall.jpg" width=90%>\n'.format(mode=mode)
    if mode in ['img2phrase', 'i2p']:
        html_str += '<table>\n'
        vis_img_idxs = np.random.choice(img_num, max_visualize_num, replace=False)
        for img_idx in vis_img_idxs:
            html_str += '<tr style="border-bottom:1px solid black">'
            img_name = dataset.img_splits[split][img_idx]
            html_str += '<td><img src=https://maxwell.cs.umass.edu/mtimm/images/{img_name} width=300></td>\n' \
                .format(img_name=img_name)

            # pred phrases
            gt_phrase_idxs = np.argwhere(gt_matrix[img_idx])
            pred_str = '<b>Predicted top 20 phrases:</b><br><br>\n'
            for i in range(10):
                phrase_idx = int(retrieve_idxs[img_idx, i])
                phrase = dataset.phrases[phrase_idx]
                score = match_scores[img_idx, phrase_idx]
                if phrase_idx in gt_phrase_idxs:
                    pred_str += '<b>%d (%.3f): %s</b><br>' % (i, score, phrase)
                else:
                    pred_str += '%d (%.3f): %s<br>' % (i, score, phrase)
            html_str += '<td>' + pred_str + '</td>\n'

            pred_str = '<br><br><br>\n'
            for i in range(10):
                phrase_idx = int(retrieve_idxs[img_idx, i + 10])
                phrase = dataset.phrases[phrase_idx]
                score = match_scores[img_idx, phrase_idx]
                if phrase_idx in gt_phrase_idxs:
                    pred_str += '<b>%d (%.3f): %s</b><br>' % (i + 10, score, phrase)
                else:
                    pred_str += '%d (%.3f): %s<br>' % (i + 10, score, phrase)
            html_str += '<td>' + pred_str + '</td>\n'

            # gt phrases
            gt_phrases = [dataset.phrases[int(idx)] for idx in gt_phrase_idxs]
            gt_str = '<b>Ground-truth phrases<br>(above frequency threshold)</b><br><br>\n'
            gt_str += '<br>'.join(gt_phrases)
            html_str += '<td>' + gt_str + '</td>\n'

            # gt descriptions
            descriptions = dataset.img_data_dict[img_name]['descriptions']
            desc_str = '<b>Ground-truth descriptions:</b><br><br>\n'
            desc_str += '<br><br>'.join(descriptions)
            html_str += '<td>' + desc_str + '</td>\n'
            html_str += '</tr>\n'

        html_str += '</table>\n</body></html>'

    elif mode in ['phrase2img', 'p2i']:
        vis_phrase_idxs = np.random.choice(phrase_num, max_visualize_num, replace=False)
        for phrase_idx in vis_phrase_idxs:
            phrase = dataset.phrases[phrase_idx]
            html_str += '<hr><h3>Input phrase: {phrase}</h3>\n'.format(phrase=phrase)

            # gt images
            html_str += '<b> Ground-truth images</b><br>\n'
            gt_img_idxs = np.argwhere(gt_matrix[:, phrase_idx]).flatten()
            for img_idx in gt_img_idxs:
                img_name = dataset.img_splits[split][img_idx]
                html_str += '<img src=https://maxwell.cs.umass.edu/mtimm/images/{} height=150>\n'.format(img_name)

            # plot pred images
            html_str += '<br><b> Retrieved top 20 images</b><br>\n'
            for i in range(20):
                img_idx = retrieve_idxs[phrase_idx, i]
                pred_score = match_scores[img_idx, phrase_idx]
                img_name = dataset.img_splits[split][img_idx]
                if img_idx in gt_img_idxs:
                    border = 5
                else:
                    border = 0
                html_str += '<figure style="display: inline-block; margin: 0">' \
                            '<img src=https://maxwell.cs.umass.edu/mtimm/images/{img_name} ' \
                            'height=150 border={border}>' \
                            '<figcaption>{idx}:{score:.3f}</figcaption></figure>\n' \
                    .format(img_name=img_name, border=border, idx=i + 1, score=pred_score)

        html_str += '</body></html>'

    else:
        raise NotImplementedError

    with open(os.path.join(visualize_path, '%s_eval.html' % mode), 'w') as f:
        f.write(html_str)

    return
