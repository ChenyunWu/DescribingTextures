import numpy as np
from tqdm import tqdm

from data_api.dataset_api import TextureDescriptionData, WordEncoder
from data_api.eval_gen_caption import compute_metrics

from models.naive_classifier.predictors import GenImgCaption as ClsCaption
from models.triplet_match.predictors import GenImgCaption as TriCaption
from models.show_attend_tell.eval import predict as sat_caption

from applications.synthetic_imgs.dataset import SyntheticData


def caption_pred():
    texture_dataset = TextureDescriptionData(phid_format=None)
    syn_dataset = SyntheticData()

    pred_cls = dict()
    pred_tri = dict()
    cls_captioner = ClsCaption(dataset=texture_dataset)
    tri_captioner = TriCaption(dataset=texture_dataset)
    for i in tqdm(range(len(syn_dataset)), desc='captioning top5'):
        img = syn_dataset.get_img(*syn_dataset.unravel_index(i))
        pred_cls[i] = [cls_captioner(img, top_k=5)]
        pred_tri[i] = [tri_captioner(img, top_k=5)]

    pred_sat_t = sat_caption(beam_size=5, img_dataset=syn_dataset, split=None)
    pred_sat = dict()
    for k, v in pred_sat_t.items():
        k = int(k.data)
        pred_sat[k] = [v[0]]
        if k == 0:
            print(k, pred_sat[k])
    pred_dicts = [pred_cls, pred_tri, pred_sat]
    np.save('applications/synthetic_imgs/visualizations/results/caption.npy', pred_dicts)
    print('pred captions saved.')
    return pred_dicts


def caption_eval(pred_dicts=None):
    syn_dataset = SyntheticData()
    if pred_dicts is None:
        pred_dicts = np.load('applications/synthetic_imgs/visualizations/results/caption.npy')

    def get_color_acc(pred_dict):
        correct = 0
        wrong = 0
        for i, caps in pred_dict.items():
            idxs = syn_dataset.unravel_index(i)
            c1 = syn_dataset.color_names[idxs[1]]
            c2 = syn_dataset.color_names[idxs[2]]
            for cap in caps:
                if c1 in cap and c2 in cap:
                    correct += 1
                else:
                    wrong += 1
        return correct / (correct + wrong)

    gt_caps = dict()
    for i in range(len(syn_dataset)):
        gt_caps[i] = [syn_dataset.get_desc(*syn_dataset.unravel_index(i))]
    img_gt_caps = list()
    for img_i in range(len(syn_dataset.img_names)):
        img_gt = {k: v for k, v in gt_caps.items() if k // len(syn_dataset.color_tuples) == img_i}
        img_gt_caps.append(img_gt)

    all_metrics = list()
    for i, pred_dict in enumerate(pred_dicts):
        if i == 0:
            print('************ caption cls top-5 ************')
        elif i == 1:
            print('************ caption tri top-5 ************')
        elif i == 2:
            print('************ caption show-attend-tell ************')

        all_metrics.append(list())
        m = compute_metrics(gt_caps, pred_dict)
        m['color_acc'] = get_color_acc(pred_dict)
        print('color_acc: ', m['color_acc'])
        all_metrics[i].append(m)
        for img_i in range(len(syn_dataset.img_names)):
            img_pred = {k: v for k, v in pred_dict.items() if k // len(syn_dataset.color_tuples) == img_i}
            print('*** ' + syn_dataset.img_names[img_i])
            m = compute_metrics(img_gt_caps[img_i], img_pred)
            m['color_acc'] = get_color_acc(img_pred)
            print('color_acc: ', m['color_acc'])
            all_metrics[i].append(m)
    np.save('applications/synthetic_imgs/visualizations/results/caption_metrics.npy', all_metrics)
    return all_metrics


def caption_visualize(pred_dicts=None, filter=False):
    if pred_dicts is None:
        pred_dicts = np.load('applications/synthetic_imgs/visualizations/results/caption.npy')
    syn_dataset = SyntheticData()
    word_encoder = WordEncoder()
    img_pref = '../modified_imgs/'
    html_str = '''<!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Caption visualize</title>
        <style>
        .correct {
            font-weight: bold;
        }
        .pred_name {
            color: ROYALBLUE;
            font-weight: bold;
        }
        img {
           width: 3cm
        }
        table {
            border-collapse: collapse;
        }
        tr {
            border-bottom: 1px solid lightgray;
        }

        </style>
    </head>
    <body>
    <table>
        <col class="column-one">
        <col class="column-two">
        <tr>
            <th style="text-align: center">Image</th>
            <th>Predicted captions</th>
        </tr>
    '''
    for idx in range(len(syn_dataset)):
        pred_caps = [pred_dict[idx][0] for pred_dict in pred_dicts]
        img_i, c1_i, c2_i = syn_dataset.unravel_index(idx)

        if filter:
            good_cap = False
            c1 = syn_dataset.color_names[c1_i]
            c2 = syn_dataset.color_names[c2_i]
            for ci, cap in enumerate(pred_caps):
                if c1 in cap and c2 in cap:
                    good_cap = True
                    break
            if not good_cap:
                continue

        img_name = '%s_%s_%s.jpg' % (syn_dataset.img_names[img_i].split('.')[0],
                                     syn_dataset.color_names[c1_i],
                                     syn_dataset.color_names[c2_i])
        gt_desc = syn_dataset.get_desc(img_i, c1_i, c2_i)

        for ci, cap in enumerate(pred_caps):
            tokens = WordEncoder.tokenize(cap)
            for ti, t in enumerate(tokens):
                if t in gt_desc and len(t) > 1:
                    tokens[ti] = '<span class="correct">%s</span>' % t
            pred_caps[ci] = word_encoder.detokenize(tokens)
        html_str += '''
    <tr>
        <td>
            <img src={img_pref}{img_name} alt="{img_name}">
        </td>
        <td>
            <span class="pred_name">Synthetic Ground-truth Description:</span><br>
            {gt}<br>
            <span class="pred_name">Classifier top 5:</span><br>
            {pred0}<br>
            <span class="pred_name">Triplet top 5:</span><br>
            {pred1}<br>
            <span class="pred_name">Show-attend-tell:</span><br>
            {pred2}<br>
        </td>
    </tr>
    '''.format(img_pref=img_pref, img_name=img_name, pred0=pred_caps[0], pred1=pred_caps[1], pred2=pred_caps[2],
               gt=gt_desc)
    html_str += '</table>\n</body\n></html>'

    html_name = 'caption.html'
    if filter:
        html_name = 'caption_filtered.html'
    with open('applications/synthetic_imgs/visualizations/results/' + html_name, 'w') as f:
        f.write(html_str)
    return


if __name__ == '__main__':
    # pred_dicts = caption_pred()
    pred_dicts = np.load('applications/synthetic_imgs/visualizations/results/caption.npy', allow_pickle=True)
    # caption_visualize(pred_dicts, filter=True)
    caption_eval(pred_dicts)
