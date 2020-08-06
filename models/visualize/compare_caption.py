import json
from tqdm import tqdm

from data_api.dataset_api import TextureDescriptionData, WordEncoder
from data_api.eval_gen_caption import compute_metrics
from models.naive_classifier.predictors import GenImgCaption as ClsCaption
from models.naive_classifier.predictors import load_model as cls_load_model
from models.triplet_match.predictors import GenImgCaption as TriCaption
from models.triplet_match.predictors import load_model as tri_load_model
# from models.show_attend_tell.eval import predict as sat_cap


def top_k_caption(top_k=5, model_type='cls', model=None, dataset=None, split='val'):
    if dataset is None:
        print('top_k_caption load dataset')
        dataset = TextureDescriptionData(phid_format=None)
    if model_type == 'cls':
        captioner = ClsCaption(dataset=dataset, model=model)
    elif model_type == 'tri':
        captioner = TriCaption(dataset=dataset, model=model)
    else:
        raise NotImplementedError

    predictions = dict()
    for img_name in tqdm(dataset.img_splits[split], desc='captioning %s top %d on %s' % (model_type, top_k, split)):
        img = dataset.load_img(img_name)
        caption = captioner(img, top_k)
        predictions[img_name] = [caption]
    return predictions


def compare_top_k(split, top_k=5):
    dataset = TextureDescriptionData()
    gt_captions = dict()
    for img_name in dataset.img_splits[split]:
        img_data = dataset.img_data_dict[img_name]
        gt_captions[img_name] = img_data['descriptions']
        # gt_captions[img_name] = list()
        # for desc in img_data['descriptions']:
        #     cap = ' '.join(WordEncoder.tokenize(desc))
        #     gt_captions[img_name].append(cap)

    for model_type in ('tri', 'cls'):
        if model_type is 'tri':
            model, _ = tri_load_model()
        else:
            model, _ = cls_load_model(dataset=dataset)

        # for top_k in range(1, 11):
        print('**** %s : top %d ****' % (model_type, top_k))
        predictions = top_k_caption(top_k, model_type=model_type, model=model, dataset=dataset, split=split)
        print(list(predictions.items())[0])
        compute_metrics(gt_captions, predictions)
    return


def compare_visualize(split='test', html_path='visualizations/caption.html', visualize_count=100):
    dataset = TextureDescriptionData()
    word_encoder = WordEncoder()
    # cls_predictions = top_k_caption(top_k=5, model_type='cls', dataset=dataset, split=split)
    # with open('output/naive_classify/v1_35_ft2,4_fc512_tuneTrue/caption_top5_%s.json' % split, 'w') as f:
    #     json.dump(cls_predictions, f)
    # tri_predictions = top_k_caption(top_k=5, model_type='tri', dataset=dataset, split=split)
    # with open('output/triplet_match/c34_bert_l2_s_lr0.00001/caption_top5_%s.json' % split, 'w') as f:
    #     json.dump(tri_predictions, f)
    cls_predictions = json.load(open('output/naive_classify/v1_35_ft2,4_fc512_tuneTrue/caption_top5_%s.json' % split))
    tri_predictions = json.load(open('output/triplet_match/c34_bert_l2_s_lr0.00001/caption_top5_%s.json' % split))
    sat_predictions = json.load(open('output/show_attend_tell/results/pred_v2_last_beam5_%s.json' % split))
    pred_dicts = [cls_predictions, tri_predictions, sat_predictions]
    img_pref = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/'

    html_str = '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Caption visualize</title>
    <link rel="stylesheet" href="caption_style.css">
</head>
<body>
<table>
    <col class="column-one">
    <col class="column-two">
    <col class="column-three">
    <tr>
        <th style="text-align: center">Image</th>
        <th>Predicted captions</th>
        <th>Ground-truth descriptions</th>
    </tr>
'''

    for img_i, img_name in enumerate(dataset.img_splits[split]):
        gt_descs = dataset.img_data_dict[img_name]['descriptions']
        gt_desc_str = '|'.join(gt_descs)
        gt_html_str = ''
        for ci, cap in enumerate(gt_descs):
            gt_html_str += '[%d] %s<br>\n' % (ci + 1, cap)

        pred_caps = [pred_dict[img_name][0] for pred_dict in pred_dicts]
        for ci, cap in enumerate(pred_caps):
            tokens = WordEncoder.tokenize(cap)
            for ti, t in enumerate(tokens):
                if t in gt_desc_str and len(t) > 1:
                    tokens[ti] = '<span class="correct">%s</span>' % t
            pred_caps[ci] = word_encoder.detokenize(tokens)
        html_str += '''
<tr>
    <td>
        <img src={img_pref}{img_name} alt="{img_name}">
    </td>
    <td>
        <span class="pred_name">Classifier top 5:</span><br>
        {pred0}<br>
        <span class="pred_name">Triplet top 5:</span><br>
        {pred1}<br>
        <span class="pred_name">Show-attend-tell:</span><br>
        {pred2}<br>
    </td>
    <td>
        {gt}
    </td>
</tr>
'''.format(img_pref=img_pref, img_name=img_name, pred0=pred_caps[0], pred1=pred_caps[1], pred2=pred_caps[2],
           gt=gt_html_str)

        if img_i >= visualize_count:
            break

    html_str += '</table>\n</body\n></html>'
    with open(html_path, 'w') as f:
        f.write(html_str)

    return


if __name__ == '__main__':
    # compare_top_k('test')
    compare_visualize()
