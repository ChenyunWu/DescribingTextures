import numpy as np
from tqdm import tqdm

from data_api.dataset_api import TextureDescriptionData
from data_api.eval_retrieve import retrieve_with_desc_eval
from models.naive_classifier.predictors import RetrieveImgFromDesc as ClsRetrieveImgFromDesc
from models.triplet_match.predictors import RetrieveImgFromDesc as TriRetrieveImgFromDesc


def retrieve_img_desc_compare(split='test'):
    dataset = TextureDescriptionData()
    cls_retriever = ClsRetrieveImgFromDesc(dataset=dataset)
    tri_retriever = TriRetrieveImgFromDesc(dataset=dataset, split_to_phrases=True)
    tri_desc_retriever = TriRetrieveImgFromDesc(dataset=dataset, split_to_phrases=False,
                                                trained_path='output/triplet_match/da3_bert_lr0.00001')
    for fn in [tri_retriever, tri_desc_retriever, cls_retriever]:
        retrieve_with_desc_eval(pred_fn=fn, dataset=dataset, split=split)
    return


def retrieve_img_visualize(split='test', top_k=10, mode='desc', input_descs=None, visualize_count=100):
    dataset = TextureDescriptionData()
    cls_retriever = ClsRetrieveImgFromDesc(dataset=dataset)
    tri_retriever = TriRetrieveImgFromDesc(dataset=dataset, split_to_phrases=True)
    tri_desc_retriever = TriRetrieveImgFromDesc(dataset=dataset, split_to_phrases=False,
                                                trained_path='output/triplet_match/da3_bert_lr0.00001')
    pred_fns = [cls_retriever, tri_retriever, tri_desc_retriever]

    img_pref = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/'

    html_str = '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Image Retrieval Comparison</title>
    <link rel="stylesheet" href="retrieve_img_style.css">
</head>
<body>
<h1>Image Retrieval Comparison</h1>
<span class="model">Classifier vs. Triplet(phrase) vs. Triplet(description)</span><br>
<hr>
'''
    if input_descs is not None:
        html_path = 'visualizations/retrieve_img_predefined.html'
        gt_img_names = None
    elif mode is 'desc':
        html_path = 'visualizations/retrieve_img_desc.html'
        gt_img_names = dataset.img_splits[split]
        if len(gt_img_names) > visualize_count:
            gt_img_names = np.random.choice(gt_img_names, visualize_count, replace=False)
        input_descs = [np.random.choice(dataset.img_data_dict[img_name]['descriptions']) for img_name in gt_img_names]
    elif mode is 'phrase':
        html_path = 'visualizations/retrieve_img_phrase.html'
        gt_img_names = None
        input_descs = dataset.phrases
        if len(input_descs) > visualize_count:
            input_descs = np.random.choice(input_descs, visualize_count, replace=False)
    else:
        raise NotImplementedError

    for input_i in tqdm(range(len(input_descs)), desc='generating html'):
        desc = input_descs[input_i]
        html_str += '<span class="desc">%s</span><br>\n' % desc
        if gt_img_names is not None:
            img_name = gt_img_names[input_i]
            html_str += 'gt image: {img_name} <img src="{img_pref}{img_name}" alt="{img_name}"><br>\n'\
                .format(img_pref=img_pref, img_name=img_name)
        for pred_fn in pred_fns:
            img_scores = pred_fn(desc, split=split)
            sorted_img_idxs = np.argsort(img_scores * -1.0)
            top_k_idxs = sorted_img_idxs[:top_k]
            for img_idx in top_k_idxs:
                img_name = dataset.img_splits[split][img_idx]
                html_str += '<img src="{img_pref}{img_name}" alt="{img_name}">\n'\
                    .format(img_pref=img_pref, img_name=img_name)
            html_str += '<br>\n'
        html_str += '<hr>\n'

    html_str += '</body\n></html>'
    with open(html_path, 'w') as f:
        f.write(html_str)
    return


if __name__ == '__main__':
    # retrieve_img_desc_compare()
    # retrieve_img_visualize(mode='desc')
    # retrieve_img_visualize(mode='phrase')
    retrieve_img_visualize(input_descs=[
        'silver lines',
        'silver zigzags',
        'golden lines',
        'yellow and blue stripes',
        'yellow and blue lines',
        'yellow and blue vertical lines',
        'yellow and blue vertical stripes',
        'yellow shiny bubbles',
        'green uneven lines',
        'yellow and blue',
        'blue and yellow',
        'orange and black',
        'yellow and red',
        'red and gray',
        'pink and purple',
        'shiny lines',
        'shiny dots',
        'shiny bubbles',
        'pink dots',
        'yellow bubbles',
        'blue stripes',
        'uneven lines',
        'purple background',
        'silver background',
        'dark background'])
