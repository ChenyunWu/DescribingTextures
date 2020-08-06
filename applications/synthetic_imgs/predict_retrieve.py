import numpy as np
from tqdm import tqdm

from applications.synthetic_imgs.dataset import SyntheticData
from data_api.dataset_api import TextureDescriptionData
from models.naive_classifier.predictors import RetrieveImgFromDesc as ClsRetrieveImgFromDesc
from models.triplet_match.predictors import RetrieveImgFromDesc as TriRetrieveImgFromDesc


def retrieve_img_eval(pred_fn, input_cases):
    sorted_ids = list()
    acc_all = list()
    acc_hard = list()
    for input_case in tqdm(input_cases, desc='retrieval with input'):
        desc, gt_ids, hard_neg_ids = input_case
        pred_scores = pred_fn(desc)  # img_num
        case_sorted_ids = np.argsort(pred_scores * -1.0)
        sorted_ids.append(case_sorted_ids)

        pred_all = list(case_sorted_ids[:len(gt_ids)])
        case_acc_all = len(set(pred_all).intersection(gt_ids)) / len(gt_ids)
        acc_all.append(case_acc_all)

        hard = [i for i in case_sorted_ids if i in gt_ids or i in hard_neg_ids]
        pred_hard = list(hard[:len(gt_ids)])
        case_acc_hard = len(set(pred_hard).intersection(gt_ids)) / len(gt_ids)
        acc_hard.append(case_acc_hard)
    return sorted_ids, acc_all, acc_hard


def retrieve_img(input_cases, exp_name='fore_color'):
    syn_dataset =SyntheticData()
    syn_imgs = syn_dataset.get_all_imgs()
    texture_dataset = TextureDescriptionData()

    cls_retriever = ClsRetrieveImgFromDesc(dataset=texture_dataset)
    cls_img_ph_scores = cls_retriever.get_img_ph_scores(imgs=syn_imgs)
    cls_fn = lambda desc: cls_retriever(desc, img_ph_scores=cls_img_ph_scores)
    cls_results = retrieve_img_eval(cls_fn, input_cases)
    del cls_retriever, cls_img_ph_scores, cls_fn
    print('classifier_retrieve done. acc_all: %.4f; acc_hard: %.4f'
          % (float(np.mean(cls_results[1])), float(np.mean(cls_results[2]))))

    tri_retriever = TriRetrieveImgFromDesc(dataset=texture_dataset, split_to_phrases=True)
    tri_img_vecs = tri_retriever.get_img_vecs(imgs=syn_imgs)
    tri_fn = lambda desc: tri_retriever(desc, img_vecs=tri_img_vecs)
    tri_results = retrieve_img_eval(tri_fn, input_cases)
    del tri_retriever, tri_img_vecs, tri_fn
    print('triplet_retrieve done. acc_all: %.4f; acc_hard: %.4f'
          % (float(np.mean(tri_results[1])), float(np.mean(tri_results[2]))))

    tri_desc_retriever = TriRetrieveImgFromDesc(dataset=texture_dataset, split_to_phrases=False,
                                                trained_path='output/triplet_match/da3_bert_lr0.00001')
    tri_desc_img_vecs = tri_desc_retriever.get_img_vecs(imgs=syn_imgs)
    tri_desc_fn = lambda desc: tri_desc_retriever(desc, img_vecs=tri_desc_img_vecs)
    tri_desc_results = retrieve_img_eval(tri_desc_fn, input_cases)
    del tri_desc_retriever, tri_desc_img_vecs, tri_desc_fn
    print('triplet_desc_retrieve done. acc_all: %.4f; acc_hard: %.4f'
          % (float(np.mean(tri_desc_results[1])), float(np.mean(tri_desc_results[2]))))

    results = {'input_cases': input_cases,
               'cls_results': cls_results,
               'tri_results': tri_results,
               'tri_desc_results': tri_desc_results}
    np.save('applications/synthetic_imgs/visualizations/results/retrieve_img_%s.npy' % exp_name, results,
            allow_pickle=True)

    return results


def get_std():
    for exp_name in ['fore_color', 'back_color', 'color_pattern', 'two_colors']:
        results = np.load('applications/synthetic_imgs/visualizations/results/retrieve_img_%s.npy' % exp_name,
                          allow_pickle=True).item()
        for model_name in ['cls_results', 'tri_results', 'tri_desc_results']:
            accs = results[model_name][2]
            std = np.std(accs)
            print(exp_name, model_name, 'std', std)


def retrieve_img_visualize(results=None, exp_name='fore_color', dataset=None, hard_only=False):
    if dataset is None:
        dataset = SyntheticData()
    if results is None:
        results = np.load('applications/synthetic_imgs/visualizations/results/retrieve_img_%s.npy' % exp_name,
                          allow_pickle=True).item()

    img_pref = '../modified_imgs/'

    html_str = '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Image Retrieval Comparison</title>
    <style>
        .desc {
            font-size: large;
            font-weight: bold;
        }
        .model {
            font-weight: bold;
        }
        img {
            width: 0.6in;
            margin: 0px 0px 5px;
            border:2px solid white;
        }
        .correct{
            border:2px dashed DodgerBlue;
        }
        .wrong{
            border:2px dotted Tomato;
        }
        
    </style>
</head>
<body>
<h1>Image Retrieval Comparison</h1>
<span class="model">Classifier vs. Triplet(phrase) vs. Triplet(description)</span><br>
<hr>
'''
    if hard_only:
        exp_name += '_hard'
    html_path = 'applications/synthetic_imgs/visualizations/results/retrieve_img_%s.html' % exp_name

    for input_i, input_case in tqdm(enumerate(results['input_cases']), desc='generating html'):
        desc = input_case[0]
        html_str += '<span class="desc">%s</span><br>\n' % desc
        for model_name in ['cls_results', 'tri_results', 'tri_desc_results']:
            sorted_ids, acc_all, acc_hard = results[model_name]
            # html_str += '<span class="model">%s: acc_all %.4f, acc_hard %.4f</span><br>\n' \
            #             % (model_name, acc_all[input_i], acc_hard[input_i])
            to_show = sorted_ids[input_i][:10]
            if hard_only:
                can = [i for i in sorted_ids[input_i] if i in input_case[1] or i in input_case[2]]
                to_show = can[:10]
            for img_idx in to_show:
                img_name = dataset.get_img_name(*dataset.unravel_index(img_idx))
                style = ''
                if img_idx in input_case[1]:
                    style = 'class="correct"'
                elif img_idx in input_case[2]:
                    style = 'class="wrong"'
                html_str += '<img {cs} src="{img_pref}{img_name}" alt="{img_name}">\n'\
                    .format(img_pref=img_pref, img_name=img_name, cs=style)
            html_str += '<br>\n'
        html_str += '<hr>\n'

    html_str += '</body>\n</html>'
    with open(html_path, 'w') as f:
        f.write(html_str)
    return


def make_input_cases(dataset=None, exp_name='fore_color'):
    if dataset is None:
        dataset = SyntheticData()
    input_cases = list()
    if exp_name == 'fore_color':
        for pi, pattern in enumerate(dataset.patterns):
            if not dataset.is_fore_back[pi]:
                continue
            for ci, color in enumerate(dataset.color_names):
                desc = color + ' ' + pattern
                gt_ids = [dataset.ravel_index(pi, ci, x) for x in range(len(dataset.colors)) if x != ci]
                hard_ids = [dataset.ravel_index(pi, x, ci) for x in range(len(dataset.colors)) if x != ci]
                input_cases.append([desc, gt_ids, hard_ids])
        return input_cases

    elif exp_name == 'back_color':
        for ci, color in enumerate(dataset.color_names):
            desc = color + ' background'
            gt_ids = list()
            hard_ids = list()
            for pi, pattern in enumerate(dataset.patterns):
                if not dataset.is_fore_back[pi]:
                    continue
                gt_ids += [dataset.ravel_index(pi, x, ci) for x in range(len(dataset.colors)) if x != ci]
                hard_ids += [dataset.ravel_index(pi, ci, x) for x in range(len(dataset.colors)) if x != ci]
            input_cases.append([desc, gt_ids, hard_ids])
        return input_cases

    elif exp_name == 'color_pattern':
        for ci, color in enumerate(dataset.color_names):
            for pi, pattern in enumerate(dataset.patterns):
                desc = color + ' ' + pattern
                if dataset.is_fore_back[pi]:
                    gt_ids = [dataset.ravel_index(pi, ci, x) for x in range(len(dataset.colors)) if x != ci]  # P C _
                    hard_ids = [dataset.ravel_index(y, x, ci) for x in range(len(dataset.colors)) if x != ci
                                for y in range(len(dataset.patterns)) if not dataset.is_similar_pattern(y, pi)]  # _C_
                    hard_ids += [dataset.ravel_index(pi, x1, x2) for x1 in range(len(dataset.colors)) if x1 != ci
                                 for x2 in range(len(dataset.colors)) if x2 != x1]  # P _ _
                else:
                    gt_ids = [dataset.ravel_index(pi, ci, x) for x in range(len(dataset.colors)) if x != ci]  # P C _
                    gt_ids += [dataset.ravel_index(pi, x, ci) for x in range(len(dataset.colors)) if x != ci]  # P _ C
                    hard_ids = [dataset.ravel_index(y, x, ci) for x in range(len(dataset.colors)) if x != ci
                                for y in range(len(dataset.patterns)) if not dataset.is_similar_pattern(y, pi)]  # _C_
                    hard_ids += [dataset.ravel_index(y, x, ci) for x in range(len(dataset.colors)) if x != ci
                                 for y in range(len(dataset.patterns)) if not dataset.is_similar_pattern(y, pi)]  # _ _C
                    hard_ids += [dataset.ravel_index(pi, x1, x2) for x1 in range(len(dataset.colors)) if x1 != ci
                                 for x2 in range(len(dataset.colors)) if x2 not in (x1, ci)]  # P _ _
                input_cases.append([desc, gt_ids, hard_ids])
        return input_cases
    elif exp_name == 'two_colors':
        for c1, color1 in enumerate(dataset.color_names):
            for c2, color2 in enumerate(dataset.color_names):
                if c1 == c2:
                    continue
                desc = '%s and %s' % (color1, color2)
                gt_ids = [dataset.ravel_index(y, c1, c2) for y in range(len(dataset.patterns))]  # _ C1C2
                gt_ids += [dataset.ravel_index(y, c2, c1) for y in range(len(dataset.patterns))]  # _ C2C1
                hard_ids = [dataset.ravel_index(y, x1, x2) for y in range(len(dataset.patterns))  # _ X1X2 (one correct)
                            for (x1, x2) in dataset.color_tuples if (c1 in (x1, x2)) != (c2 in (x1, x2))]
                input_cases.append([desc, gt_ids, hard_ids])
        return input_cases
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # dataset = SyntheticData()
    # for exp_name in ['fore_color', 'back_color', 'color_pattern', 'two_colors']:
    #     input_cases = make_input_cases(dataset, exp_name)
    #     print('%s: #input %d, #pos %d, pos_rate %.4f'
    #           % (exp_name, len(input_cases), np.mean([len(ic[1]) for ic in input_cases]),
    #              np.mean([len(ic[1]) / (len(ic[1]) + len(ic[2])) for ic in input_cases])))
    #     # print(input_cases[0])
    #     # results = retrieve_img(input_cases, exp_name)
    #     results = None
    #     retrieve_img_visualize(results, exp_name, dataset)
    #     retrieve_img_visualize(results, exp_name, dataset, hard_only=True)
    get_std()
