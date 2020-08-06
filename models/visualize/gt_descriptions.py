from tqdm import tqdm

from data_api.dataset_api import TextureDescriptionData


def gt_visualize(split='train', visualize_count=600, visualize_start=201):
    dataset = TextureDescriptionData()
    # cls_predictions = top_k_caption(top_k=5, model_type='cls', dataset=dataset, split=split)
    # with open('output/naive_classify/v1_35_ft2,4_fc512_tuneTrue/caption_top5_%s.json' % split, 'w') as f:
    #     json.dump(cls_predictions, f)
    # tri_predictions = top_k_caption(top_k=5, model_type='tri', dataset=dataset, split=split)
    # with open('output/triplet_match/c34_bert_l2_s_lr0.00001/caption_top5_%s.json' % split, 'w') as f:
    #     json.dump(tri_predictions, f)
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
    <tr>
        <th style="text-align: center">Image</th>
        <th>Ground-truth descriptions</th>
    </tr>
'''

    for img_i, img_name in enumerate(dataset.img_splits[split][visualize_start-1:]):
        gt_descs = dataset.img_data_dict[img_name]['descriptions']
        gt_html_str = ''
        for ci, cap in enumerate(gt_descs):
            gt_html_str += '[%d] %s<br>\n' % (ci + 1, cap)

        html_str += '''
        <tr>
            <td>
                <img src={img_pref}{img_name} alt="{img_name}">
            </td>
            <td >
                {gt}
            </td>
        </tr>
        '''.format(img_pref=img_pref, img_name=img_name, gt=gt_html_str)

        if img_i >= visualize_count:
            break

    html_str += '</table>\n</body\n></html>'

    html_path = 'visualizations/gt_descriptions_%s_%d_%d.html' \
                % (split, visualize_start, visualize_start + visualize_count - 1)
    with open(html_path, 'w') as f:
        f.write(html_str)
    return


if __name__ == '__main__':
    gt_visualize()
