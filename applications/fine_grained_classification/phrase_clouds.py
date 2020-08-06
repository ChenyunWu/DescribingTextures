import numpy as np
import os
from wordcloud import WordCloud
import matplotlib.cm as cm
import matplotlib.pyplot as plt


from data_api.dataset_api import TextureDescriptionData
from applications.fine_grained_classification.cub_dataset import CUBDataset
from applications.fine_grained_classification.classify import classify

plt.switch_backend('agg')


def main(ph_num=100):
    cub_dataset = CUBDataset(split=None, val_ratio=0)
    classes = cub_dataset.class_names
    # model_name = 'LogisticRegression_newton-cg_multinomial_C1'
    # _, model = classify(cub_dataset=cub_dataset, feat_mode='ph_cls', model_name=model_name, norm=False,
    #                     val_ratio=0, ks=[ph_num])
    # weights = model.coef_  # classes x phrases
    # np.save('applications/fine_grained_classification/classify_ph_weights_%d.npy' % ph_num, weights)
    weights = np.load('applications/fine_grained_classification/classify_ph_weights_%d.npy' % ph_num)

    texture_dataset = TextureDescriptionData()
    phrases = texture_dataset.phrases

    # mean_v = np.mean(weights)
    std_v = np.std(weights)
    min_v = -1 * std_v
    max_v = std_v

    pos_std = weights[weights > 0]
    neg_std = weights[weights < 0]

    print('ready')
    folder = 'applications/fine_grained_classification/ph_clouds_%d_np' % ph_num
    if not os.path.exists(folder):
        os.makedirs(folder)
    # for cls_i, cls in enumerate(classes):
    for cls_i in [17, 55, 188]:
        cls = classes[cls_i]
        ph_weights = weights[cls_i, :]
        ph_freq_dict = {ph: abs(w) for ph, w in zip(phrases, ph_weights)}
        pos_dict = {ph: w for ph, w in zip(phrases, ph_weights) if w > 0}
        neg_dict = {ph: -w for ph, w in zip(phrases, ph_weights) if w < 0}

        def color_fn(phrase, *args, **kwargs):
            ph_i = texture_dataset.phrase_to_phid(phrase)
            # v = (ph_weights[ph_i] - min_v) / (max_v - min_v)
            w = ph_weights[ph_i]
            if w > 0:
                v = w / pos_std + 0.5
            else:
                v = w / neg_std + 0.5
            cmap = cm.get_cmap('coolwarm')
            rgba = cmap(v, bytes=True)
            return rgba
        red_fn = lambda *args, **kwargs: "red"
        blue_fn = lambda *args, **kwargs: "blue"
        # wc = WordCloud(background_color="white", color_func=color_fn, prefer_horizontal=0.9,
        #                height=600, width=1200, min_font_size=5, margin=4, max_words=500,
        #                font_path='visualizations/DIN Alternate Bold.ttf')
        # wc.generate_from_frequencies(ph_freq_dict)
        # wc_path = os.path.join(folder, '%s.jpg' % cls)
        # print(wc_path)
        # wc.to_file(wc_path)
        wc = WordCloud(background_color="white", color_func=red_fn, prefer_horizontal=0.9,
                       height=600, width=1200, min_font_size=4, margin=2, max_words=500,
                       font_path='visualizations/DIN Alternate Bold.ttf')
        wc.generate_from_frequencies(pos_dict)
        wc_path = os.path.join(folder, '%s_pos.jpg' % cls)
        print(wc_path)
        wc.to_file(wc_path)

        wc = WordCloud(background_color="white", color_func=blue_fn, prefer_horizontal=0.9,
                       height=600, width=1200, min_font_size=4, margin=2, max_words=500,
                       font_path='visualizations/DIN Alternate Bold.ttf')
        wc.generate_from_frequencies(neg_dict)
        wc_path = os.path.join(folder, '%s_neg.jpg' % cls)
        print(wc_path)
        wc.to_file(wc_path)


if __name__ == '__main__':
    main(100)
