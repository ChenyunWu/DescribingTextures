import os
import json
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from data_api.dataset_api import TextureDescriptionData, WordEncoder

plt.switch_backend('agg')

img_pref = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/'
img_html_format = '<figure>\n<figcaption>[{i}] {v:.4f}<br>{cls}</figcaption>\n' + \
             '<img src="{url}" alt="{name}">\n</figure>\n'
html_head = \
'''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Subset Analysis</title>
<!--    <link rel="stylesheet" href="retrieve_img_style.css">-->
    <style>
        figure {
            display: inline-block;
            margin: 0;
        }
        img {
            width: 1in;
            margin: 0;
        }
        figcaption {
            font-size: 10pt;
        }
        .plot {
            width: 8in;
        }
        table {
            width: 100%;
        }

        th {
            text-align: left;
        }
    </style>
</head>
<body>\n'''


class SubsetAnalyzer:
    def __init__(self, metric_name=''):
        self.ph_freq_dict = None
        self.metric_name = metric_name
        self.img_values = dict()  # key: img_name, value: list of evaluated values on this image
        self.phrase_values = dict()
        self.desc_values = dict()
        return

    def update(self, value, img_names, phrases=None, desc=None):
        for img_name in img_names:
            if img_name not in self.img_values.keys():
                self.img_values[img_name] = list()
            self.img_values[img_name].append(value)

        if desc is not None and phrases is None:
            phrases = TextureDescriptionData.description_to_phrases(desc)
        for ph in phrases:
            if ph not in self.phrase_values.keys():
                self.phrase_values[ph] = list()
            self.phrase_values[ph].append(value)

        if desc is not None:
            if desc not in self.desc_values.keys():
                self.desc_values[desc] = list()
            self.desc_values[desc].append(value)

    def draw_img_class_histogram(self, out_path):
        img_class_dict = dict()  # cls_name: [value_count, value_sum]
        for img_name, values in self.img_values.items():
            cls = img_name.split('/')[0]
            if cls not in img_class_dict:
                img_class_dict[cls] = [0, 0]
            img_class_dict[cls][0] += len(values)
            img_class_dict[cls][1] += np.sum(values)
        for cls, (v_count, v_sum) in img_class_dict.items():
            img_class_dict[cls] = v_sum / v_count

        with open(os.path.join(out_path, 'img_cls_histogram.json'), 'w') as f:
            json.dump(img_class_dict, f)

        cls_values = sorted(img_class_dict.items(), key=lambda kv: kv[1], reverse=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(x=range(len(cls_values)), tick_label=[cv[0] for cv in cls_values], height=[cv[1] for cv in cls_values])
        plt.xticks(rotation=30, ha='right', fontsize=8)
        fig.tight_layout()
        img_path = os.path.join(out_path, 'img_cls_histogram.jpg')
        print(img_path)
        plt.savefig(img_path, dpi=250)
        plt.close(fig)
        return

    def draw_ph_freq(self, out_path):
        if self.ph_freq_dict is None:
            dataset = TextureDescriptionData(phrase_freq_thresh=2)
            self.ph_freq_dict = {ph: freq for ph, freq in zip(dataset.phrases, dataset.phrase_freq)}
        x = list()
        y = list()
        for ph, vals in self.phrase_values.items():
            x.append(self.ph_freq_dict.get(ph, 0))
            y.append(np.mean(vals))

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=2, marker='.', alpha=0.2)
        ax.set_xlim(5, 200)
        fig.tight_layout()
        img_path = os.path.join(out_path, 'ph_freq.jpg')
        print(img_path)
        plt.savefig(img_path, dpi=250)
        plt.close(fig)
        return

    def draw_ph_len(self, out_path):
        ph_len_dict = dict()  # len: [count_v, sum_v]
        for ph, vals in self.phrase_values.items():
            ph_l = len(WordEncoder.tokenize(ph))
            if ph_l not in ph_len_dict:
                ph_len_dict[ph_l] = [0, 0]
            ph_len_dict[ph_l][0] += len(vals)
            ph_len_dict[ph_l][1] += np.sum(vals)
        for ph, (v_count, v_sum) in ph_len_dict.items():
            ph_len_dict[ph] = v_sum / v_count
        len_values = sorted(ph_len_dict.items(), key=lambda kv: kv[0], reverse=False)

        fig, ax = plt.subplots()
        ax.bar(x=[cv[0] for cv in len_values], height=[cv[1] for cv in len_values])
        fig.tight_layout()
        img_path = os.path.join(out_path, 'ph_len.jpg')
        print(img_path)
        plt.savefig(img_path, dpi=250)
        plt.close(fig)
        return

    def draw_ph_cloud(self, out_path, wc=None, cm_range=None):
        from wordcloud import WordCloud
        if self.ph_freq_dict is None:
            dataset = TextureDescriptionData(phrase_freq_thresh=2)
            self.ph_freq_dict = {ph: freq for ph, freq in zip(dataset.phrases, dataset.phrase_freq)}
        ph_freq_dict = {ph: np.sqrt(freq) for ph, freq in self.ph_freq_dict.items() if ph in self.phrase_values.keys()}

        vals = [np.mean(vs) for vs in self.phrase_values.values()]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        if cm_range is None:
            min_v = mean_v - 1 * std_v
            max_v = mean_v + 1 * std_v
        else:
            min_v, max_v = cm_range
        print('cm range', min_v, max_v)
        print('mean, std, min, max', mean_v, std_v, np.min(vals), np.max(vals))

        # min_v = max(mean_v - 1 * std_v, min(vals))
        # max_v = min(mean_v + 1 * std_v, max(vals))

        def color_fn(phrase, *args, **kwargs):
            cmap = cm.get_cmap('coolwarm_r')
            v = np.mean(self.phrase_values[phrase])
            v = (v - min_v) / (max_v - min_v)
            rgba = cmap(v, bytes=True)
            return rgba

        if wc is None:
            wc = WordCloud(background_color="white", color_func=color_fn, prefer_horizontal=0.9,
                           height=1200, width=2400, min_font_size=5, margin=4, max_words=500,
                           font_path='visualizations/DIN Alternate Bold.ttf', relative_scaling=1)
            wc.generate_from_frequencies(ph_freq_dict)
        else:
            wc.recolor(color_func=color_fn)
        wc_path = os.path.join(out_path, 'ph_cloud.jpg')
        print(wc_path)
        wc.to_file(wc_path)

        return wc

    def report(self, out_path, wc=None, cm_range=(0, 1)):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        self.draw_img_class_histogram(out_path)
        self.draw_ph_freq(out_path)
        self.draw_ph_len(out_path)
        wc = self.draw_ph_cloud(out_path, wc=wc, cm_range=cm_range)

        html_str = html_head + '<h1>Analysis on %s</h1>\n\n' % self.metric_name

        # image
        html_str += '<h2>Performance over image categories </h2>\n'
        html_str += '<img class="plot" src="img_cls_histogram.jpg" alt="img_cls_histogram.jpg"><hr>\n\n'

        img_mean_values = [(img_name, np.mean(values)) for img_name, values in self.img_values.items()]
        img_mean_values.sort(key=lambda kv: kv[1], reverse=True)
        html_str += '<h2>50 worst images </h2>\n'
        for i in range(1, 51):
            img_name = img_mean_values[-i][0]
            img_val = img_mean_values[-i][1]
            html_str += img_html_format.format(i=-i, v=img_val, url=img_pref + img_name,
                                               cls=img_name.split('/')[0], name=img_name)

        html_str += '<h2>50 best images </h2>\n'
        for i in range(50):
            img_name = img_mean_values[i][0]
            img_val = img_mean_values[i][1]
            html_str += img_html_format.format(i=i, v=img_val, url=img_pref + img_name,
                                               cls=img_name.split('/')[0], name=img_name)

        html_str += '<hr>\n\n<h2>Sorted images (sampled)</h2>\n'
        for i in range(len(img_mean_values))[::len(img_mean_values)//50]:
            img_name = img_mean_values[i][0]
            img_val = img_mean_values[i][1]
            html_str += img_html_format.format(i=i, v=img_val, url=img_pref + img_name,
                                               cls=img_name.split('/')[0], name=img_name)

        # phrase
        html_str += '<hr><hr><h2>Phrase performance by frequency</h2>\n'
        html_str += '<img class="plot" src="ph_freq.jpg" alt="ph_freq.jpg"><hr>\n\n'
        html_str += '<h2>Phrase performance by len</h2>\n'
        html_str += '<img class="plot" src="ph_len.jpg" alt="ph_len.jpg"><hr>\n\n'
        html_str += '<h2>Phrase performance cloud</h2>\n'
        html_str += '<img class="plot" src="ph_cloud.jpg" alt="ph_cloud.jpg"><hr>\n\n'

        ph_mean_values = [(ph, np.mean(values)) for ph, values in self.phrase_values.items()]
        ph_mean_values.sort(key=lambda kv: kv[1], reverse=True)
        html_str += '''
        <table>
            <tr>
                <th>Worst 50 phrases</th>
                <th>Best 50 phrases</th>
                <th>sorted phrases (sampled)</th>
            </tr>
            <tr>
                <td>
        '''
        for i in range(1, 51):
            ph = ph_mean_values[-i][0]
            ph_val = ph_mean_values[-i][1]
            html_str += '[{i}] {ph}: {v:.4f}<br>\n'.format(i=-i, ph=ph, v=ph_val)

        html_str += '</td>\n<td>\n'
        for i in range(50):
            ph = ph_mean_values[i][0]
            ph_val = ph_mean_values[i][1]
            html_str += '[{i}] {ph}: {v:.4f}<br>\n'.format(i=i, ph=ph, v=ph_val)

        html_str += '</td>\n<td>\n'
        for i in range(len(ph_mean_values))[::len(ph_mean_values) // 50]:
            ph = ph_mean_values[i][0]
            ph_val = ph_mean_values[i][1]
            html_str += '[{i}] {ph}: {v:.4f}<br>\n'.format(i=i, ph=ph, v=ph_val)
        html_str += '</td>\n</tr>\n</table>\n<hr>\n'

        # desc
        if len(self.desc_values) > 0:
            desc_mean_values = [(desc, np.mean(values)) for desc, values in self.desc_values.items()]
            desc_mean_values.sort(key=lambda kv: kv[1], reverse=True)
            html_str += '''
            <table>
            <tr>
                <th>Worst 50 descriptions</th>
                <th>Best 50 descriptions</th>
                <th>sorted descriptions (sampled)</th>
            </tr>
            <tr>
                <td>
                '''
            for i in range(1, 51):
                desc = desc_mean_values[-i][0]
                desc_val = desc_mean_values[-i][1]
                html_str += '[{i}]{v:.4f}: {desc}<br>\n'.format(i=-i, desc=desc, v=desc_val)

            html_str += '</td>\n<td>\n'
            for i in range(50):
                desc = desc_mean_values[i][0]
                desc_val = desc_mean_values[i][1]
                html_str += '[{i}]{v:.4f}: {desc}<br>\n'.format(i=i, desc=desc, v=desc_val)

            html_str += '</td>\n<td>\n'
            for i in range(len(desc_mean_values))[::len(desc_mean_values) // 50]:
                desc = desc_mean_values[i][0]
                desc_val = desc_mean_values[i][1]
                html_str += '[{i}]{v:.4f}: {desc}<br>\n'.format(i=i, desc=desc, v=desc_val)
            html_str += '</td>\n</tr>\n</table>\n<hr>\n'

        html_str += '</body>\n</html>'
        with open(os.path.join(out_path, 'subset_analyze.html'), 'w') as f:
            f.write(html_str)
        return wc
