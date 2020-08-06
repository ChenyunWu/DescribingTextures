import os
from PIL import Image


def gen_phrase_clouds():
    from models.triplet_match.predictors import GenImgWordCloud
    predictor = GenImgWordCloud()
    fig_path = 'applications/max_texture_phrase_cloud/figs/fgvc_figs'
    input_path = os.path.join(fig_path, 'texture_images')
    output_path = os.path.join(fig_path, 'phrase_clouds')
    for in_sub in os.listdir(input_path):
        in_subdir = os.path.join(input_path, in_sub)
        out_subdir = os.path.join(output_path, in_sub)
        if not os.path.exists(out_subdir):
            os.makedirs(out_subdir)
        for img_f in os.listdir(in_subdir):
            if img_f.endswith('.png'):
                print(os.path.join(out_subdir, img_f))
                img = Image.open(os.path.join(in_subdir, img_f)).convert('RGB')
                predictor(img, out_path=os.path.join(out_subdir, img_f))
    return


def generate_latex_table(dataset_class_list, col_num):
    latex_str = '\\begin{figure*}[t]\n\\begin{center}\n' + \
                '\\setlength{\\tabcolsep}{1.5pt}\n' + \
                '\\begin{tabular}{*{%d}{c}}\n' % col_num
    width = 0.95 / col_num

    start_i = 0
    while start_i < len(dataset_class_list):
        end_i = min(start_i + col_num, len(dataset_class_list))
        cur_row = dataset_class_list[start_i: end_i]
        for ci, (ds, cls) in enumerate(cur_row):
            if '.' in cls:
                cls = cls.replace('.', '_')
            if ' ' in cls:
                cls = cls.replace(' ', '_')
            cur_row[ci] = (ds, cls)

        # for ci, (ds, cls) in enumerate(cur_row):
        #     if ci == col_num - 1:
        #         endChar = '\\\\ \n'
        #     else:
        #         endChar = '& \n'
        #     latex_str += '\\verb|%s|%s' % (ds, endChar)
        #
        # for ci, (ds, cls) in enumerate(cur_row):
        #     if ci == col_num - 1:
        #         endChar = '\\\\ \n'
        #     else:
        #         endChar = '& \n'
        #     latex_str += '\\verb|%s|%s' % (cls, endChar)

        for img_folder in ['montage_images', 'texture_images', 'phrase_clouds']:
            for ci, (ds, cls) in enumerate(cur_row):
                if ci == col_num - 1:
                    endChar = '\\\\ \n'
                else:
                    endChar = '& \n'
                img_path = os.path.join('fgvc_figs', img_folder, ds, cls + '.png')
                latex_str += '\\includegraphics[width=%.2f\\linewidth]{%s}%s' % (width, img_path, endChar)
        latex_str += '\\hline\n'
        start_i = end_i

    latex_str += '\\end{tabular}\n\\caption{some caption}\n\\end{center}\n' + \
                 '\\end{figure*}\n'
    return latex_str


def latex_table1():
    examples = dict()
    examples['cub'] = ['018.Spotted_Catbird',
                       '014.Indigo_Bunting',
                       '016.Painted_Bunting',
                       '056.Pine_Grosbeak',
                       '110.Geococcyx',
                       '189.Red_bellied_Woodpecker',
                       '036.Northern_Flicker',
                       '074.Florida_Jay',
                       '162.Canada_Warbler']
    examples['oxford_flowers'] = ['class_10',
                                  'class_6',
                                  'class_15',
                                  'class_19',
                                  'class_33',
                                  'class_79',
                                  'class_22',
                                  'class_45',
                                  'class_94']
    '''
    examples['cars'] = []
    examples['aircrafts'] = []
    '''
    examples['flowers'] = ['Amaranthus spinosus',
                           'Amaranthus tricolor',
                           'Angelonia angustifolia',
                           'Bryophyllum delagoense',
                           'Convallaria majalis',
                           'Ipomoea aquatica']
    examples['butterflies'] = ['4e09ab97427939539c65136a6dd50cfc',
                               '14475cb029eeb89dd5e3a5a35f6b29ff',
                               '173e525cb6567fec860c44d7e97a1f76',
                               '2ed41e8c2224ab2cdbdf9fc7752cb65b',
                               '2ac2df74f5820eb844b3a41bccbc5439',
                               '38cead7abce6b6dd80ef8115ce967e73']
    examples['fungi'] = ['Agaricus xanthodermus',
                         'Agrocybe praecox',
                         'Boletus reticulatus',
                         'Cerioporus squamosus',
                         'Crepidotus cesatii',
                         'Kretzschmaria deusta']

    latex_str = '\\documentclass{article}\n\\usepackage{graphicx}\\begin{document}\n'
    d_cls_list = list()
    for ds, classes in examples.items():
        d_cls_list += [(ds, cls) for cls in classes]
    latex_str += generate_latex_table(d_cls_list[:18], col_num=6)
    latex_str += generate_latex_table(d_cls_list[18:], col_num=6)
    latex_str += '\n\\end{document}'
    with open('applications/max_texture_phrase_cloud/figs/fgvc_fig.tex', 'w') as f:
        f.write(latex_str)


def latex_table2():
    examples = dict()
    examples['cub'] = ['018.Spotted_Catbird',
                       '014.Indigo_Bunting',
                       '016.Painted_Bunting',
                       '056.Pine_Grosbeak',
                       '110.Geococcyx',
                       '189.Red_bellied_Woodpecker',
                       '036.Northern_Flicker',
                       '074.Florida_Jay',
                       '162.Canada_Warbler']
    examples['cub'] = [examples['cub'][i] for i in [2, 6, 7]]
    examples['oxford_flowers'] = ['class_10',
                                  'class_6',
                                  'class_15',
                                  'class_19',
                                  'class_33',
                                  'class_79',
                                  'class_22',
                                  'class_45',
                                  'class_94']
    examples['oxford_flowers'] = [examples['oxford_flowers'][i] for i in [0, 1, 3]]

    examples['butterflies'] = ['4e09ab97427939539c65136a6dd50cfc',
                               '14475cb029eeb89dd5e3a5a35f6b29ff',
                               '173e525cb6567fec860c44d7e97a1f76',
                               '2ed41e8c2224ab2cdbdf9fc7752cb65b',
                               '2ac2df74f5820eb844b3a41bccbc5439',
                               '38cead7abce6b6dd80ef8115ce967e73']
    examples['butterflies'] = [examples['butterflies'][i] for i in [0, 1, 2]]

    latex_str = '\\documentclass{article}\n\\usepackage{graphicx}\\begin{document}\n'
    d_cls_list = list()
    for ds, classes in examples.items():
        d_cls_list += [(ds, cls) for cls in classes]
    latex_str += generate_latex_table(d_cls_list, col_num=9)
    latex_str += '\n\\end{document}'
    with open('applications/max_texture_phrase_cloud/figs/fgvc_fig_final.tex', 'w') as f:
        f.write(latex_str)


if __name__ == '__main__':
    # gen_phrase_clouds()
    latex_table1()
    latex_table2()
