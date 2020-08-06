import numpy as np
import time
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from applications.fine_grained_classification.cub_dataset import CUBDataset


def classify(cub_dataset=None, val_ratio=0.1, feat_mode='ph_cls', model_name='LogisticRegression_lbfgs_multinomial',
             norm=True, ks=None):
    if norm:
        norm_str = '_norm'
    else:
        norm_str = ''
    exp_name = '%s_%s%s_val%.1f' % (model_name, feat_mode, norm_str, val_ratio)
    print(exp_name)

    if cub_dataset is None or cub_dataset.val_ratio != val_ratio:
        cub_dataset = CUBDataset(split=None, val_ratio=val_ratio)
    labels = [img['class_label'] for img in cub_dataset.img_data_list]

    def att_feat_filter_rank(att_feat):
        att_vars = np.var(cub_dataset.class_att_labels, axis=0)
        att_idxs_sorted = np.argsort(att_vars * -1)

        att_type_str = feat_mode.split('_')[-1]
        att_idxs_allowed = set()
        if 's' in att_type_str:
            att_idxs_allowed.update(cub_dataset.att_types['shape'])
        if 'c' in att_type_str:
            att_idxs_allowed.update(cub_dataset.att_types['color'])
        if 'p' in att_type_str:
            att_idxs_allowed.update(cub_dataset.att_types['pattern'])

        att_idxs_filtered = [i for i in att_idxs_sorted if i in att_idxs_allowed]
        att_feat = att_feat[:, att_idxs_filtered]
        return att_feat

    feat_start_k = 0
    if feat_mode.startswith('ph_cls'):
        feats = np.load('applications/fine_grained_classification/phrase_scores_cls.npy')
        if feat_mode.endswith('sig'):
            feats = 1 / (1 + np.exp(-feats))
    elif feat_mode == 'ph_tri':
        feats = np.sqrt(-np.load('applications/fine_grained_classification/phrase_scores_tri.npy'))

    elif feat_mode.startswith('att_gt'):
        feats = np.array([d['att_labels'] for d in cub_dataset.img_data_list])
        feats = att_feat_filter_rank(feats)
    elif feat_mode.startswith('att_pred'):
        feats = np.load('applications/fine_grained_classification/att_scores.npy')
        feats = att_feat_filter_rank(feats)

    elif feat_mode.startswith('joint'):
        modes = feat_mode.split('_')
        if 'gt' in modes[1]:  # att_gt
            feat1 = np.array([d['att_labels'] for d in cub_dataset.img_data_list])
            feat1 = att_feat_filter_rank(feat1)
        else:  # att_pred
            feat1 = np.load('applications/fine_grained_classification/att_scores.npy')
            feat1 = att_feat_filter_rank(feat1)
        if 'cls' in modes[2]:  # ph_cls
            feat2 = np.load('applications/fine_grained_classification/phrase_scores_cls.npy')
        else:  # ph_tri
            feat2 = np.sqrt(-np.load('applications/fine_grained_classification/phrase_scores_tri.npy'))
        feats = np.concatenate((feat1, feat2), axis=1)
        feat_start_k = feat1.shape[1]
        # print(feats.shape)
    else:
        raise NotImplementedError

    if val_ratio > 0:
        eval_split = 'val'
    else:
        eval_split = 'test'
    test_feats = np.array([feats[i] for i in cub_dataset.img_splits[eval_split]])
    test_labels = np.array([labels[i] for i in cub_dataset.img_splits[eval_split]])
    train_feats = np.array([feats[i] for i in cub_dataset.img_splits['train']])
    train_labels = np.array([labels[i] for i in cub_dataset.img_splits['train']])
    # print(test_feats.shape)
    # print(test_labels.shape)
    # print(train_feats.shape)
    # print(train_labels.shape)

    f = open('applications/fine_grained_classification/logs/%s.txt' % exp_name, 'w')
    score = 0
    score_all = []

    if ks is None:
        if eval_split == 'val':
            ks = [100]
        elif feat_mode.startswith('ph') or feat_mode.startswith('joint'):
            ks = list(range(1, 50, 5)) + list(range(51, 656, 25)) + [655]
        elif feat_mode.startswith('att'):
            ks = list(range(1, 50, 2)) + list(range(51, 313, 25)) + [312]
        else:
            raise NotImplementedError
    elif ks[0] <= 0:
        ks = [train_feats.shape[-1]]

    model = None
    for k in ks:
        tic = time.time()
        train_feats_k = train_feats[:, :feat_start_k + k]
        test_feats_k = test_feats[:, :feat_start_k + k]
        print(train_feats_k.shape)
        print(test_feats_k.shape)
        if norm:
            train_feats_k = preprocessing.normalize(train_feats_k, norm='l2', axis=0)
            test_feats_k = preprocessing.normalize(test_feats_k, norm='l2', axis=0)

        if model_name.startswith('LogisticRegression'):
            settings = model_name.split('_')
            c = float(settings[3][1:])
            model = LogisticRegression(solver=settings[1], multi_class=settings[2], C=c, verbose=False,
                                       max_iter=10000, tol=1e-4, n_jobs=-1)
        elif model_name.startswith('DecisionTreeClassifier'):
            settings = model_name.split('_')
            cls_weight = settings[3]
            if cls_weight == 'none':
                cls_weight = None
            d = int(settings[4][1:])
            ss = int(settings[5][1:])
            sl = int(settings[6][1:])
            model = DecisionTreeClassifier(criterion=settings[1], splitter=settings[2], class_weight=cls_weight,
                                           max_depth=d, min_samples_split=ss, min_samples_leaf=sl)
        else:
            raise NotImplementedError
        model.fit(train_feats_k, train_labels)

        score = model.score(test_feats_k, test_labels)
        score_all.append((k, score))
        time_stamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        if model_name.startswith('DecisionTreeClassifier'):
            train_score = model.score(train_feats_k, train_labels)
            log_str = '%s %s: k = %d train_acc = %.3f, accuracy = %.3f; %.2f minutes' \
                      % (time_stamp, exp_name, k, train_score * 100, score * 100, (time.time() - tic) / 60)
        else:
            log_str = '%s %s: k = %d, accuracy = %.3f; %.2f minutes' \
                      % (time_stamp, exp_name, k, score * 100, (time.time() - tic) / 60)

        print(log_str)
        f.write(log_str + '\n')
        np.save('applications/fine_grained_classification/logs/%s.npy' % exp_name, score_all)

    f.close()
    return score, model


def rand_tune_tree(num):
    cub_dataset = CUBDataset(split=None, val_ratio=0.1)
    model_names = list()
    best_score = 0
    best_model = ''
    # settings = {'criterion': ['entropy', 'gini'],
    #             'splitter': ['best', 'random'],
    #             'class_weight': ['none', 'balanced'],
    #             'max_depth': range(10, 52, 5),
    #             'min_samples_split': range(2, 101, 5),
    #             'min_samples_leaf': range(1, 101, 5)}
    settings = [['entropy', 'gini'],
                ['best', 'random'],
                ['none', 'balanced'],
                range(2, 51, 1),
                range(2, 11, 1),
                range(1, 11, 1)]
    while len(model_names) < num:
        model_name = 'DecisionTreeClassifier'
        for i, ss in enumerate(settings):
            s = np.random.choice(ss)
            if i < 3:
                model_name += '_' + s
            elif i == 3:
                model_name += '_D%d' % s
            elif i == 4:
                model_name += '_S%d' % s
            elif i == 5:
                model_name += '_L%d' % s
        if model_name not in model_names:
            model_names.append(model_name)
            print('trial %d: %s' % (len(model_names), model_name))
            score = classify(cub_dataset=cub_dataset, feat_mode='ph_cls', model_name=model_name, norm=False,
                             val_ratio=0.1)
            if score > best_score:
                best_score = score
                best_model = model_name
                print('NEW BEST!')
    print(best_model, best_score)
    return


if __name__ == '__main__':
    # rand_tune_tree(500)
    # cub_dataset = CUBDataset(split=None, val_ratio=0.1)
    # model_name = 'LogisticRegression_newton-cg_multinomial_C1'
    # for norm in [False, True]:
    #     for feat_mode in ['joint_gt_cls', 'joint_pred_cls', 'joint_gt_tri', 'joint_pred_tri']:
    #         classify(cub_dataset=cub_dataset, feat_mode=feat_mode, model_name=model_name, norm=norm, val_ratio=0.1)

    #     for criterion in ['entropy', 'gini']:
    #         for splitter in ['best', 'random']:
    #             for cls_weight in ['none', 'balanced']:
    #                 for d in range(10, 52, 10):
    #                     model_name = 'DecisionTreeClassifier_%s_%s_%s_D%d' % (criterion, splitter, cls_weight, d)
    #                     classify(cub_dataset=cub_dataset, feat_mode='ph_cls', model_name=model_name, norm=norm,
    #                              val_ratio=0.1)

    cub_dataset = CUBDataset(split=None, val_ratio=0)
    model_name = 'LogisticRegression_newton-cg_multinomial_C1'
    # for f in ['att_gt_', 'joint_gt_cls_', 'att_pred_', 'joint_pred_cls_']:
    #     ks = [0]
    #     if f.startswith('joint'):
    #         ks = [100]
    #     for att_types in ['s', 'c', 'p', 'cp', 'sc', 'sp', 'scp']:
    #         classify(cub_dataset=cub_dataset, feat_mode=f + att_types, model_name=model_name, norm=False,
    #                  val_ratio=0, ks=ks)

    # classify(cub_dataset=cub_dataset, feat_mode='ph_cls_sig', model_name=model_name, norm=False,
    #          val_ratio=0, ks=None)
    for att_types in ['s', 'c', 'p', 'sp', 'scp', 'cp', 'sc']:
        classify(cub_dataset=cub_dataset, feat_mode='joint_gt_cls_' + att_types, model_name=model_name, norm=False,
                 val_ratio=0, ks=[100])
