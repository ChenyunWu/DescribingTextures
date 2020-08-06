from yacs.config import CfgNode as CN

C = CN()
C.DEVICE = 'cuda'
C.RAND_SEED = 2020
C.OUTPUT_PATH = 'output/triplet_match/temp'
C.TRAIN_SPLIT = 'train'
C.EVAL_SPLIT = 'val'

C.LOAD_WEIGHTS = ''
C.INIT_WORD_EMBED = 'fast_text'  # rand / fast_text
C.LANG_INPUT = 'phrase'  # description

# config for model
C.MODEL = CN()
C.MODEL.VEC_DIM = 256
C.MODEL.IMG_FEATS = (2, 4)
C.MODEL.LANG_ENCODER = 'mean'
C.MODEL.DISTANCE = 'l2_s'  # l2, cos

# config for loss
C.LOSS = CN()
C.LOSS.MARGIN = 1.0
C.LOSS.IMG_SENT_WEIGHTS = (1.0, 1.0)

# config for training
C.TRAIN = CN()
C.TRAIN.TUNE_RESNET = True
C.TRAIN.TUNE_LANG_ENCODER = True
C.TRAIN.BATCH_SIZE = 16
C.TRAIN.MAX_EPOCH = 6
C.TRAIN.CHECKPOINT_EVERY_EPOCH = 0.5
C.TRAIN.EVAL_EVERY_EPOCH = 0.05

C.TRAIN.WEIGHT_DECAY = 1e-6
C.TRAIN.INIT_LR = 0.0001
C.TRAIN.LR_DECAY_GAMMA = 0.1
C.TRAIN.LR_DECAY_EVAL_COUNT = 10
C.TRAIN.EARLY_STOP_EVAL_COUNT = 40


# settings for Adam
C.TRAIN.ADAM = CN()
C.TRAIN.ADAM.ALPHA = 0.8
C.TRAIN.ADAM.BETA = 0.999
C.TRAIN.ADAM.EPSILON = 1e-8


def prepare(cfg):
    if cfg.INIT_WORD_EMBED == 'rand' or cfg.MODEL.LANG_ENCODER == 'lstm':
        cfg.TRAIN.TUNE_LANG_ENCODER = True
    if cfg.MODEL.LANG_ENCODER in ('elmo', 'bert'):
        cfg.TRAIN.TUNE_LANG_ENCODER = False
