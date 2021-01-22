import os
import argparse
import datetime
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='unet', help='Architecture: unet or resnet')
parser.add_argument('--load', type=str, default=None,
                    help='loads pretrained model. Arguments model, batchnorm and dropout have to be set accordingly')
parser.add_argument('--multifilter', action="store_true", 
                    help="Activates data augmentation using multiple low-pass filters")
parser.add_argument('--batchnorm', action="store_true", help="Batch normalization")
parser.add_argument('--dropout', action="store_true", help="Dropout")
parser.add_argument('--test', action="store_true", help="Test only, no training")
parser.add_argument('--batchsize', type=int, default=8, help="Batch size")
parser.add_argument('--n_workers', type=int, default=8, help='number of cpu cores to use')
parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_ONLY = args.test

DROPOUT = 0.5 if args.dropout else 0.0

N_SONGS_VALID = 8     # number of songs for validation
DURATION_VALID = 8     # duration in seconds for validation
START_VALID = 8     # start of validation

METRIC_TRAIN = 'trn_snr'    # metric to adjust lr

N_SONGS_TEST = None
DURATION_TEST = None

LOAD_MODEL = args.load
MODEL = args.model

MULTIFILTER = args.multifilter

if args.lr > 0:
    LEARNING_RATE = args.lr
elif 'resnet' in MODEL:
    LEARNING_RATE = 1e-5
else:
    LEARNING_RATE = 5e-5

OVERWRITE_LR = False    # overwrites learning rate if model is loaded

BATCHNORM = args.batchnorm
BATCH_SIZE = args.batchsize

EXPERIMENT = MODEL

if BATCHNORM:
    EXPERIMENT += '_bn'
if DROPOUT > 0:
    EXPERIMENT += '_do'
if MULTIFILTER:
    EXPERIMENT += '_da'

ITER_VAL = 2500   # Tests, prints logs, saves models, samples every n iteration
VALID = True    # perform validation or not
# To adjust learning rate
PATIENCE = int(15 * 2500 / ITER_VAL)
LR_FACTOR = 0.1

# Low-pass filters. (type, order)
if MULTIFILTER:
    FILTERS_TRAIN = [
        ('cheby1', 6), ('cheby1', 8),
        ('cheby1', 10), ('cheby1', 12),
        ('bessel', 6), ('bessel', 12),
        ('ellip', 6), ('ellip', 12)
    ]
else:
    FILTERS_TRAIN = [('cheby1', 6)]
# By default, validation is done using training (seen) filters,
# so no need to specify again:
FILTERS_VALID = [('butter', 6)]     # unseen filter only
FILTERS_TEST = [('cheby1', 6), ('butter', 6)]
CUTOFF = 11025

SAMPLE_RATE = 44100

SAMPLE_LEN = 2**13   # length of training samples

# parameters for outputting wav files
WAV_SAMPLE_LEN = 2**13    # if not float, it is a ratio
WAV_BATCH_SIZE = BATCH_SIZE
TEST_DURATION = None    # None for entire song

PAD_TYPE = 'zero'

L_LOSS = 2      # L1 or L2 loss

MAX_ITER = 500000  # Training ends after this many iterations
MIN_LR = 1e-8   # Training ends after learning rate smaller then this

NUM_WORKERS = args.n_workers     # Number of CPU cores for loading and processing batches

ADAPTIVE_LR = True

SAVE_MODEL = True

assert LOAD_MODEL or not TEST_ONLY   # if you're testing, you need to load a model

DATE = datetime.datetime.now().strftime("%m-%d")
if TEST_ONLY:
    SAVE_NAME = LOAD_MODEL.replace('.pt', '')
else:
    SAVE_NAME = DATE + '_' + EXPERIMENT

MAIN_DIR = os.path.abspath('..')

SAVE_SAMPLES = True     # saves an entire song created by the model

OUTPUT_DIR = os.path.join(MAIN_DIR, 'output')

TRAIN_DIRS = [
    # os.path.join(MAIN_DIR, 'datasets', 'medleydb'),
    os.path.join(MAIN_DIR, 'datasets', 'DSD100', 'Mixtures', 'Dev')
]
TEST_DIRS = [os.path.join(MAIN_DIR, 'datasets', 'DSD100', 'Mixtures', 'Test')]

MODEL_DIR = os.path.join(MAIN_DIR, OUTPUT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
GENERATION_DIR = os.path.join(MAIN_DIR, OUTPUT_DIR, 'generation', SAVE_NAME)
os.makedirs(GENERATION_DIR, exist_ok=True)

# if using multiple filters, you need that many songs for validation
# so that it is one filter per song (each filter has equal weight in the average)
assert len(FILTERS_TRAIN) == 1 or len(FILTERS_TRAIN) == N_SONGS_VALID or TEST_ONLY
