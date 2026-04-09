import logging
import os
import warnings

os.environ.setdefault('WANDB_SILENT', 'true')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

logging.getLogger('wandb').setLevel(logging.ERROR)

warnings.filterwarnings('ignore', message=r'Gym has been unmaintained since 2022.*')
warnings.filterwarnings('ignore', message=r'The `format` argument was not provided, defaulting to `gif`\..*')
