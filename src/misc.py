from os.path import exists
from os import makedirs

device = 'gpu:1'

def create_folder(fp):
    if not exists(fp):
        makedirs(fp)