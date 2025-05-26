from os.path import exists
from os import makedirs

def create_folder(fp):
    if not exists(fp):
        makedirs(fp)