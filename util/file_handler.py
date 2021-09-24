import json
import os


def load_dataset(path):
    with open(path) as fd:
        return json.load(fd)


def write_dataset(path, dataset):
    folder = '/'.join([folder for folder in path.split('/')[:-1]])
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    with open(path, 'w') as fd:
        json.dump(dataset, fd)
