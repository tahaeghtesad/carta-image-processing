import json


def load_dataset(path):
    with open(path) as fd:
        return json.load(fd)


def write_dataset(path, dataset):
    with open(path, 'w') as fd:
        json.dump(dataset, fd)