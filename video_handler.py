import logging
from multiprocessing.pool import ThreadPool
import csv

import cv2
import os

import numpy as np


class VideoHandler:

    @staticmethod
    def get_file_name_by_id(id):
        with open('dataset/info.csv') as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                if row['video_id'] == id:
                    return row['video original name'].split('.')[0]

    @staticmethod
    def extract_panes(image, pane_count=4):
        dim = image.shape
        panes = [
            image[:dim[0] // 2, : dim[1] // 2, :],
            image[dim[0] // 2:, : dim[1] // 2, :],
            image[:dim[0] // 2, dim[1] // 2:, :],
            image[dim[0] // 2:, dim[1] // 2:, :]
        ]
        return panes

    @staticmethod
    def merge_panes(panes):
        return np.hstack((
            np.vstack((panes[0], panes[1])),
            np.vstack((panes[2], panes[3]))
        ))

    @staticmethod
    def split(pbar, id, pane_count=4):

        file = VideoHandler.get_file_name_by_id(id)

        path = f'dataset/videos/{file}.avi'
        video = cv2.VideoCapture(path)

        if not os.path.isdir(f'dataset/split/{id}'):
            os.mkdir(f'dataset/split/{id}')

        for i in range(pane_count):
            if not os.path.isdir(f'dataset/split/{id}/pane_{i}'):
                os.mkdir(f'dataset/split/{id}/pane_{i}')

        success, image = video.read()

        dim = image.shape

        thread_pool = ThreadPool(4)

        count = 0
        while success:

            if pane_count == 4:
                panes = VideoHandler.extract_panes(image, pane_count)

                for i in range(pane_count):
                    assert panes[i].shape == (1080, 1920, 3), f'Shape mismatch: {panes[i].shpae}'

                thread_pool.map(lambda en: cv2.imwrite(f'dataset/split/{id}/pane_{en[0]}/frame_{count}.jpg', en[1]),
                                enumerate(panes))

                success, image = video.read()

            # pbar.update(1)
            count += 1

        thread_pool.close()

        return count
