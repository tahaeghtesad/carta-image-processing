import logging
from multiprocessing.pool import ThreadPool
import csv

import cv2
import os

import numpy as np
from tqdm import tqdm

import util.file_handler
from struct_util import FixedSizeBuffer


class VideoHandler:

    def __init__(self, path, hold) -> None:
        super().__init__()
        self.path = path
        self.first_frames = []
        self.last_frames = []

        self.width = None
        self.height = None
        self.fps = None
        self.frame_count = None
        self.hold = hold
        self.__load_video()

    def __load_video(self):
        video_in = cv2.VideoCapture(self.path)
        count = 0

        self.width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(video_in.get(cv2.CAP_PROP_FPS))

        read_buffer = FixedSizeBuffer(self.hold)

        while True:
            ret, frame = video_in.read()
            if not ret:
                break
            read_buffer.add(frame)

            count += 1

            if count == self.hold:
                self.first_frames = read_buffer.buffer

        self.last_frames = read_buffer.buffer

        video_in.release()
        self.frame_count = count
        print(f'Video {self.path} loaded. {self.width}x{self.height}|{self.fps}fps|{count}')

    @staticmethod
    def get_file_name_by_id(video_id):
        with open('dataset/info.csv') as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                if int(row['video_id']) == video_id:
                    return row['video original name'].split('.')[0]

    @staticmethod
    def get_coco_annotation_by_id(video_id):
        with open('dataset/info.csv') as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                if int(row['video_id']) == video_id:
                    return int(row['past_ids'][3:])

    @staticmethod
    def get_info_row_by_id(id, info):
        with open('dataset/info.csv') as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                if int(row['video_id']) == id:
                    return row[info]

    @staticmethod
    def extract_panes(image, pane_count=4):
        dim = image.shape
        panes = [
            image[:dim[0] // 2, : dim[1] // 2, :],
            image[:dim[0] // 2, dim[1] // 2:, :],
            image[dim[0] // 2:, : dim[1] // 2, :],
            image[dim[0] // 2:, dim[1] // 2:, :]
        ]
        return panes

    @staticmethod
    def merge_panes(panes):
        return np.hstack((
            np.vstack((panes[0], panes[2])),
            np.vstack((panes[1], panes[3]))
        ))

    @staticmethod
    def split(id, pane_count=4):

        file = VideoHandler.get_file_name_by_id(id)

        path = f'dataset/videos/{file}.avi'
        video = cv2.VideoCapture(path)
        pbar = tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT))

        dataset = {
            'images': [],
            'categories': [{
                'id': 1,
                'name': 'person',
                'supercategory': 'none'
            }],
            'annotations': []
        }

        if not os.path.isdir(f'dataset/split/video_{id}'):
            os.makedirs(f'dataset/split/video_{id}', exist_ok=True)

        for i in range(pane_count):
            if not os.path.isdir(f'dataset/split/video_{id}/pane_{i}'):
                os.makedirs(f'dataset/split/video_{id}/pane_{i}', exist_ok=True)

        success, image = video.read()
        assert success is True

        dim = image.shape

        thread_pool = ThreadPool(4)

        count = 0
        while success:

            if pane_count == 4:
                panes = VideoHandler.extract_panes(image, pane_count)
                for i in range(pane_count):
                    dataset['images'].append({
                        'id': count * pane_count + i,
                        'file_name': f'video_{id}/pane_{i}/frame_{count}.jpg',
                        'width': panes[i].shape[1],
                        'height': panes[i].shape[0]
                    })

                thread_pool.map(
                    lambda en: cv2.imwrite(f'dataset/split/video_{id}/pane_{en[0]}/frame_{count}.jpg', en[1]),
                    enumerate(panes))

                success, image = video.read()

            pbar.update(1)
            count += 1

        util.file_handler.write_json(f'dataset/split/annotations/video_{id}.coco.json', dataset)

        thread_pool.close()

        return count
