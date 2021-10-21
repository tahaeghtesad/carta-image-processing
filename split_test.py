import multiprocessing.pool

from util.video_handler import VideoHandler

if __name__ == '__main__':
    with multiprocessing.pool.ThreadPool(4) as tp:
        tp.map(VideoHandler.split, range(1, 25))
