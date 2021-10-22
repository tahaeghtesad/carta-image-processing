import multiprocessing.pool

from util.video_handler import VideoHandler

if __name__ == '__main__':
    with multiprocessing.pool.ThreadPool(40) as tp:
        tp.starmap(VideoHandler.split, [(25, 6)])
