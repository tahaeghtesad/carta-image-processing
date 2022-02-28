import multiprocessing.pool

from util.video_handler import VideoHandler

if __name__ == '__main__':
    with multiprocessing.pool.ThreadPool(40) as tp:
        tp.starmap(VideoHandler.split, [(i, 4) for i in range(1, 25)])

    # for i in range(1, 25):
    #     VideoHandler.split(i, 4)
