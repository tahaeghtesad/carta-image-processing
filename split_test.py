from video_handler import VideoHandler
import multiprocessing
from tqdm import tqdm

pbar = tqdm()


def split(i):
    VideoHandler.split(pbar, i)


if __name__ == '__main__':

    with multiprocessing.Pool(4) as pool:
        pool.map(split, [f'VID{i:03}' for i in range(1, 25)])

    pbar.close()
