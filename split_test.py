from video_handler import VideoHandler
import multiprocessing
from tqdm import tqdm


def split(id):
    VideoHandler.split(id)


if __name__ == '__main__':
    split(1)