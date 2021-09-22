_base_ = ['mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py']
data_root = 'dataset/split/'
data = dict(
    test=dict(
        ann_file=data_root + 'annotations/VID001.coco.json',
        img_prefix=data_root + '/'
    )
)
