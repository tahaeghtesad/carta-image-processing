configs = {
    # 'yolof': {
    #     'r_50_c5': {
    #         'config': 'mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py',
    #         'checkpoint': 'checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth',
    #         'person_class': 0,
    #         'color': (0, 0, 255)
    #     }
    # },
    'seasaw': {
        'r_50_fpn': {
            'config': 'mmdetection/configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py',
            'checkpoint': 'checkpoints/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-cd0f6a12.pth',
            'person_class': 0,
            'color': (0, 255, 0)
        }
    },
    # 'yolox': {
    #     'r_50_fpn': {
    #         'config': 'mmdetection/configs/yolox/yolox_tiny_8x8_300e_coco.py',
    #         'checkpoint': 'checkpoints/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth',
    #         'person_class': 0,
    #         'color': (255, 0, 0)
    #     }
    # }
}