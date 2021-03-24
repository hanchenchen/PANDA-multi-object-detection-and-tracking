checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #'./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' # None
resume_from = None # './work_dirs/faster_rcnn_r50_fpn_1x_coco/latest.pth'
workflow = [('train', 1)]
