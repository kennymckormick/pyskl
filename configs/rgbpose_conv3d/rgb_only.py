model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=2048,
        num_classes=60,
        dropout=0.5),
    test_cfg = dict(average_clips='prob'))

dataset_type = 'PoseDataset'
data_root = '/new-pool/dhd/data/nturgbd'
ann_file = 'data/nturgbd/ntu60_hrnet.pkl'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=10),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, split='xsub_train', ann_file=ann_file, data_prefix=data_root, pipeline=train_pipeline)),
    val=dict(type=dataset_type, split='xsub_val', ann_file=ann_file, data_prefix=data_root, pipeline=val_pipeline),
    test=dict(type=dataset_type, split='xsub_val', ann_file=ann_file, data_prefix=data_root, pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.15, momentum=0.9, weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 18
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/rgbpose_conv3d/rgb_only'
