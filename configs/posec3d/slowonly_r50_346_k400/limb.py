model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(3, 4, 6),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=400,
        dropout=0.5),
    test_cfg=dict(average_clips='prob'))

memcached = True
mc_cfg = ('localhost', 22077)
dataset_type = 'PoseDataset'
ann_file = 'data/k400/k400_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
skeletons = [[0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
             [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [0, 1], [0, 2],
             [1, 3], [2, 4], [11, 12]]
left_limb = [0, 2, 3, 6, 7, 8, 12, 14]
right_limb = [1, 4, 5, 9, 10, 11, 13, 15]
box_thr = 0.5
valid_ratio = 0.0

train_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True, double=True,
        left_kp=left_kp, right_kp=right_kp, left_limb=left_limb, right_limb=right_limb, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            split='train',
            pipeline=train_pipeline,
            box_thr=box_thr,
            valid_ratio=valid_ratio,
            memcached=memcached,
            mc_cfg=mc_cfg)),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='val',
        pipeline=val_pipeline,
        box_thr=box_thr,
        memcached=memcached,
        mc_cfg=mc_cfg),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='val',
        pipeline=test_pipeline,
        box_thr=box_thr,
        memcached=memcached,
        mc_cfg=mc_cfg))
# optimizer
optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 24
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/posec3d/slowonly_r50_346_k400/limb'
