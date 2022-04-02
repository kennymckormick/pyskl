model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        tcn_dropout=0.5,
        graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_hrnet.pkl'
pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='PadTo', length=300, mode='zero'),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=pipeline, split='xview_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=pipeline, split='xview_val'))
data['test'] = data['val']

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[2, 10])
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/stgcn/stgcn_vanilla_ntu60_xview_hrnet/j'
