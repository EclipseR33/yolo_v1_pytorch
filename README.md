# yolo_v1_pytorch
Paper: [[1506.02640] You Only Look Once: Unified, Real-Time Object Detection (arxiv.org)](https://arxiv.org/abs/1506.02640)

train.py中的超参数都再class CFG中
> root0712 存储了VOC07/12两年数据集的根目录
> model_root 存储模型路径
> backbone 'resnet'则代表使用修改后的resnet;'darknet'则代表使用原文中的darknet
> pretrain None则不使用预训练模型;或直接输入预训练权重地址
> with_amp 混合精度选项
> transforms 需要使用代码中定义的transforms而不是PyTorch直接给出的transforms
> start_epoch 中断训练重启的开始epoch
> epoch 总epoch数
> freeze_backbone_till 若为-1则不冻结backbone，其他数则会冻结backbone直到该epoch后解冻
