codex规则：忽略这些文件[config/bev_lss_efficientnetb0_multitask_nuscenes_b3.py]，[config/bev_lss_efficientnetb0_multitask_nuscenes_begin.py], [config/bev_lss_efficientnetb0_multitask_nuscenes_gather1d.py], [config/fisheye3dod_config_resnet.py], [lmdbdata/calib_results.txt], [lmdbdata/nuscenes_dataset.py], [.gitignore], [./model/bevPool], [model/fisheye_lss.py], [model/gatherElements2gather.py], [model/gridsample_replace.py], [model/hbir_module.py], [model/multi_views.py], [model/view_transformerFisheye_begin.py], [model/view_transformerFisheye_conv_nouse.py], [export_hbir.py], [imageCheck.py], [infer_hbir.py], [*/__init__.py]，忽略这个文件夹[./model/bevPool]和[./paper_outputs/build]和[./paper_outputs/node_modules], 忽略文件夹：[./venv]

使用skill: Research-Paper-Writing-Skills和paperjsx和pdf-reading-mineru

使用pdf阅读skill

不应该关注耗时，而是没有gather, gridsample等内存不友好的算子，优化完善论文，内存访问模式差，本来速度就慢，matmul速度快

激活python3虚拟环境，source ./venv/bin/activate，包含了pytorch库和常见库

一些部分相关引用的PDF在目录：PDF， pdftotext 

输出在这个文件夹：paper_outputs，paper_outputs不读取编译文件夹build和node_modules，保持输出paper_outputs目录整洁，该删除的删除

可以联网的，可以使用浏览器，使用浏览器skill

npm config set registry https://registry.npmmirror.com

官方 LaTeX 模板 GitHub：https://github.com/cvpr-org/author-kit，已经下载好放到了目录github/author-kit

用英文写论文，需要导出可以发表的论文格式latex，排版按照通用论文格式，论文输出在这个目录：paper_outputs

论文背景：需要用4路或者多路鱼眼相机图片，送入到bev网络中，得到目标的3D框表示[x, y, z, w, l, h, yaw]，而且需要部署到NPU嵌入式设备上，需要保证网络的所有算子都被NPU支持，最开始直接用的项目的网络[https://github.com/weiyangdaren/Fisheye3DOD]，但是很多算子不支持，像是scatterND，2维的gather，gridsample算子，onehot，==算子，由于网络太多算子不支持，需要修改整个网络，得不偿失的，所以转到了地平线的技术架构，用地平线的模块来开发，但是地平线的bev算法包含gridsample算子，也不支持的，尝试了将gridsample改造成2维的gather算子，NPU还是不支持2维的gather算子，最终提出了不使用2维gather算子不使用gridsample算子的可部署网络结构，NDS基本达到了Fisheye3DOD的水平, mAP低一些，但是可以部署的。

本人所在公司官网：https://www.goyumeta.com，本人在公司的邮箱：jiuz@goyu-ai.com，本人常用邮箱：1069679911@qq.com

现有方法有什么不足： 板端嵌入式设备部署不友好，NPU不支持2维的gather算子, scatterND，gridsample算子，onehot，==算子，需要改造网络，有时候整个网络基本不能量化，不能部署的

给出代码写篇论文，代码主要依靠了地平线的容器训练环境，这个文件夹里面主要是训练和配置和评测，训练框架在容器内，地平线的容器下载链接是https://oe.horizon.auto/工具链版本是J6，主题是轻量化嵌入式设备可部署的bev模型，没有lss(Lift, Splat, Shoot-Encoding)，没有bevpool，没有2维的gather算子，没有gridsample算子

主要贡献是：提出了自学习矩阵，不需要外参lidar2camera，不需要ego2camera，不需要生成参考点，不需要生成网格点，不需要bev网格了，不需要采样，不需要gridsample，不需要2维的gather，完全通过网络自己学习拿到bev特征，主要的代码部分在[model/view_transformerFisheye.py], 在ViewTransformerFisheye类的初始函数__init__(*)，以及函数_spatial_transfom_learn，通过learning_point_version配置版本的，主要使用了矩阵来学习相关特征，论文中需要画出网络结构图，以及自学习矩阵的流程图、网络图，详细描述自学习矩阵的贡献，详细描述嵌入式部署的优势，整体网络比较小，部署没有算子造成困难，嵌入式设备基本都支持这些算子，量化友好的，PTQ量化还是QAT都比较好

实验设计： 先下载fisheye3dod数据集，然后打包成lmdb文件，最后用来训练网络，训练好以后评测结果

数据集主要使用了：https://github.com/weiyangdaren/Fisheye3DOD，fisheye3dod鱼眼仿真图片，真实数据集暂时不考虑，算是仿真数据集的作用

主要参考以下论文，其他相关论文引用，请上网搜索查找，加上完整的论文引用
Exploring Surround-View Fisheye Camera 3D Object Detection.pdf
1203引用 BEVDet-High-performance Multi-camera 3D Object Detection in Bird-Eye-View.pdf
1900引用 A generic camera model and calibration method for conventional, wide-angle, and ﬁsh-eye lense.pdf
1954引用 Lift, Splat, Shoot-Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D.pdf
5573引用 Objects as Points.pdf
52 Detecting As Labeling-Rethinking LiDAR-camera Fusion in 3D Object Detection.pdf

训练30多个epoch，fisheye3dod的4路鱼眼图片输入，数据集训练30多个epoch达到的测评标准，该网络分别有3个版本，配置方式是learning_point_version='1', '2', '3'，infer_float.py用模型推理可视化图片，鱼眼图片的结果是<img src="./infer_out/fisheye.jpg" />，写好完善的实验结果，训练的超参数在配置文件里面：config/bev_lss_efficientnetb0_multitask_nuscenes.py
```
fisheye3dod dataset https://github.com/weiyangdaren/Fisheye3DOD fisheye3dod鱼眼仿真图片
learning_point_version = 1，版本1
模型权重路径：model/ckpt/float-checkpoint-best1.pth.tar
================== Fisheye3DOD Evaluation ==================
mAP:         0.3047
mATE:        0.7160
mASE:        0.1324
mAOE:        0.4489
NDS:         0.4361
Eval time:   4.337 s
Object Class            AP      ATE     ASE     AOE     AP@0.5  AP@1.0  AP@2.0  AP@4.0
Car                     0.2594  0.7503  0.2042  0.1979  0.0108  0.1206  0.3538  0.5523
Van                     0.2940  0.6331  0.1875  0.1879  0.0324  0.1941  0.4078  0.5416
Truck                   0.3536  0.6987  0.1248  0.2484  0.0233  0.2168  0.4874  0.6869
Bus                     0.3795  0.7424  0.0052  0.3347  0.0274  0.2187  0.5302  0.7418
Pedestrian              0.2096  0.8393  0.0552  1.5313  0.0000  0.0790  0.3090  0.4504
Cyclist                 0.3322  0.6324  0.2177  0.1932  0.0373  0.2360  0.4518  0.6037

2026-10-02 21:25:00,605 WARNING [metric.py:200] Node[0] <class 'model.fcos3d_goyu_metric.Fcos3dGOYUMultiCamMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-10-02 21:25:00,605 INFO [metric_updater.py:360] Node[0] Epoch[0] Validation bev_lss_efficientnetb0_multitask_nuscenes: NDS[0.4361]
```

```
fisheye3dod dataset https://github.com/weiyangdaren/Fisheye3DOD fisheye3dod鱼眼仿真图片
learning_point_version = 2，版本2
模型权重路径：model/ckpt/float-checkpoint-best.pth2.tar
================== Fisheye3DOD Evaluation ==================
mAP:         0.3132
mATE:        0.7202
mASE:        0.1399
mAOE:        0.4448
NDS:         0.4392
Eval time:   8.745 s
Object Class            AP      ATE     ASE     AOE     AP@0.5  AP@1.0  AP@2.0  AP@4.0
Car                     0.2771  0.7258  0.2094  0.1960  0.0166  0.1492  0.3797  0.5628
Van                     0.3120  0.6359  0.2007  0.1859  0.0407  0.2077  0.4301  0.5694
Truck                   0.3452  0.7119  0.1213  0.2452  0.0219  0.2133  0.4867  0.6590
Bus                     0.3699  0.7592  0.0095  0.3649  0.0129  0.1974  0.5255  0.7440
Pedestrian              0.2286  0.8365  0.0658  1.4968  0.0000  0.1042  0.3446  0.4656
Cyclist                 0.3466  0.6517  0.2325  0.1799  0.0342  0.2406  0.4896  0.6220

2026-10-02 20:22:46,044 WARNING [metric.py:200] Node[0] <class 'model.fcos3d_goyu_metric.Fcos3dGOYUMultiCamMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-10-02 20:22:46,044 INFO [metric_updater.py:360] Node[0] Epoch[0] Validation bev_lss_efficientnetb0_multitask_nuscenes: NDS[0.4392]
```

```
fisheye3dod dataset https://github.com/weiyangdaren/Fisheye3DOD fisheye3dod鱼眼仿真图片
learning_point_version = 3，版本3
模型权重路径：model/ckpt/float-checkpoint-best3.pth.tar
================== Fisheye3DOD Evaluation ==================
mAP:         0.3533
mATE:        0.6478
mASE:        0.1319
mAOE:        0.4209
NDS:         0.4766
Eval time:   5.026 s
Object Class            AP      ATE     ASE     AOE     AP@0.5  AP@1.0  AP@2.0  AP@4.0
Car                     0.3327  0.6590  0.2048  0.1787  0.0363  0.2095  0.4513  0.6335
Van                     0.3609  0.5800  0.1907  0.1696  0.0620  0.2655  0.4919  0.6243
Truck                   0.3853  0.6435  0.1172  0.2504  0.0466  0.2680  0.5340  0.6926
Bus                     0.4056  0.6743  0.0026  0.2850  0.0311  0.2890  0.5512  0.7509
Pedestrian              0.2504  0.7657  0.0520  1.4801  0.0028  0.1416  0.3748  0.4822
Cyclist                 0.3851  0.5641  0.2240  0.1615  0.0783  0.3043  0.5230  0.6348

2026-10-02 19:35:18,004 WARNING [metric.py:200] Node[0] <class 'model.fcos3d_goyu_metric.Fcos3dGOYUMultiCamMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-10-02 19:35:18,004 INFO [metric_updater.py:360] Node[0] Epoch[0] Validation bev_lss_efficientnetb0_multitask_nuscenes: NDS[0.4766] 
```

```
fisheye3dod dataset https://github.com/weiyangdaren/Fisheye3DOD fisheye3dod鱼眼仿真图片
learning_point_version = 3，版本3
模型权重路径：model/ckpt/float-checkpoint-best333.pth.tar
================== Fisheye3DOD Evaluation ==================
mAP:         0.3515
mATE:        0.6578
mASE:        0.1333
mAOE:        0.3826
NDS:         0.4801
Eval time:   4.355 s
Object Class            AP      ATE     ASE     AOE     AP@0.5  AP@1.0  AP@2.0  AP@4.0
Car                     0.3277  0.6691  0.2052  0.1491  0.0334  0.2043  0.4477  0.6252
Van                     0.3547  0.5881  0.1938  0.1397  0.0595  0.2544  0.4848  0.6200
Truck                   0.3876  0.6509  0.1150  0.2291  0.0433  0.2748  0.5416  0.6908
Bus                     0.3952  0.6853  0.0032  0.2542  0.0322  0.2552  0.5407  0.7525
Pedestrian              0.2520  0.7736  0.0543  1.3926  0.0015  0.1401  0.3787  0.4876
Cyclist                 0.3919  0.5794  0.2282  0.1313  0.0761  0.3099  0.5379  0.6436

2026-10-02 19:56:22,980 WARNING [metric.py:200] Node[0] <class 'model.fcos3d_goyu_metric.Fcos3dGOYUMultiCamMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-10-02 19:56:22,980 INFO [metric_updater.py:360] Node[0] Epoch[0] Validation bev_lss_efficientnetb0_multitask_nuscenes: NDS[0.4801] 
```

和原始评测结果比较的时候，着重强调比较模型参数量的大小，训练好的模型在目录: horizon/model/ckpt，float-checkpoint-best.pth2.tar是版本2，float-checkpoint-best3.pth.tar是版本3，float-checkpoint-best1.pth.tar是版本1

原始github[https://github.com/weiyangdaren/Fisheye3DOD]的模型在这里：model/ckpt/fisheye_bevdet.pth

统计模型参数量的脚本是：model/statistic.py，运行得到的结果是：

项目https://github.com/weiyangdaren/Fisheye3DOD，已经下载好放到了目录github/Fisheye3DOD，model/ckpt/fisheye_bevdet.pth对应的配置文件是github/Fisheye3DOD/configs/fisheye_bevdet.py


原始github[https://github.com/weiyangdaren/Fisheye3DOD]的评测结果
```
原始github[https://github.com/weiyangdaren/Fisheye3DOD]的评测结果
================== Fisheye3DOD Evaluation ==================
mAP:         0.3821
mATE:        0.5906
mASE:        0.1637
mAOE:        0.4799
NDS:         0.4853
Eval time:   10.891s
Object Class            AP      ATE     ASE     AOE     AP@0.5  AP@1.0  AP@2.0  AP@4.0
Car                     0.4643  0.5577  0.1857  0.2384  0.1264  0.3618  0.6139  0.7552
Van                     0.3669  0.5602  0.1842  0.2884  0.0886  0.3088  0.4905  0.5797
Truck                   0.3797  0.6788  0.1710  0.3309  0.0585  0.2520  0.5325  0.6756
Bus                     0.4194  0.6229  0.0964  0.2666  0.0519  0.3406  0.5788  0.7061
Pedestrian              0.2982  0.5699  0.1282  1.4559  0.1039  0.2622  0.3771  0.4498
Cyclist                 0.3642  0.5542  0.2170  0.2993  0.1230  0.3171  0.4709  0.5458

05/27 02:21:03 - mmengine - INFO - Saved 4464 detections to /home/Desktop/data/mmdetection3d/work_dirs/20260527_020834/save_detection/epoch_14.pkl
05/27 02:21:03 - mmengine - INFO - Epoch(val) [14][4464/4464]    mAP: 0.3821  mATE: 0.5906  mASE: 0.1637  mAOE: 0.4799  NDS: 0.4853  Car_AP: 0.4643  Car_ATE: 0.5577  Car_ASE: 0.1857  Car_AOE: 0.2384  Car_AP@0.5: 0.1264  Car_AP@1.0: 0.3618  Car_AP@2.0: 0.6139  Car_AP@4.0: 0.7552  Van_AP: 0.3669  Van_ATE: 0.5602  Van_ASE: 0.1842  Van_AOE: 0.2884  Van_AP@0.5: 0.0886  Van_AP@1.0: 0.3088  Van_AP@2.0: 0.4905  Van_AP@4.0: 0.5797  Truck_AP: 0.3797  Truck_ATE: 0.6788  Truck_ASE: 0.1710  Truck_AOE: 0.3309  Truck_AP@0.5: 0.0585  Truck_AP@1.0: 0.2520  Truck_AP@2.0: 0.5325  Truck_AP@4.0: 0.6756  Bus_AP: 0.4194  Bus_ATE: 0.6229  Bus_ASE: 0.0964  Bus_AOE: 0.2666  Bus_AP@0.5: 0.0519  Bus_AP@1.0: 0.3406  Bus_AP@2.0: 0.5788  Bus_AP@4.0: 0.7061  Pedestrian_AP: 0.2982  Pedestrian_ATE: 0.5699  Pedestrian_ASE: 0.1282  Pedestrian_AOE: 1.4559  Pedestrian_AP@0.5: 0.1039  Pedestrian_AP@1.0: 0.2622  Pedestrian_AP@2.0: 0.3771  Pedestrian_AP@4.0: 0.4498  Cyclist_AP: 0.3642  Cyclist_ATE: 0.5542  Cyclist_ASE: 0.2170  Cyclist_AOE: 0.2993  Cyclist_AP@0.5: 0.1230  Cyclist_AP@1.0: 0.3171  Cyclist_AP@2.0: 0.4709  Cyclist_AP@4.0: 0.5458  data_time: 0.0094  time: 0.1486
```

只写latex论文，暂时不需要导出pptx文件和pdf文件和论文发表文件, 不需要编译，重点优化完善论文：paper_outputs/paper_main.tex

一篇论文通常包含、相关工作、方法、实验、结果、结论和局限。真正做汇报时，我们还要进一步提炼：



nuscenes validation result
```
nuscenes datasets数据集
learning_point_version = 11，版本11
模型权重路径：model/ckptnuscene/

Loading NuScenes tables for version v1.0-trainval...
Loading nuScenes-lidarseg...
Loading nuScenes-panoptic...
32 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
2631083 ego_pose,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
1166187 sample_annotation,
4 map,
34149 lidarseg,
34149 panoptic,
Done loading in 23.648 seconds.
======
Reverse indexing ...
Done reverse indexing in 2.8 seconds.
======
Initializing nuScenes detection evaluation
Loaded results from ./WORKSPACE/resultsbev_lss_efficientnetb0_multitask_nuscenes/results_nusc.json. Found detections for 6019 samples.
Loading annotations for val split from nuScenes version: v1.0-trainval
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6019/6019 [00:03<00:00, 1784.56it/s]
Loaded ground truth annotations for 6019 samples.
Filtering predictions
=> Original number of boxes: 234441
=> After distance based filtering: 202324
=> After LIDAR and RADAR points based filtering: 202324
=> After bike rack filtering: 202295
Filtering ground truth annotations
=> Original number of boxes: 187528
=> After distance based filtering: 134565
=> After LIDAR and RADAR points based filtering: 121871
=> After bike rack filtering: 121861
Accumulating metric data...
Calculating metrics...
Saving metrics to: ./WORKSPACE/resultsbev_lss_efficientnetb0_multitask_nuscenes
mAP: 0.0773
mATE: 0.9403
mASE: 0.2958
mAOE: 0.9346
mAVE: 1.1301
mAAE: 0.3126
NDS: 0.1903
Eval time: 20.3s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.182   0.844   0.181   0.474   1.479   0.286
truck   0.045   1.006   0.263   0.735   1.607   0.470
bus     0.102   0.873   0.229   0.438   2.080   0.371
trailer 0.013   1.165   0.275   0.837   0.483   0.052
construction_vehicle    0.004   0.982   0.514   1.653   0.084   0.267
pedestrian      0.059   0.936   0.313   1.482   0.980   0.765
motorcycle      0.042   0.918   0.282   1.320   1.987   0.241
bicycle 0.047   0.848   0.284   1.274   0.342   0.048
traffic_cone    0.157   0.913   0.340   nan     nan     nan
barrier 0.122   0.917   0.279   0.198   nan     nan
```

```
nuscenes datasets数据集
learning_point_version = 10，版本10
模型权重路径：model/ckptnuscene/
Loaded ground truth annotations for 6019 samples.
Filtering predictions
=> Original number of boxes: 237321
=> After distance based filtering: 204828
=> After LIDAR and RADAR points based filtering: 204828
=> After bike rack filtering: 204801
Filtering ground truth annotations
=> Original number of boxes: 187528
=> After distance based filtering: 134565
=> After LIDAR and RADAR points based filtering: 121871
=> After bike rack filtering: 121861
Accumulating metric data...
Calculating metrics...
Saving metrics to: ./WORKSPACE/resultsbev_lss_efficientnetb0_multitask_nuscenes
mAP: 0.0788
mATE: 0.9651
mASE: 0.2907
mAOE: 0.8978
mAVE: 1.0478
mAAE: 0.3455
NDS: 0.1895
Eval time: 19.6s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.187   0.830   0.185   0.452   1.403   0.287
truck   0.047   0.950   0.270   0.673   1.371   0.330
bus     0.105   0.991   0.259   0.345   2.092   0.475
trailer 0.012   1.201   0.262   0.926   0.449   0.067
construction_vehicle    0.006   1.144   0.454   1.627   0.116   0.325
pedestrian      0.060   0.959   0.303   1.477   0.975   0.815
motorcycle      0.050   0.917   0.279   1.058   1.496   0.362
bicycle 0.059   0.856   0.269   1.336   0.480   0.104
traffic_cone    0.162   0.908   0.351   nan     nan     nan
barrier 0.100   0.895   0.276   0.186   nan     nan
2026-08-27 09:03:53,698 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.1895, mAP:0.0788
car_AP: [0.5]:0.0000  [1.0]:0.0652  [2.0]:0.2489  [4.0]:0.4342 
truck_AP: [0.5]:0.0000  [1.0]:0.0016  [2.0]:0.0468  [4.0]:0.1410 
trailer_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0018  [4.0]:0.0457 
bus_AP: [0.5]:0.0000  [1.0]:0.0103  [2.0]:0.1344  [4.0]:0.2750 
construction_vehicle_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0005  [4.0]:0.0226 
bicycle_AP: [0.5]:0.0000  [1.0]:0.0114  [2.0]:0.0778  [4.0]:0.1483 
motorcycle_AP: [0.5]:0.0000  [1.0]:0.0066  [2.0]:0.0547  [4.0]:0.1369 
pedestrian_AP: [0.5]:0.0000  [1.0]:0.0032  [2.0]:0.0637  [4.0]:0.1747 
traffic_cone_AP: [0.5]:0.0000  [1.0]:0.0463  [2.0]:0.2221  [4.0]:0.3796 
barrier_AP: [0.5]:0.0000  [1.0]:0.0284  [2.0]:0.1490  [4.0]:0.2221 
```

```
nuscenes datasets数据集
learning_point_version = 9，版本9
模型权重路径：model/ckptnuscene/
Loaded ground truth annotations for 6019 samples.
Filtering predictions
=> Original number of boxes: 347924
=> After distance based filtering: 302584
=> After LIDAR and RADAR points based filtering: 302584
=> After bike rack filtering: 302524
Filtering ground truth annotations
=> Original number of boxes: 187528
=> After distance based filtering: 134565
=> After LIDAR and RADAR points based filtering: 121871
=> After bike rack filtering: 121861
Accumulating metric data...
Calculating metrics...
Saving metrics to: ./WORKSPACE/resultsbev_lss_efficientnetb0_multitask_nuscenes
mAP: 0.0599
mATE: 1.0179
mASE: 0.3113
mAOE: 0.9385
mAVE: 1.3683
mAAE: 0.3333
NDS: 0.1717
Eval time: 24.9s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.146   0.893   0.181   0.518   1.615   0.295
truck   0.028   1.105   0.313   0.743   1.716   0.324
bus     0.047   1.031   0.246   0.303   2.580   0.457
trailer 0.002   1.192   0.278   0.989   0.579   0.219
construction_vehicle    0.000   1.187   0.554   1.511   0.102   0.282
pedestrian      0.049   0.992   0.300   1.504   0.973   0.809
motorcycle      0.026   0.982   0.285   1.290   3.024   0.238
bicycle 0.041   0.873   0.292   1.381   0.357   0.042
traffic_cone    0.125   0.973   0.354   nan     nan     nan
barrier 0.134   0.952   0.310   0.208   nan     nan
```

## docker
docker_open_explorer_ubuntu_22_j6_gpu_v3.8.1.tar.gz

[https://oe.horizon.auto/download/oe](https://oe.horizon.auto/download/oe)

# 1. fisheye3dod

4路鱼眼图片输入

## dataset
* (1) firstly, generate `fisheye3dod_infos_train.pkl` and `fisheye3dod_infos_val.pkl`

    prepare same as: **Data Preparation** part [https://github.com/weiyangdaren/Fisheye3DOD](https://github.com/weiyangdaren/Fisheye3DOD)

    download dataset, unzip it, run `python3 projects/Fisheye3DOD/tools/fisheye3dod_converter.py`

* (2) secondly, pack data in lmdb format, run 
    ```bash []
    python3 lmdbdata/fisheye_carla_packer.py --src-data-dir ./Fisheye3DODdataset \\
      --meta_json_dir ./ImageSets-2hz --pack-type lmdb \\
      --target-data-dir ./lmdb --split-name train \\
      --num-workers 10

    python3 lmdbdata/fisheye_carla_packer.py --src-data-dir ./Fisheye3DODdataset \\
      --meta_json_dir ./ImageSets-2hz --pack-type lmdb \\
      --target-data-dir ./lmdb --split-name val \\
      --num-workers 10
    ```

## a pair of config and weight checkpoint
if you want to get a validation result, you should use correct config file and it's params, alse with correct checkpoint file.

>bev_lss_efficientnetb0_multitask_nuscenes.py   learning_point_version='1'  model/ckpt/float-checkpoint-best1.pth.tar

>bev_lss_efficientnetb0_multitask_nuscenes.py   learning_point_version='2'  model/ckpt/float-checkpoint-best.pth2.tar

>bev_lss_efficientnetb0_multitask_nuscenes.py   learning_point_version='3'  model/ckpt/float-checkpoint-best3.pth.tar ./model/ckpt/float-checkpoint-best333.pth.tar

## single gpu train
>python3 train.py --stage float --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py --device-ids 0

## multi gpu train
>python3 train.py --stage float --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py  --device-ids 0,1

## validation result

>python3 predict.py --stage float --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --device-ids 0 --learning_point_version 3 --ckpt ./model/ckpt/float-checkpoint-best3.pth.tar

>python3 predict.py --stage float --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --device-ids 0 --learning_point_version 2 --ckpt ./model/ckpt/float-checkpoint-best.pth2.tar

>python3 predict.py --stage float --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --device-ids 0 --learning_point_version 1 --ckpt ./model/ckpt/float-checkpoint-best1.pth.tar

>python3 predict.py --stage float --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --device-ids 0 --learning_point_version 3 --ckpt ./model/ckpt/float-checkpoint-best333.pth.tar

## visualization

>python3 infer_float.py --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --save-path ./infer_out --model-inputs ./demo/fisheye3dod_demo/16 \\
      --learning_point_version 3 --ckpt ./model/ckpt/float-checkpoint-best3.pth.tar

>python3 infer_float.py --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --save-path ./infer_out --model-inputs ./demo/fisheye3dod_demo/16 \\
      --learning_point_version 2 --ckpt ./model/ckpt/float-checkpoint-best.pth2.tar

>python3 infer_float.py --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --save-path ./infer_out --model-inputs ./demo/fisheye3dod_demo/16 \\
      --learning_point_version 1 --ckpt ./model/ckpt/float-checkpoint-best1.pth.tar

>python3 infer_float.py --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --save-path ./infer_out --model-inputs ./demo/fisheye3dod_demo/16 \\
      --learning_point_version 3 --ckpt ./model/ckpt/float-checkpoint-best333.pth.tar

## export float onnx
>python3 export_onnx.py --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \\
      --learning_point_version 3 --ckpt ./model/ckpt/float-checkpoint-best3.pth.tar

# 2. nuscenes

6路针孔图片输入

## dataset
download nuscenes [https://github.com/nutonomy/nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

*   pack data in lmdb format, run 
    ```bash []
    python3 lmdbdata/nuscenes_packer.py --src-data-dir ./nuscenes 
      --version ./v1.0-trainval --pack-type lmdb
      --target-data-dir ./nuscenes_lmdb --split-name train
      --num-workers 10

    python3 lmdbdata/nuscenes_packer.py --src-data-dir ./nuscenes 
      --version ./v1.0-trainval --pack-type lmdb
      --target-data-dir ./nuscenes_lmdb --split-name val
      --num-workers 10
    ```
version `v1.0-mini` is for exploring not for training.

## single gpu train
>python3 train.py --stage float --config ./config/nuscene_config.py --device-ids 0

## multi gpu train
>python3 train.py --stage float --config ./config/nuscene_config.py  --device-ids 0,1

## validation result

>python3 predict.py --stage float --config ./config/nuscene_config.py \\
      --device-ids 0 --learning_point_version 11 --ckpt ./model/ckptnuscene/float-checkpoint-best.pth111111.tar

>python3 predict.py --stage float --config ./config/nuscene_config.py \\
      --device-ids 0 --learning_point_version 10 --ckpt ./model/ckptnuscene/float-checkpoint-best.pth101010.tar

>python3 predict.py --stage float --config ./config/nuscene_config.py \\
      --device-ids 0 --learning_point_version 9 --ckpt ./model/ckptnuscene/float-checkpoint-best.pth99.tar

## visualization
>python3 infer_float.py --config ./config/nuscene_config.py --model-inputs ./demo/bev_lss_efficientnetb0_multitask_nuscenes --save-path ./demo

## export float onnx
>python3 --config ./config/nuscene_config.py


## reference
[https://doc.oe.horizon.auto/guide/release_note.html][https://doc.oe.horizon.auto/guide/release_note.html]
