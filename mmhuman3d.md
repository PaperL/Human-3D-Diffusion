# mmhuman3d

## Command

安装 mmcv 命令

```
pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.10.1/index.html
```



demo 命令（single person）

```
python demo/estimate_smpl.py  configs/hmr/resnet50_hmr_pw3d.py pretrained/resnet50_hmr_pw3d-04f40f58_20211201.pth  --single_person_demo  --det_config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py  --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth  --input_path  demo/resources/single_person_demo.mp4  --output outputs/result  --draw_bbox  --device cuda --show_path outputs/
```



```
python demo/estimate_smpl.py  configs/hmr/resnet50_hmr_pw3d.py pretrained/resnet50_hmr_pw3d-04f40f58_20211201.pth  --single_person_demo  --det_config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py  --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth  --input_path  ../PW3D/imageFiles/courtyard_arguing_00  --output outputs/result  --draw_bbox  --device cuda
```

- `--det_config` 和 `--det_checkpoint` 是和 mmdetection 有关的
- `--input_path` 是图片或者视频的路径，也可以是一个文件夹下面包含很多图片/视频。看代码好像不能二层文件夹。
- `--output` 是输出文件夹
- `--show_path` 如果不指定（为 None）就不会渲染出视频



`courtyard_arguing_00` 两个人？



multi person

```
python demo/estimate_smpl.py  configs/hmr/resnet50_hmr_pw3d.py pretrained/resnet50_hmr_pw3d-04f40f58_20211201.pth  --multi_person_demo  --tracking_config demo/mmtracking_cfg/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py --input_path  ../PW3D/imageFiles/courtyard_arguing_00  --output outputs/result  --draw_bbox  --device cuda
```



test 命令

```
python tools/test.py configs/hmr/resnet50_hmr_pw3d.py pretrained/resnet50_hmr_pw3d-04f40f58_20211201.pth --work-dir=work/latest.pth --metrics pa-mpjpe mpjpe
```





## Body Model

### SMLP

models/body_models/smlp.py

data/body_models/SMLP/

J_regressor



```
keypoints_3d = output['joints']
keypoints_2d = self.camera.transform_points_screen(keypoints_3d)
```



## Some APIs

- core/visualization/visualize_smpl.py

```python
_prepare_mesh(poses, betas, transl, verts, start, end, body_model)
"""
	Args
	===
	verts: 如果不是 None, 直接用 verts 推导出 joints
	transl: translations of smpl(x)
	body_model: 是一个具体的 torch 模型

	Return
	===
	vertices, joints, num_frames, num_person
"""

visualize_smpl_pose
"""
	可视化, 输出 video/gif/images (folder)
"""
```



- core/evaluation/eval_utils.py

```python
def keypoint_mpjpe(pred, gt, mask, alignment='none'):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (PA-MPJPE).
    batch_size: N
    num_keypoints: K
    keypoint_dims: C
```



- core/render/torch3d_renderer/

```python
"""
   一堆 render 模型
"""
```



- core/cameras/

```
transform_points_screen
```

