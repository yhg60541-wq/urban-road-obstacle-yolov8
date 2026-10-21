# Train YOLOv8s on the ROD-Dataset

Real-Time Obstacle Detection — 25 classes, 24,326 images.

## Before you run
1. **Settings → Accelerator → `GPU T4 x2`**
2. **Settings → Internet → ON**
3. **Add Data** → search `obstacle-detection-dataset` by `abtinzandi`


```python
!pip install -q ultralytics
```

    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m19.7 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import torch, ultralytics

print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '—')
```

    Creating new Ultralytics Settings v0.0.6 file ✅ 
    View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
    Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
    CUDA: True | Tesla T4
    

## Locate the dataset and write a fresh `data.yaml`

`/kaggle/input/` is read-only, so we generate a corrected `data.yaml` in `/kaggle/working/`.


```python
import yaml
from pathlib import Path

# Dataset path
INPUT_ROOT = Path('/kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset')

src = next(INPUT_ROOT.rglob('data.yaml'), None)
assert src is not None, f'data.yaml not found under {INPUT_ROOT}'
data_root = src.parent
print('Dataset root:', data_root)

with open(src) as f:
    orig = yaml.safe_load(f)

data_yaml = {
    'path':  str(data_root),
    'train': 'train/images',
    'val':   'valid/images',
    'test':  'test/images',
    'nc':    orig['nc'],
    'names': orig['names'],
}

out = Path('/kaggle/working/data.yaml')
with open(out, 'w') as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)

print('Wrote', out)
print('Classes:', data_yaml['nc'])
print(data_yaml['names'])
```

    Dataset root: /kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset
    Wrote /kaggle/working/data.yaml
    Classes: 25
    ['Bike', 'Building', 'Car', 'Person', 'Stairs', 'Traffic sign', 'Electrical Pole', 'Road', 'Motorcycle', 'Dustbin', 'Dog', 'Manhole', 'Tree', 'Guard rail', 'Pedestrian crosswalk', 'Truck', 'Bus', 'Bench', 'Traffic Cone', 'Fire hydrant', 'Teraffic Barrel', 'Plant Pot', 'Electrical Box', 'Chair', 'Bicycle Rack']
    

## Train


```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data    = '/kaggle/working/data.yaml',
    task    = 'detect',
    epochs  = 100,
    imgsz   = 640,
    batch   = 32,
    device  = 0,
    amp     = True,
    cache   = False,
    workers = 4,
    project = '/kaggle/working/runs',
    name    = 'rod_yolov8s',
    patience= 20,
    cos_lr  = True,
    plots   = True,
    seed    = 42,
)
```

    [KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt to 'yolov8s.pt': 100% ━━━━━━━━━━━━ 21.5MB 157.5MB/s 0.1s
    Ultralytics 8.4.50 🚀 Python-3.12.12 torch-2.10.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
    [34m[1mengine/trainer: [0magnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/kaggle/working/data.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8s.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=rod_yolov8s, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=20, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/kaggle/working/runs, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/kaggle/working/runs/rod_yolov8s, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=42, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=4, workspace=None
    [KDownloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 16.0MB/s 0.0s
    Overriding model.yaml nc=80 with nc=25
    
                       from  n    params  module                                       arguments                     
      0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
      1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
      2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]             
      3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
      4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
      5                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
      6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]           
      7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
      8                  -1  1   1838080  ultralytics.nn.modules.block.C2f             [512, 512, 1, True]           
      9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  1    591360  ultralytics.nn.modules.block.C2f             [768, 256, 1]                 
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
     16                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
     19                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  1   1969152  ultralytics.nn.modules.block.C2f             [768, 512, 1]                 
     22        [15, 18, 21]  1   2125723  ultralytics.nn.modules.head.Detect           [25, 16, None, [128, 256, 512]]
    Model summary: 130 layers, 11,145,275 parameters, 11,145,259 gradients, 28.7 GFLOPs
    
    Transferred 349/355 items from pretrained weights
    Freezing layer 'model.22.dfl.conv.weight'
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks...
    [KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 66.0MB/s 0.1s
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1mtrain: [0mFast image access ✅ (ping: 5.7±3.5 ms, read: 5.3±3.6 MB/s, size: 86.0 KB)
    [K[34m[1mtrain: [0mScanning /kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset/train/labels... 19186 images, 83 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 19186/19186 89.7it/s 3:34
    WARNING ⚠️ [34m[1mtrain: [0mCache directory /kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset/train is not writable, cache not saved.
    WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 3968, len(boxes) = 32251. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    [34m[1mval: [0mFast image access ✅ (ping: 0.8±0.7 ms, read: 8.4±1.6 MB/s, size: 57.1 KB)
    [K[34m[1mval: [0mScanning /kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset/valid/labels... 3511 images, 11 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 3511/3511 139.0it/s 25.3s
    WARNING ⚠️ [34m[1mval: [0mCache directory /kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset/valid is not writable, cache not saved.
    WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 821, len(boxes) = 5572. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
    [34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    [34m[1moptimizer:[0m MuSGD(lr=0.01, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
    Plotting labels to /kaggle/working/runs/rod_yolov8s/labels.jpg... 
    Image sizes 640 train, 640 val
    Using 2 dataloader workers
    Logging results to [1m/kaggle/working/runs/rod_yolov8s[0m
    Starting training for 100 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      1/100      7.54G      1.063       1.91      1.406         95        640: 100% ━━━━━━━━━━━━ 600/600 1.6it/s 6:12
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.7it/s 31.8s
                       all       3511       5572       0.72      0.701      0.745      0.537
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      2/100       7.1G      1.049      1.157      1.385         59        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:43
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.5it/s 36.1s
                       all       3511       5572      0.719       0.65      0.724      0.484
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      3/100       7.1G      1.145      1.334      1.461         70        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:25
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.5s
                       all       3511       5572      0.504      0.441      0.445       0.27
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      4/100      7.57G      1.211      1.461      1.517         78        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.6it/s 34.4s
                       all       3511       5572      0.703      0.599      0.672      0.434
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      5/100      7.42G      1.134      1.287      1.455         64        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:25
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.9s
                       all       3511       5572      0.755      0.682      0.759      0.524
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      6/100      7.45G      1.086      1.176      1.415         72        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:26
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.7s
                       all       3511       5572      0.787      0.711      0.789      0.565
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      7/100      6.99G      1.047      1.102       1.39         78        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:25
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.7s
                       all       3511       5572      0.782      0.715      0.795      0.571
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      8/100      7.32G      1.021      1.043      1.369         53        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:26
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.7s
                       all       3511       5572      0.816      0.732      0.822      0.597
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      9/100      7.18G     0.9905     0.9893      1.347         55        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.7s
                       all       3511       5572      0.795      0.794      0.844      0.625
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     10/100      8.31G     0.9735     0.9491      1.328         69        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.803      0.776      0.839      0.627
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     11/100      7.16G     0.9603     0.9273      1.319         53        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:29
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.8s
                       all       3511       5572      0.874      0.763      0.857      0.651
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     12/100      7.16G     0.9485     0.9037       1.31         58        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.836      0.815      0.871      0.662
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     13/100      7.16G     0.9309     0.8689      1.296         51        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.8s
                       all       3511       5572      0.845      0.796      0.862      0.653
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     14/100      7.16G      0.918     0.8503      1.291         62        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.859      0.786      0.866      0.667
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     15/100      7.16G     0.9064     0.8334      1.284         64        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.2s
                       all       3511       5572      0.859      0.796      0.877      0.673
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     16/100      7.16G     0.8956     0.8167      1.272         58        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.878      0.791      0.877      0.675
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     17/100      7.16G     0.8893     0.8009      1.267         55        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.878      0.799      0.879      0.679
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     18/100      7.16G     0.8813     0.7936      1.263         87        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.871      0.822      0.892      0.693
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     19/100      7.16G     0.8753     0.7752      1.257         64        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.871      0.816      0.882      0.692
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     20/100      7.16G     0.8706     0.7657      1.251         61        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.871      0.825      0.891      0.698
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     21/100      7.16G     0.8543     0.7419      1.239         55        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.3s
                       all       3511       5572      0.871      0.821      0.895      0.707
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     22/100      7.16G     0.8475     0.7377      1.236         66        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.871       0.83      0.899      0.709
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     23/100      7.16G     0.8469     0.7332      1.236         70        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.6s
                       all       3511       5572      0.869      0.838      0.902      0.709
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     24/100      7.16G     0.8388     0.7222      1.232         60        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.869      0.848      0.904      0.713
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     25/100      7.16G     0.8293     0.7186      1.224         55        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.884      0.833      0.902      0.717
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     26/100      7.16G      0.832     0.7067      1.224         64        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:33
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.3s
                       all       3511       5572      0.881      0.843      0.907       0.72
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     27/100      7.16G      0.822      0.695      1.218         55        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:26
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.8it/s 29.7s
                       all       3511       5572      0.896      0.838      0.909      0.723
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     28/100      7.16G     0.8175       0.69      1.214         58        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:26
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.5s
                       all       3511       5572      0.853      0.876      0.911      0.728
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     29/100      7.16G     0.8073     0.6802       1.21         69        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.1s
                       all       3511       5572      0.853      0.867      0.909      0.727
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     30/100      7.64G     0.8038     0.6756      1.206         63        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.895      0.841      0.913      0.728
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     31/100      7.21G     0.8049     0.6669      1.204         73        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.862      0.875       0.91       0.73
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     32/100      7.21G     0.7961     0.6636        1.2         65        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.885      0.846       0.91      0.731
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     33/100      7.21G     0.7947     0.6545      1.198         72        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.2s
                       all       3511       5572      0.888      0.845      0.911      0.734
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     34/100      7.21G     0.7873     0.6467      1.194         61        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.849       0.89      0.915      0.738
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     35/100      7.21G      0.778     0.6394       1.19         63        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.3s
                       all       3511       5572      0.853      0.889      0.917      0.737
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     36/100      7.21G     0.7732     0.6283      1.184         54        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:29
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.2s
                       all       3511       5572      0.886      0.853      0.916      0.738
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     37/100      7.21G     0.7683     0.6246      1.182         65        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:29
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.8s
                       all       3511       5572      0.893      0.846      0.915      0.737
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     38/100      7.21G     0.7658     0.6229      1.181         59        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.2s
                       all       3511       5572      0.886      0.856      0.913       0.74
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     39/100      7.21G     0.7597     0.6101      1.175         64        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.3s
                       all       3511       5572      0.889      0.855      0.912      0.739
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     40/100      7.21G     0.7523     0.5997       1.17         73        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.1s
                       all       3511       5572      0.883      0.864      0.913      0.741
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     41/100      7.21G     0.7541      0.608      1.173         78        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.2s
                       all       3511       5572       0.89      0.857      0.913      0.742
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     42/100      7.21G     0.7484     0.5943      1.165         62        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:29
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.3s
                       all       3511       5572      0.895      0.855      0.914      0.743
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     43/100      7.21G     0.7408     0.5896      1.165         59        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.1s
                       all       3511       5572        0.9      0.855      0.914      0.745
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     44/100      7.21G     0.7337     0.5854      1.159         60        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.0s
                       all       3511       5572      0.897      0.859      0.914      0.746
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     45/100      7.21G      0.731     0.5763      1.157         70        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.1s
                       all       3511       5572      0.854      0.899      0.915      0.746
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     46/100      7.21G     0.7239     0.5744      1.152         57        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.0s
                       all       3511       5572      0.899       0.86      0.916      0.746
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     47/100      7.21G     0.7203     0.5681      1.148         73        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.8s
                       all       3511       5572      0.885       0.87      0.917      0.746
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     48/100      7.21G     0.7147     0.5623      1.147         58        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:32
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.894      0.862      0.917      0.747
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     49/100      7.21G     0.7147     0.5579      1.144         61        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:31
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.888      0.865      0.917      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     50/100      7.21G      0.709     0.5522      1.143         56        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:29
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.9s
                       all       3511       5572      0.893      0.862      0.917      0.747
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     51/100      7.21G     0.7007     0.5444      1.139         59        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:31
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.6s
                       all       3511       5572      0.906      0.856      0.917      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     52/100      7.21G     0.6978     0.5433      1.136         59        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.2s
                       all       3511       5572      0.905      0.859      0.917      0.747
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     53/100      7.21G     0.6937     0.5378      1.133         62        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.9s
                       all       3511       5572      0.896      0.865      0.917      0.747
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     54/100      7.21G     0.6899     0.5309      1.128         62        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.9s
                       all       3511       5572      0.907      0.858      0.918      0.747
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     55/100      7.21G     0.6839     0.5253      1.127         44        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.8s
                       all       3511       5572      0.863      0.903      0.918      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     56/100      7.21G     0.6797     0.5205      1.123         50        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:31
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.1s
                       all       3511       5572      0.863      0.899      0.918      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     57/100      7.21G     0.6723     0.5136      1.118         54        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:29
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.3s
                       all       3511       5572      0.899      0.863      0.918      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     58/100      7.21G     0.6685     0.5078      1.116         67        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.895      0.864      0.918      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     59/100      7.21G     0.6594     0.5012      1.113         91        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.897      0.866      0.918      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     60/100      7.21G     0.6576     0.4991      1.108         78        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:31
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.898      0.867      0.918      0.749
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     61/100      7.21G     0.6581     0.4965      1.109         65        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.899      0.866      0.918      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     62/100      7.21G       0.65     0.4882      1.103         74        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.899      0.864      0.918      0.749
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     63/100      7.21G     0.6448     0.4861      1.103         56        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:34
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.5s
                       all       3511       5572        0.9      0.863      0.918      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     64/100      7.21G     0.6426     0.4813      1.101         69        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:36
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.3s
                       all       3511       5572        0.9      0.864      0.918      0.748
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     65/100      7.21G     0.6327     0.4748      1.098         37        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:35
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.5s
                       all       3511       5572      0.895      0.866      0.918      0.749
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     66/100      7.21G     0.6306      0.472      1.094         69        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:36
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.5s
                       all       3511       5572      0.894       0.87      0.918      0.749
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     67/100      7.21G     0.6281     0.4664      1.094         58        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:35
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.5s
                       all       3511       5572      0.895      0.868      0.918      0.749
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     68/100      7.21G     0.6209     0.4604      1.088         58        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:35
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.6s
                       all       3511       5572      0.894       0.87      0.918       0.75
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     69/100      7.21G     0.6163      0.457      1.086         71        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:36
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.3s
                       all       3511       5572      0.864        0.9      0.919       0.75
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     70/100      7.21G     0.6147     0.4545      1.084         58        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:36
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.2s
                       all       3511       5572      0.865        0.9      0.919      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     71/100      7.21G     0.6055     0.4474       1.08         64        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.864      0.904      0.919      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     72/100      7.21G     0.6062     0.4469       1.08         64        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.0s
                       all       3511       5572      0.866      0.906      0.919       0.75
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     73/100      7.21G     0.5988     0.4381      1.074         69        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.8s
                       all       3511       5572      0.867      0.906      0.918       0.75
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     74/100      7.21G     0.5925     0.4361      1.073         49        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.8s
                       all       3511       5572      0.867      0.901      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     75/100      7.21G     0.5891     0.4281      1.067         55        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.8s
                       all       3511       5572      0.868      0.903      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     76/100      7.21G     0.5835     0.4264      1.066         67        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.9s
                       all       3511       5572      0.872        0.9      0.919      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     77/100      7.21G     0.5826     0.4244      1.066         47        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.3s
                       all       3511       5572       0.87        0.9      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     78/100      7.21G     0.5782      0.419      1.062         84        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.9s
                       all       3511       5572      0.872      0.898      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     79/100      7.21G     0.5741     0.4179       1.06         75        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.872      0.899      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     80/100      7.21G     0.5731     0.4167       1.06         48        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.1s
                       all       3511       5572      0.872        0.9      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     81/100      7.21G     0.5655     0.4107      1.056         60        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.4s
                       all       3511       5572      0.872        0.9      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     82/100      7.21G     0.5627     0.4086      1.055         66        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.8s
                       all       3511       5572       0.87      0.903      0.918      0.752
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     83/100      7.21G     0.5604     0.4062      1.054         78        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.6s
                       all       3511       5572       0.87      0.903      0.917      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     84/100      7.21G     0.5539     0.4013      1.051         67        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.3s
                       all       3511       5572      0.871      0.903      0.917      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     85/100      7.67G     0.5523     0.4004      1.049         60        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:27
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.6s
                       all       3511       5572      0.872      0.903      0.918      0.752
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     86/100      7.08G     0.5455     0.3951      1.047         60        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.9s
                       all       3511       5572      0.871      0.903      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     87/100      7.42G     0.5495     0.3932      1.047         55        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.9s
                       all       3511       5572      0.872      0.904      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     88/100      7.06G     0.5413     0.3913      1.044         70        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.2s
                       all       3511       5572      0.873      0.903      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     89/100      7.42G      0.541     0.3905      1.042         76        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:29
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.0s
                       all       3511       5572      0.874      0.904      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     90/100      7.09G      0.539      0.389      1.042         62        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.5s
                       all       3511       5572      0.874      0.903      0.918      0.751
    Closing dataloader mosaic
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     91/100      7.09G       0.49     0.3252      1.019         49        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:26
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.0s
                       all       3511       5572      0.875      0.901      0.918      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     92/100      7.09G     0.4782     0.3092      1.013         25        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:25
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.8s
                       all       3511       5572      0.874      0.901      0.917      0.752
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     93/100      7.09G     0.4716     0.3043      1.007         29        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:25
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 27.7s
                       all       3511       5572      0.874      0.902      0.917      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     94/100      7.09G     0.4692     0.3002      1.006         29        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:26
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 2.0it/s 28.0s
                       all       3511       5572      0.874        0.9      0.916      0.752
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     95/100      7.09G     0.4634     0.2975      1.001         26        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:28
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 28.9s
                       all       3511       5572      0.875        0.9      0.917      0.752
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     96/100      7.09G     0.4647     0.2997      1.006         30        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:32
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.1s
                       all       3511       5572      0.876        0.9      0.916      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     97/100      7.09G     0.4576     0.2945     0.9972         36        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.1s
                       all       3511       5572      0.877        0.9      0.916      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     98/100      7.09G      0.456     0.2943     0.9993         38        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.1s
                       all       3511       5572      0.878      0.901      0.917      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     99/100      7.09G     0.4568     0.2952     0.9982         28        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.0s
                       all       3511       5572      0.878      0.901      0.916      0.751
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K    100/100      7.09G     0.4542      0.293     0.9984         20        640: 100% ━━━━━━━━━━━━ 600/600 1.8it/s 5:30
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.9it/s 29.2s
                       all       3511       5572      0.879        0.9      0.916      0.751
    
    100 epochs completed in 9.961 hours.
    Optimizer stripped from /kaggle/working/runs/rod_yolov8s/weights/last.pt, 22.5MB
    Optimizer stripped from /kaggle/working/runs/rod_yolov8s/weights/best.pt, 22.5MB
    
    Validating /kaggle/working/runs/rod_yolov8s/weights/best.pt...
    Ultralytics 8.4.50 🚀 Python-3.12.12 torch-2.10.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
    Model summary (fused): 73 layers, 11,135,259 parameters, 0 gradients, 28.5 GFLOPs
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 1.8it/s 31.3s
                       all       3511       5572      0.871      0.904      0.918      0.752
                      Bike         88         95       0.94      0.916      0.956      0.856
                  Building         95        101      0.783      0.901      0.874      0.711
                       Car        557       1003      0.874      0.875      0.923      0.654
                    Person        347       1033      0.813      0.842      0.873      0.604
                    Stairs        237        339      0.878       0.95      0.973      0.789
              Traffic sign         84        114      0.758      0.798      0.788      0.533
           Electrical Pole         71         85      0.868      0.953       0.97      0.728
                      Road         82         84      0.729      0.831      0.773      0.494
                Motorcycle         51         82      0.766      0.707      0.776      0.506
                   Dustbin         99        153       0.94      0.895      0.903      0.831
                       Dog         56         65      0.956      0.969      0.978      0.833
                   Manhole        455        480       0.95      0.954      0.976      0.856
                      Tree        225        293      0.822      0.894      0.893      0.654
                Guard rail        105        131      0.769      0.814      0.836      0.671
      Pedestrian crosswalk        317        429      0.849      0.851      0.879      0.721
                     Truck        134        159       0.94      0.893      0.926      0.723
                       Bus        100        150      0.865      0.887      0.926      0.714
                     Bench         38         40      0.881      0.922      0.928       0.77
              Traffic Cone        123        171      0.915      0.965      0.972      0.737
              Fire hydrant         25         25      0.936          1      0.995      0.976
           Teraffic Barrel         13         17      0.914      0.941      0.945      0.925
                 Plant Pot         25         37      0.952      0.946      0.962      0.841
            Electrical Box         27         29      0.942          1      0.995      0.975
                     Chair        317        419      0.864       0.94      0.965      0.837
              Bicycle Rack         32         38      0.881      0.947      0.964      0.853
    Speed: 0.2ms preprocess, 5.0ms inference, 0.0ms loss, 0.9ms postprocess per image
    Results saved to [1m/kaggle/working/runs/rod_yolov8s[0m
    

## Validate on the held-out test split
Reports mAP50, mAP50-95, precision, recall.


```python
# Reload best.pt — more reliable than using the training model object
best = Path(results.save_dir) / 'weights' / 'best.pt'
eval_model = YOLO(str(best))

metrics = eval_model.val(
    data   = '/kaggle/working/data.yaml',
    task   = 'detect',
    split  = 'test',
    imgsz  = 640,
    device = 0,
    plots  = True,
)
print(metrics.results_dict)
```

    Ultralytics 8.4.50 🚀 Python-3.12.12 torch-2.10.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
    Model summary (fused): 73 layers, 11,135,259 parameters, 0 gradients, 28.5 GFLOPs
    [34m[1mval: [0mFast image access ✅ (ping: 0.4±0.8 ms, read: 8.3±6.7 MB/s, size: 83.1 KB)
    [K[34m[1mval: [0mScanning /kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset/test/labels... 1629 images, 5 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1629/1629 125.1it/s 13.0s
    WARNING ⚠️ [34m[1mval: [0mCache directory /kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset/test is not writable, cache not saved.
    WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 416, len(boxes) = 2372. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 102/102 4.7it/s 21.7s
                       all       1629       2372      0.909      0.883      0.919      0.765
                      Bike         45         49      0.977      0.959       0.97      0.894
                  Building         54         54      0.861      0.981      0.951      0.826
                       Car        149        247      0.921      0.845      0.917      0.668
                    Person         88        320      0.885      0.842      0.913      0.654
                    Stairs        152        223      0.861      0.901      0.929       0.75
              Traffic sign         84        104      0.905      0.904      0.933      0.702
           Electrical Pole         41         42      0.884      0.911      0.936      0.767
                      Road         36         37      0.792      0.514      0.717      0.482
                Motorcycle         24         37      0.905      0.769      0.821      0.552
                   Dustbin         27         45      0.934      0.667      0.755      0.591
                       Dog         29         30      0.928          1      0.995      0.933
                   Manhole        231        242      0.937      0.934      0.946      0.865
                      Tree        114        150       0.85       0.86      0.883      0.654
                Guard rail         65         69      0.893      0.884      0.938      0.747
      Pedestrian crosswalk        153        189      0.931       0.82      0.895      0.771
                     Truck         38         43      0.954      0.837      0.914      0.704
                       Bus         35         55      0.863      0.855      0.881      0.691
                     Bench         28         28      0.872      0.975      0.987       0.84
              Traffic Cone         71         92      0.902      0.967      0.937      0.742
              Fire hydrant         25         25       0.97          1      0.995      0.972
           Teraffic Barrel         13         19      0.855          1      0.995      0.995
                 Plant Pot         24         34       0.99      0.853      0.901      0.786
            Electrical Box         27         34      0.961      0.941      0.983      0.884
                     Chair        128        168      0.897      0.946      0.968      0.839
              Bicycle Rack         26         36      0.988      0.917      0.915      0.817
    Speed: 0.9ms preprocess, 9.1ms inference, 0.0ms loss, 0.8ms postprocess per image
    Results saved to [1m/kaggle/working/runs/detect/val[0m
    {'metrics/precision(B)': 0.9086051455612258, 'metrics/recall(B)': 0.8833026831737762, 'metrics/mAP50(B)': 0.9190673326688189, 'metrics/mAP50-95(B)': 0.7650478931210041, 'fitness': 0.7650478931210041}
    

## Show results — training curves, confusion matrix, PR curve


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(path, title=''):
    path = Path(path)
    if not path.exists():
        print(f'[skip] {path.name} not found')
        return
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(mpimg.imread(str(path)))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

run_dir = Path(results.save_dir)
val_dir = Path(metrics.save_dir)

show_image(run_dir / 'results.png',                        'Training curves')
show_image(val_dir / 'confusion_matrix_normalized.png',    'Confusion Matrix (normalised)')
show_image(val_dir / 'confusion_matrix.png',               'Confusion Matrix (raw)')
show_image(val_dir / 'PR_curve.png',                       'Precision-Recall Curve')
show_image(val_dir / 'F1_curve.png',                       'F1-Confidence Curve')
```


    
![png](output_10_0.png)
    



    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    


    [skip] PR_curve.png not found
    [skip] F1_curve.png not found
    

## Predict on a handful of test images and display them


```python
import random, glob

test_imgs = sorted(glob.glob(str(data_root / 'test/images/*.jpg')))
samples = random.sample(test_imgs, 6)

preds = eval_model.predict(samples, imgsz=640, conf=0.25, save=True,
                           project='/kaggle/working/runs', name='predict')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, p in zip(axes.flat, sorted(glob.glob('/kaggle/working/runs/predict/*.jpg'))[:6]):
    ax.imshow(mpimg.imread(p))
    ax.axis('off')
plt.suptitle('YOLOv8s predictions (conf ≥ 0.25)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

    
    0: 640x640 1 Guard rail, 1 Chair, 13.2ms
    1: 640x640 1 Tree, 13.2ms
    2: 640x640 4 Cars, 13.2ms
    3: 640x640 1 Traffic sign, 13.2ms
    4: 640x640 2 Electrical Poles, 13.2ms
    5: 640x640 1 Traffic Cone, 13.2ms
    Speed: 1.8ms preprocess, 13.2ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 640)
    Results saved to [1m/kaggle/working/runs/predict[0m
    


    
![png](output_12_1.png)
    


## Copy best weights to `/kaggle/working/` for download


```python
import shutil

for src in (run_dir / 'weights' / 'best.pt', run_dir / 'weights' / 'last.pt'):
    if src.exists():
        dst = Path('/kaggle/working') / src.name
        shutil.copy(str(src), str(dst))
        print('Saved', dst)
```

    Saved /kaggle/working/best.pt
    Saved /kaggle/working/last.pt
    

## (Optional) Export for mobile / ONNX


```python
# eval_model.export(format='onnx', imgsz=640, opset=12)
# eval_model.export(format='tflite', imgsz=320, int8=True)
# eval_model.export(format='ncnn', imgsz=320)
```
