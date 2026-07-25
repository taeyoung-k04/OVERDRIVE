# OVERDRIVE
> 2026 SKKU Autonomous Driving Competition
<br/>

## ✨ Quick Start
> The environment is based on `Windows 11`, `Python 3.14.6`, `Arduino IDE 2.3.10`.
<br/>

### 🔸 Configure the `.env` file
&emsp;Copy the `.env.example` file to create `.env` file and fill in the required variables.  
<br/><br/>

### 🔸 Install Python Packages
```bash
pip install rplidar-roboticia opencv-python pyserial matplotlib
pip install python-dotenv ultralytics
```
<br/>

### 🔸 [Not Required] Train the Semantic Segmentation Model
```bash
New-Item -ItemType Directory -Force weights
curl.exe -L "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem-ade20k.pt" -o "weights/yolo26n-sem-ade20k.pt"
```

```bash
python .\script\lane_detection\prepare_sem_class_dataset.py --clean
python .\script\lane_detection\train_sem_class.py
```

```bash
python .\script\parking\prepare_sem_class_dataset.py --clean
python .\script\parking\train_sem_class.py
```

```bash
python -c "from ultralytics import YOLO; YOLO('runs/semantic/yolo_lane_sem_class/train_cpu_640_yolo26n_8class/weights/best.pt').export(format='onnx', imgsz=640)"
python -c "from ultralytics import YOLO; YOLO('runs/semantic/yolo_parking_sem_class/train_cpu_640_yolo26n_5class/weights/best.pt').export(format='onnx', imgsz=640)"
```
<br/>

### 🔸 Run Inference with the Semantic Segmentation Model
```bash
python .\script\lane_detection\infer_sem_class.py --backend onnx --postprocess
python .\script\lane_detection\render_sem_class_video.py --backend onnx --postprocess
python .\script\lane_detection\realtime_sem_class_camera.py --backend onnx --postprocess --camera 1 --show-fps
```

```bash
python .\script\parking\infer_sem_class.py --backend onnx
```
<br/>
