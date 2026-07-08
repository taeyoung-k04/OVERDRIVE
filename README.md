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
python .\script\lane_detection\prepare_sem_class_dataset.py --clean
python .\script\lane_detection\train_sem_class.py
```

```bash
python -c "from ultralytics import YOLO; YOLO('runs/semantic/yolo_lane_sem_class/train_cpu_640_yolo26n_ade20k/weights/best.pt').export(format='onnx', imgsz=640)"
```
<br/>

### 🔸 Run Inference with the Semantic Segmentation Model
```bash
python .\script\lane_detection\infer_sem_class.py --backend onnx
python .\script\lane_detection\render_sem_class_video.py --backend onnx
python .\script\lane_detection\realtime_sem_class_camera.py --backend onnx --camera 1 --show-fps
```
<br/>
