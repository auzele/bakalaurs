# Model structure and approach based on Ultralytics YOLOv8 (v8.0.0):
# Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
# Licensed under AGPL-3.0.

from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt") 

# Trenēšanu
model.train(
    data="../data.yaml",         
    epochs=50,              
    imgsz=640,             
    batch=4,                  
    name="yolov8n_seg",      
    device="cpu",               
    pretrained=True,         
    project="runs/segment",     
    task="segment",            
)
