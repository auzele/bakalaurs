# Model structure and approach based on:
# Wang, C.-Y., & Liao, H.-Y. M. (2024). YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.
# arXiv preprint arXiv:2402.13616. https://arxiv.org/abs/2402.13616

from ultralytics import YOLO


model = YOLO("yolov9c-seg.pt") 


model.train(
    data="../data.yaml",     
    epochs=50,                  
    imgsz=640,                  
    batch=4,                     
    name="yolov9c_seg",       
    device="cpu",               
    pretrained=True,             
    project="runs/segment",     
    task="segment",             
)
