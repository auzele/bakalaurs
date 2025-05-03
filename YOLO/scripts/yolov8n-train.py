from ultralytics import YOLO

# Ielādē YOLOv8m-seg modeli (vidēja izmēra modelis)
model = YOLO("yolov8n-seg.pt")  # vai mazāku: "yolov8n-seg.pt"

# Trenēšanu
model.train(
    data="../data.yaml",         # ceļš uz datu YAML
    epochs=50,                  # vairāk epohas, mazai datu kopai vajag ilgāku trenēšanu
    imgsz=640,                   # attēla izmērs
    batch=4,                     # batch size, paliek kā bija
    name="yolov8n_istais",       # kā nosauksies folderis zem runs/
    device="cpu",                # ja ir GPU, tad device="0"
    pretrained=True,             # sāk no ImageNet modeļa svaru (default, bet ļoti svarīgi mazām datu kopām)
    project="runs/segment",      # glabāsies `runs/segment/`
    task="segment",              # segmentācija
)
