from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")
results = model.val(data="Hackathon2_scripts/yolo_params.yaml", device="mps")
print(results)
