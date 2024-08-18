import ultralytics
from ultralytics import YOLO

def train_person_detection(data_yaml, model_name, epochs=50):
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, imgsz=640)

if __name__ == "__main__":
    data_yaml = "datasets/person_detection.yaml"
    model_name = "yolov8n.pt"
    train_person_detection(data_yaml, model_name)
