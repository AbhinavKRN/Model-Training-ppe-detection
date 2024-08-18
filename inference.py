import os
import argparse
import cv2
from ultralytics import YOLO

def perform_inference(input_dir, output_dir, person_det_model, ppe_det_model):
    os.makedirs(output_dir, exist_ok=True)
    
    person_model = YOLO(person_det_model)
    ppe_model = YOLO(ppe_det_model)
    
    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        person_results = person_model(img)
        for person_result in person_results:
            bbox = person_result['boxes']
            for box in bbox:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                ppe_results = ppe_model(cropped_img)
                for ppe_result in ppe_results:
                    ppe_bbox = ppe_result['boxes']
                    for ppe_box in ppe_bbox:
                        px1, py1, px2, py2 = ppe_box.xyxy[0].cpu().numpy()
                        cv2.rectangle(cropped_img, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
                
                img[int(y1):int(y2), int(x1):int(x2)] = cropped_img
        
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using YOLOv8 models.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for saving results.")
    parser.add_argument("person_det_model", type=str, help="Path to the YOLOv8 model for person detection.")
    parser.add_argument("ppe_det_model", type=str, help="Path to the YOLOv8 model for PPE detection.")
    args = parser.parse_args()
    
    perform_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_det_model)
