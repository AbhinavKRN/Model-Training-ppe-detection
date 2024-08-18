import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(voc_dir, yolo_dir):
    os.makedirs(yolo_dir, exist_ok=True)
    
    for file in os.listdir(voc_dir):
        if file.endswith(".xml"):
            voc_file = os.path.join(voc_dir, file)
            tree = ET.parse(voc_file)
            root = tree.getroot()
            
            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)
            
            yolo_label = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = get_class_id(class_name)
                
                if class_id is None:
                    continue
                
                bbox = obj.find('bndbox')
                x_min = int(bbox.find('xmin').text)
                y_min = int(bbox.find('ymin').text)
                x_max = int(bbox.find('xmax').text)
                y_max = int(bbox.find('ymax').text)
                
                x_center = (x_min + x_max) / 2.0 / img_width
                y_center = (y_min + y_max) / 2.0 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                yolo_label.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            yolo_filename = os.path.join(yolo_dir, file.replace(".xml", ".txt"))
            with open(yolo_filename, "w") as f:
                f.write("\n".join(yolo_label))

def get_class_id(class_name):
    class_mapping = {
        "person": 0,
        "hard-hat": 1,
        "gloves": 2,
        "mask": 3,
        "glasses": 4,
        "boots": 5,
        "vest": 6,
        "ppe-suit": 7,
        "ear-protector": 8,
        "safety-harness": 9
    }
    return class_mapping.get(class_name)

if __name__ == "__main__":
    voc_dir = r"C:\Users\vlt08\OneDrive - Scaler School of Technology\Desktop\project\syook\datasets\labels"
    yolo_dir = r"C:\Users\vlt08\OneDrive - Scaler School of Technology\Desktop\project\syook\datasets\yolo_labels"
    
    convert_voc_to_yolo(voc_dir, yolo_dir)
