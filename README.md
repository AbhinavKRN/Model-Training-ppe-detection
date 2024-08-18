# Syook Object Detection Project

This project involves training object detection models for detecting persons and PPE (Personal Protective Equipment) using YOLOv8. The project includes scripts for converting annotations, training models, and performing inference.

## Project Structure

```
project_root/
│
├── pascalVOC_to_yolo.py       # Script for converting PascalVOC to YOLOv8 format
├── train_person_detection.py  # Script to train the person detection model
├── train_ppe_detection.py     # Script to train the PPE detection model
├── inference.py               # Script for performing inference using the trained models
├── datasets/                  # Directory for datasets (images and labels)
│   ├── images/
│   ├── labels/
│   ├── person_detection.yaml  # YAML file for person detection model training
│   └── ppe_detection.yaml     # YAML file for PPE detection model training

```

## Requirements

- Python 3.10+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- argparse

You can install the required Python packages using pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python argparse
```

## Setup

1. **Download the Dataset:**
   - Place your dataset inside the `datasets/` directory.
   - Ensure the `images/` and `labels/` directories are structured correctly within `datasets/`.

2. **Convert Annotations:**
   - Use the `pascalVOC_to_yolo.py` script to convert PascalVOC annotations to YOLOv8 format.
   - This script can be run directly, and it will read from the specified directories within the script.
   - Example:
     ```bash
     python pascalVOC_to_yolo.py
     ```

3. **Train the Person Detection Model:**
   - Use the `train_person_detection.py` script to train the YOLOv8 model for detecting persons.
   - Ensure the paths in the script are set correctly.
   - Example:
     ```bash
     python train_person_detection.py
     ```

4. **Train the PPE Detection Model:**
   - Use the `train_ppe_detection.py` script to train the YOLOv8 model for detecting PPE on cropped person images.
   - Example:
     ```bash
     python train_ppe_detection.py
     ```

5. **Perform Inference:**
   - Use the `inference.py` script to perform inference using both the person detection and PPE detection models.
   - This script takes input images, runs the person detection model to find persons, crops these regions, and then runs the PPE detection model on these cropped images.
   - Example:
     ```bash
     python inference.py --input_dir <input_directory> --output_dir <output_directory> --person_det_model <path_to_person_model_weights> --ppe_det_model <path_to_ppe_model_weights>
     ```

6. **Generate Report:**
   - Include a `report.pdf` that contains the approach, learning, and evaluation metrics for the models you trained.

## Troubleshooting

### Common Errors

- **OSError: [WinError 126] The specified module could not be found**:
  - This is likely an issue with your PyTorch installation. Ensure you have installed all necessary dependencies (e.g., Visual C++ Redistributable) and that your Python environment is correctly set up.

- **Incompatible Python Version**:
  - If you encounter issues, try using a different Python version (e.g., Python 3.10 or 3.9). Virtual environments are recommended.

### Tips

- **Using Virtual Environments**:
  - It's recommended to use a virtual environment to manage dependencies and avoid conflicts with system-wide packages.
  - Create and activate a virtual environment:
    ```bash
    python -m venv myenv
    myenv\Scripts\activate
    ```
  
- **Using Anaconda/Miniconda**:
  - If you face persistent issues with dependencies, consider using Anaconda or Miniconda to manage your environment.

## Conclusion

This project provides a full pipeline from annotation conversion to model training and inference for object detection tasks. It is designed to be modular, allowing easy extension and adaptation to different datasets or tasks.

For any issues or further questions, please consult the official PyTorch or Ultralytics documentation, or consider reaching out on relevant forums or GitHub issues.

---