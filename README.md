# PROMPT-based object detection using deep learning (YOLO)

This Python application implements a prompt-based object detection system using YOLOv8. The system only detects objects when the user specifically enters the name of the object they're looking for.

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download YOLOv8 weights** (automatically done on first run if not present)
   - The script will use the smallest YOLOv8 model (yolov8n) by default
   - Alternatively, you can download manually from the Ultralytics repository

## Running the Application

1. **Basic usage**:
   ```bash
   python object_detection_system.py
   ```

2. **Advanced options**:
   ```bash
   # Specify a custom YOLOv8 model
   python object_detection_system.py --model yolov8m.pt
   
   # Specify a different camera (if you have multiple cameras)
   python object_detection_system.py --camera 1
   ```

## How to Use

1. Run the application and wait for the camera to initialize
2. A window will appear showing the camera feed
3. In the terminal, you'll be prompted to enter the name of an object to detect
4. Type the name of the object (e.g., "person", "car", "chair") and press Enter
5. The system will analyze the current frame and report if the object was found
6. If found, a new window will show the detection with bounding boxes
7. Press any key to continue and detect another object
8. To exit the program, type 'q' or press the 'q' key in the camera window

## Supported Objects

The system can detect the 80 object classes from the COCO dataset, including:
- person, bicycle, car, motorcycle, airplane, bus, train, truck, boat
- common animals: cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- household items: chair, couch, bed, dining table, tv, laptop, cell phone, book
- and many more

If you enter an object name that's not in the COCO dataset, the system will warn you.

## YOLO models:
For this project use one of following versions:
* V8
* v9
* v10
* v11

NOTE: earlier versions do not support the system.

## Troubleshooting

- **Camera not working**: Make sure your webcam is connected and not being used by another application
- **YOLOv8 installation issues**: Refer to the [Ultralytics documentation](https://docs.ultralytics.com/) for detailed installation instructions
- **Performance issues**: 
  - Try using a smaller model (yolov8n.pt) for faster detection
  - If you have a GPU, ensure CUDA is properly configured to speed up inference

## Basic options:

# Use webcam
python yolov8_object_detection.py

# Process an image
python yolov8_object_detection.py --source path/to/image.jpg

# Process a video
python yolov8_object_detection.py --source path/to/video.mp4

## Advanced options:
# Use a specific YOLOv8 model (default is yolov8n.pt)
python yolov8_object_detection.py --model yolov8s.pt

# Set confidence threshold (default is 0.5)
python yolov8_object_detection.py --conf 0.6

# Save output with detections
python yolov8_object_detection.py --source path/to/image.jpg --save


