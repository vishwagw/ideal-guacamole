# system name : Prompt-based Detector
# system version : 1.0.0
# system date : 2025-05-05

import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse

class PromptBasedObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the YOLOv8-based object detector.
        
        Args:
            model_path: Path to the YOLOv8 model weights
        """
        print("Initializing YOLOv8 Object Detector...")
        self.model = YOLO(model_path)
        
        # COCO dataset class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Initialize webcam
        self.cap = None
        print("Initialization complete!")
        
    def start_camera(self, camera_id=0):
        """Start the webcam capture"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera. Please check if the camera is connected.")
        return True
        
    def stop_camera(self):
        """Stop the webcam capture"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
            
    def get_frame(self):
        """Get a frame from the webcam"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                return None
            return frame
        return None
        
    def detect_specific_object(self, frame, target_object):
        """
        Detect if a specific object is in the frame
        
        Args:
            frame: The image frame to analyze
            target_object: The name of the object to detect
            
        Returns:
            detection_results: List of detections for the target object
        """
        if frame is None:
            return []
            
        # Normalize target object name
        target_object = target_object.lower().strip()
        
        # Check if the target object is in our class list
        if target_object not in self.class_names:
            print(f"Warning: '{target_object}' is not in the COCO dataset class names")
            return []
            
        # Get the target class ID
        target_class_id = self.class_names.index(target_object)
        
        # Run YOLOv8 inference
        results = self.model(frame)
        
        # Extract detections for the target class
        detection_results = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id == target_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf.item())
                    detection_results.append({
                        'class': target_object,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
        return detection_results
        
    def draw_detection_results(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: The image frame
            detections: List of detection results with class, confidence and bbox
            
        Returns:
            The frame with annotations
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        return frame
        
    def run_interactive_detection(self):
        """
        Run the interactive object detection loop where user can input object names
        """
        if not self.start_camera():
            print("Failed to start camera")
            return
            
        print("Camera started successfully!")
        print("\nPrompt-Based Object Detection System")
        print("===================================")
        print("Enter the name of an object to detect or 'q' to quit")
        print("Supported objects include: person, car, cat, dog, chair, etc.")
        
        while True:
            # Get frame
            frame = self.get_frame()
            if frame is None:
                print("Can't get frame from camera. Exiting...")
                break
                
            # Display the frame
            cv2.imshow("Prompt-Based Object Detection (Press 'q' to quit)", frame)
            
            # Wait for user input or quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            # Check if user has entered an object name
            target_object = input("\nEnter object name to detect (or 'q' to quit): ")
            if target_object == 'q':
                break
                
            # Skip empty inputs
            if not target_object.strip():
                continue
                
            print(f"Detecting '{target_object}'...")
            
            # Get a fresh frame for detection
            frame = self.get_frame()
            if frame is None:
                continue
                
            # Detect the specified object
            start_time = time.time()
            detections = self.detect_specific_object(frame, target_object)
            detection_time = time.time() - start_time
            
            # Display results
            if detections:
                print(f"✓ {target_object} detected! Found {len(detections)} instances.")
                for i, det in enumerate(detections):
                    print(f"  - Detection {i+1}: Confidence = {det['confidence']:.2f}")
                
                # Draw results on frame
                frame = self.draw_detection_results(frame, detections)
                cv2.imshow(f"'{target_object}' Detection Result (Press any key to continue)", frame)
                cv2.waitKey(0)
            else:
                print(f"✗ {target_object} not found in the current frame.")
                
            print(f"Detection completed in {detection_time:.2f} seconds")
            
        # Clean up
        self.stop_camera()

def main():
    parser = argparse.ArgumentParser(description="Prompt-Based Object Detection using YOLOv8")
    parser.add_argument('--model', type=str, default="yolov8n.pt", 
                        help="Path to YOLOv8 model weights (default: yolov8n.pt)")
    parser.add_argument('--camera', type=int, default=0, 
                        help="Camera device ID (default: 0)")
    
    args = parser.parse_args()
    
    try:
        detector = PromptBasedObjectDetector(model_path=args.model)
        detector.run_interactive_detection()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
