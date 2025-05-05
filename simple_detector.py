# simple detector mode:
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import time
import os

class YOLOv8Detector:
    def __init__(self, model_path=None, conf_threshold=0.5):
        """
        Initialize the YOLOv8 detector
        
        Args:
            model_path: Path to a custom YOLOv8 model or name of a pre-trained model
            conf_threshold: Confidence threshold for detections
        """
        # If no model specified, use YOLOv8n (nano) which is small and fast
        if model_path is None:
            self.model = YOLO("yolov10n.pt")
        else:
            self.model = YOLO(model_path)
        
        self.conf_threshold = conf_threshold
        print(f"Model loaded successfully with confidence threshold: {conf_threshold}")
        
    def detect_image(self, image_path, save_output=True):
        """
        Perform object detection on a single image
        
        Args:
            image_path: Path to the input image
            save_output: Whether to save the output image with detections
            
        Returns:
            Image with detections drawn
            List of detection results
        """
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold)[0]
        
        # Process results
        img = cv2.imread(image_path)
        self._draw_detections(img, results)
        
        # Save output if requested
        if save_output:
            output_path = f"output_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, img)
            print(f"Detection results saved to {output_path}")
        
        return img, results
    
    def detect_video(self, video_source=0, save_output=False):
        """
        Perform object detection on video
        
        Args:
            video_source: Path to video file or camera index (default 0 for webcam)
            save_output: Whether to save the output video
        """
        # Open video source
        try:
            if isinstance(video_source, int):
                cap = cv2.VideoCapture(video_source)
                print(f"Opened camera {video_source}")
            else:
                cap = cv2.VideoCapture(video_source)
                print(f"Opened video file: {video_source}")
                
            if not cap.isOpened():
                print("Error: Could not open video source.")
                return
        except Exception as e:
            print(f"Error opening video source: {e}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  # Sometimes happens with webcams
            fps = 30
        
        # Setup video writer if saving output
        out = None
        if save_output:
            output_path = 'output_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        print("Starting detection. Press 'q' to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Measure FPS
            start_time = time.time()
            
            # Run inference
            results = self.model(frame, conf=self.conf_threshold)[0]
            
            # Draw detections
            self._draw_detections(frame, results)
            
            # Calculate and display FPS
            fps_current = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps_current:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("YOLOv8 Object Detection", frame)
            
            # Save frame if requested
            if out is not None:
                out.write(frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
    
    def _draw_detections(self, img, results):
        """
        Draw bounding boxes and labels on the image
        
        Args:
            img: The image to draw on
            results: YOLOv8 results object
        """
        # Get the boxes and class information
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        # Get class names
        class_names = results.names
        
        # Draw each detection
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = confs[i]
            cls_id = int(classes[i])
            
            # Get class name
            cls_name = class_names[cls_id]
            
            # Choose color based on class ID
            color = (int(hash(cls_name) % 256), 
                    int(hash(cls_name + "salt") % 256),
                    int(hash(cls_name + "pepper") % 256))
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{cls_name} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument("--source", type=str, default=None, 
                        help="Path to image or video file, or camera index")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLOv8 model or name of pre-trained model")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold for detections")
    parser.add_argument("--save", action="store_true", 
                        help="Save output with detections")
    
    args = parser.parse_args()
    
    # Initialize the detector
    detector = YOLOv8Detector(model_path=args.model, conf_threshold=args.conf)
    
    # Process based on input type
    if args.source is None:
        # Default to webcam if no source provided
        print("No source provided, using webcam")
        detector.detect_video(0, args.save)
    elif args.source.isdigit():
        # If source is a number, interpret as camera index
        detector.detect_video(int(args.source), args.save)
    elif args.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video file
        detector.detect_video(args.source, args.save)
    elif args.source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Process image file
        img, results = detector.detect_image(args.source, args.save)
        
        # Display image until key press
        cv2.imshow("YOLOv8 Object Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Unsupported file format: {args.source}")


if __name__ == "__main__":
    main()