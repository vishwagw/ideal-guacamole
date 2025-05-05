# adding gui for program:
import cv2
import numpy as np
import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

class YOLOv8DetectorApp:
    def __init__(self, root):
        """Initialize the YOLOv8 detector application with GUI"""
        self.root = root
        self.root.title("YOLOv8 Object Detection")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Application state
        self.model = None
        self.cap = None
        self.is_running = False
        self.video_thread = None
        self.source_type = "webcam"  # webcam, video, image
        self.source_path = 0  # Default to webcam
        self.conf_threshold = 0.5
        self.target_objects = []
        self.current_frame = None
        self.frame_width = 640
        self.frame_height = 480
        
        # Available YOLOv8 models
        self.available_models = {
            "YOLOv8n (Nano)": "yolov8n.pt",
            "YOLOv8s (Small)": "yolov8s.pt",
            "YOLOv8m (Medium)": "yolov8m.pt",
            "YOLOv8l (Large)": "yolov8l.pt",
            "YOLOv8x (Extra Large)": "yolov8x.pt"
        }
        
        # Create GUI elements
        self._create_gui()
        
        # Load default model
        self._load_model("yolov8n.pt")
        
    def _create_gui(self):
        """Create all GUI components"""
        # Main layout - split into control panel and display area
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control Panel (left side)
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Source Selection
        ttk.Label(self.control_frame, text="Source:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.source_var = tk.StringVar(value="Webcam")
        ttk.Radiobutton(self.control_frame, text="Webcam", variable=self.source_var, value="Webcam", 
                        command=self._update_source).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="Video File", variable=self.source_var, value="Video", 
                        command=self._update_source).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="Image File", variable=self.source_var, value="Image", 
                        command=self._update_source).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # File selection button
        self.file_button = ttk.Button(self.control_frame, text="Select File", command=self._select_file, state=tk.DISABLED)
        self.file_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        
        # Selected file display
        self.file_label = ttk.Label(self.control_frame, text="Using webcam")
        self.file_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Model Selection
        ttk.Label(self.control_frame, text="YOLOv8 Model:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="YOLOv8n (Nano)")
        model_dropdown = ttk.Combobox(self.control_frame, textvariable=self.model_var, 
                                      values=list(self.available_models.keys()), state="readonly")
        model_dropdown.grid(row=5, column=1, padx=5, pady=5, sticky=tk.EW)
        model_dropdown.bind("<<ComboboxSelected>>", self._update_model)
        
        # Confidence threshold
        ttk.Label(self.control_frame, text="Confidence:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_slider = ttk.Scale(self.control_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                               variable=self.conf_var, command=self._update_conf)
        conf_slider.grid(row=6, column=1, padx=5, pady=5, sticky=tk.EW)
        self.conf_label = ttk.Label(self.control_frame, text="0.50")
        self.conf_label.grid(row=6, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Target object entry
        ttk.Label(self.control_frame, text="Target Objects:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_entry = ttk.Entry(self.control_frame)
        self.target_entry.grid(row=7, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(self.control_frame, text="Add", command=self._add_target).grid(row=7, column=2, padx=5, pady=5)
        
        # Target objects list
        self.target_listbox = tk.Listbox(self.control_frame, height=10)
        self.target_listbox.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        target_scrollbar = ttk.Scrollbar(self.control_frame, orient=tk.VERTICAL, command=self.target_listbox.yview)
        target_scrollbar.grid(row=8, column=2, sticky=tk.NS)
        self.target_listbox.configure(yscrollcommand=target_scrollbar.set)
        
        # Remove target button
        ttk.Button(self.control_frame, text="Remove Selected", command=self._remove_target).grid(
            row=9, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        
        # Start/Stop button
        self.start_button = ttk.Button(self.control_frame, text="Start Detection", command=self._toggle_detection)
        self.start_button.grid(row=10, column=0, columnspan=2, padx=5, pady=20, sticky=tk.EW)
        
        # Save result button
        self.save_button = ttk.Button(self.control_frame, text="Save Current Frame", command=self._save_frame, state=tk.DISABLED)
        self.save_button.grid(row=11, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        
        # Video Display Area (right side)
        self.display_frame = ttk.LabelFrame(self.main_frame, text="Video Feed")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video canvas
        self.canvas = tk.Canvas(self.display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Object detection results area
        self.results_frame = ttk.LabelFrame(self.display_frame, text="Detection Results")
        self.results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.results_text = tk.Text(self.results_frame, height=6, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X, padx=5, pady=5)
        
    def _update_source(self):
        """Update source type based on selection"""
        source_value = self.source_var.get()
        
        if source_value == "Webcam":
            self.source_type = "webcam"
            self.source_path = 0
            self.file_button.configure(state=tk.DISABLED)
            self.file_label.configure(text="Using webcam")
        else:
            self.file_button.configure(state=tk.NORMAL)
            if source_value == "Video":
                self.source_type = "video"
            else:
                self.source_type = "image"
            # Reset file path until user selects a file
            self.source_path = None
            self.file_label.configure(text="No file selected")
    
    def _select_file(self):
        """Open file dialog to select input file"""
        if self.source_type == "video":
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        else:  # image
            filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.source_path = file_path
            # Show only filename, not full path
            self.file_label.configure(text=os.path.basename(file_path))
    
    def _update_model(self, event=None):
        """Update the model based on dropdown selection"""
        model_name = self.model_var.get()
        model_path = self.available_models[model_name]
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load the YOLOv8 model"""
        self.status_var.set(f"Loading model {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.status_var.set(f"Model {model_path} loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.status_var.set("Error loading model")
    
    def _update_conf(self, value=None):
        """Update confidence threshold from slider"""
        self.conf_threshold = self.conf_var.get()
        self.conf_label.configure(text=f"{self.conf_threshold:.2f}")
    
    def _add_target(self):
        """Add a target object to the list"""
        target = self.target_entry.get().strip().lower()
        if target and target not in self.target_objects:
            self.target_objects.append(target)
            self.target_listbox.insert(tk.END, target)
            self.target_entry.delete(0, tk.END)
    
    def _remove_target(self):
        """Remove selected target object from the list"""
        selection = self.target_listbox.curselection()
        if selection:
            index = selection[0]
            self.target_objects.pop(index)
            self.target_listbox.delete(index)
    
    def _toggle_detection(self):
        """Start or stop detection"""
        if self.is_running:
            self._stop_detection()
        else:
            self._start_detection()
    
    def _start_detection(self):
        """Start the detection process"""
        if self.source_path is None and self.source_type != "webcam":
            messagebox.showwarning("Warning", "Please select an input file first.")
            return
        
        if self.model is None:
            messagebox.showwarning("Warning", "Model not loaded. Please wait or try another model.")
            return
        
        self.is_running = True
        self.start_button.configure(text="Stop Detection")
        
        if self.source_type == "image":
            # Process single image
            self._process_image()
        else:
            # Process video or webcam
            self._process_video()
    
    def _stop_detection(self):
        """Stop the detection process"""
        self.is_running = False
        self.start_button.configure(text="Start Detection")
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.video_thread is not None and self.video_thread.is_alive():
            self.video_thread.join()
    
    def _process_image(self):
        """Process a single image"""
        try:
            # Read the image
            img = cv2.imread(self.source_path)
            if img is None:
                raise ValueError(f"Could not read image from {self.source_path}")
            
            # Run detection
            self.status_var.set("Running detection...")
            results = self.model(img, conf=self.conf_threshold)[0]
            
            # Process results
            processed_img, detected_classes = self._draw_detections(img.copy(), results)
            self.current_frame = processed_img
            
            # Check for target objects
            self._update_detection_results(detected_classes)
            
            # Display the image
            self._update_canvas(processed_img)
            
            self.save_button.configure(state=tk.NORMAL)
            self.status_var.set("Detection completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {e}")
            self.status_var.set("Error processing image")
            self.is_running = False
            self.start_button.configure(text="Start Detection")
    
    def _process_video(self):
        """Process video or webcam feed"""
        self.save_button.configure(state=tk.NORMAL)
        
        # Start video processing in a separate thread
        self.video_thread = threading.Thread(target=self._video_detection_thread)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def _video_detection_thread(self):
        """Thread function for video processing"""
        try:
            # Open video source
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source {self.source_path}")
            
            self.status_var.set("Running detection...")
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Process frames
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    if self.source_type == "video":
                        # Video ended, restart from beginning
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # Camera error
                        break
                
                # Measure processing time
                start_time = time.time()
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold)[0]
                
                # Process results
                processed_frame, detected_classes = self._draw_detections(frame.copy(), results)
                self.current_frame = processed_frame
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Check for target objects and update results
                self._update_detection_results(detected_classes)
                
                # Update the display
                self._update_canvas(processed_frame)
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
            
            # Clean up
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
        except Exception as e:
            # Handle exceptions
            error_msg = f"Error in video processing: {e}"
            self.status_var.set(error_msg)
            print(error_msg)  # Print to console for debugging
            
            # Stop detection on error
            self.root.after(0, self._stop_detection)
    
    def _draw_detections(self, img, results):
        """
        Draw bounding boxes and labels on the image
        
        Returns:
            Processed image and list of detected classes
        """
        # Get the boxes and class information
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        # Get class names
        class_names = results.names
        
        # Track all detected classes
        detected_classes = {}
        
        # Draw each detection
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = confs[i]
            cls_id = int(classes[i])
            
            # Get class name
            cls_name = class_names[cls_id]
            
            # Add to detected classes with confidence
            if cls_name in detected_classes:
                if conf > detected_classes[cls_name]:
                    detected_classes[cls_name] = conf
            else:
                detected_classes[cls_name] = conf
            
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
            
        return img, detected_classes
    
    def _update_canvas(self, img):
        """Update the canvas with a new image"""
        # Convert from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas if needed
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has valid dimensions
            # Calculate aspect ratio
            img_aspect = img.shape[1] / img.shape[0]
            canvas_aspect = canvas_width / canvas_height
            
            if img_aspect > canvas_aspect:
                # Fit to width
                new_width = canvas_width
                new_height = int(new_width / img_aspect)
            else:
                # Fit to height
                new_height = canvas_height
                new_width = int(new_height * img_aspect)
            
            img_resized = cv2.resize(img_rgb, (new_width, new_height))
        else:
            img_resized = img_rgb
        
        # Convert to PhotoImage
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
        
        # Update canvas
        self.canvas.config(width=img_resized.shape[1], height=img_resized.shape[0])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
    
    def _update_detection_results(self, detected_classes):
        """Update detection results text area with target object status"""
        self.results_text.delete(1.0, tk.END)
        
        if not self.target_objects:
            self.results_text.insert(tk.END, "No target objects specified.\n")
            self.results_text.insert(tk.END, f"Detected {len(detected_classes)} different objects.")
            return
        
        # Check each target object
        for target in self.target_objects:
            found = False
            # Check if target matches or is a substring of any detected class
            for cls_name, conf in detected_classes.items():
                if target == cls_name.lower() or target in cls_name.lower():
                    self.results_text.insert(tk.END, f"✓ {target.upper()}: DETECTED (Confidence: {conf:.2f})\n")
                    found = True
                    break
            
            if not found:
                self.results_text.insert(tk.END, f"✗ {target.upper()}: NOT DETECTED\n")
    
    def _save_frame(self):
        """Save the current frame with detections"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame to save.")
            return
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_frame)
                self.status_var.set(f"Frame saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving frame: {e}")
                self.status_var.set("Error saving frame")
    
    def run(self):
        """Start the main event loop"""
        self.root.mainloop()
        # Clean up on exit
        self._stop_detection()


def main():
    # Create root window
    root = tk.Tk()
    app = YOLOv8DetectorApp(root)
    app.run()


if __name__ == "__main__":
    main()