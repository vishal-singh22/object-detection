import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import argparse
import os
from datetime import datetime
from collections import defaultdict
import threading

# Simple implementation of ByteTrack algorithm concepts
class ByteTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_id_count = 0
        self.tracks = []  # Active tracks
        self.lost_tracks = []  # Lost tracks
        self.track_history = defaultdict(list)  # Track history for visualization
        
    def update(self, detections):
        """Update tracks with new detections"""
        # If no tracks yet, initialize with all detections
        if not self.tracks:
            new_tracks = []
            for det in detections:
                new_track = {
                    'id': self.track_id_count,
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'conf': det['conf'],
                    'age': 1,
                    'hits': 1,
                    'time_since_update': 0,
                    'state': 'confirmed' if det['conf'] >= 0.5 else 'tentative'
                }
                self.track_id_count += 1
                new_tracks.append(new_track)
                
                # Add to history
                self.track_history[new_track['id']].append({
                    'bbox': new_track['bbox'],
                    'state': 'new'
                })
                
            self.tracks = new_tracks
            return self.tracks
        
        # Match detections with existing tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            
            # Update track with new detection
            track['bbox'] = det['bbox']
            track['class'] = det['class']
            track['conf'] = det['conf']
            track['hits'] += 1
            track['time_since_update'] = 0
            track['state'] = 'confirmed' if track['hits'] >= self.min_hits else 'tentative'
            
            # Add to history
            self.track_history[track['id']].append({
                'bbox': track['bbox'],
                'state': 'tracked'
            })
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = {
                'id': self.track_id_count,
                'bbox': det['bbox'],
                'class': det['class'],
                'conf': det['conf'],
                'age': 1,
                'hits': 1,
                'time_since_update': 0,
                'state': 'tentative'
            }
            self.track_id_count += 1
            self.tracks.append(new_track)
            
            # Add to history
            self.track_history[new_track['id']].append({
                'bbox': new_track['bbox'],
                'state': 'new'
            })
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track['time_since_update'] += 1
            track['age'] += 1
            
            # Change state to lost if not updated for a while
            if track['time_since_update'] > self.max_age:
                track['state'] = 'removed'
            else:
                track['state'] = 'lost'
                
                # Add to history for visualization
                if track['time_since_update'] <= 15:  # Only keep visible in history for a while
                    self.track_history[track['id']].append({
                        'bbox': track['bbox'],
                        'state': 'lost'
                    })
        
        # Remove tracks marked for removal
        self.tracks = [t for t in self.tracks if t['state'] != 'removed']
        
        return self.tracks
    
    def _match_detections_to_tracks(self, detections):
        """Match detections to existing tracks based on IoU"""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Calculate IoU between all detections and tracks
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._calculate_iou(track['bbox'], det['bbox'])
        
        # Apply Hungarian algorithm (simplified with greedy matching for speed)
        matched_indices = self._greedy_match(iou_matrix)
        
        # Filter matches based on IoU threshold
        matched_tracks = []
        for t_idx, d_idx in matched_indices:
            if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                matched_tracks.append((t_idx, d_idx))
        
        # Get unmatched detections and tracks
        matched_t_indices = [t for t, _ in matched_tracks]
        matched_d_indices = [d for _, d in matched_tracks]
        
        unmatched_tracks = [t_idx for t_idx in range(len(self.tracks)) if t_idx not in matched_t_indices]
        unmatched_detections = [d_idx for d_idx in range(len(detections)) if d_idx not in matched_d_indices]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
        
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def _greedy_match(self, iou_matrix):
        """Greedy matching algorithm (faster than Hungarian for our use case)"""
        # Make a copy since we'll modify it
        iou_matrix_copy = iou_matrix.copy()
        
        matched_indices = []
        
        # While there are still matches to make
        while iou_matrix_copy.size > 0 and iou_matrix_copy.max() > self.iou_threshold:
            # Get max IoU indices
            a, b = np.unravel_index(iou_matrix_copy.argmax(), iou_matrix_copy.shape)
            
            # Add to matches
            matched_indices.append((a, b))
            
            # Remove used rows and columns
            iou_matrix_copy[a, :] = 0
            iou_matrix_copy[:, b] = 0
        
        return matched_indices

class AsyncVideoWriter:
    def __init__(self, filename, fps, resolution):
        self.filename = filename
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._write_frames)
        self.thread.daemon = True
        self.thread.start()

    def add_frame(self, frame):
        with self.lock:
            self.frames.append(frame.copy())

    def _write_frames(self):
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.resolution)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.filename = self.filename.replace('.mp4', '.avi')  # Also change extension
        writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.resolution)

        while self.is_running or len(self.frames) > 0:
            if len(self.frames) > 0:
                with self.lock:
                    frame = self.frames.pop(0)
                writer.write(frame)
            else:
                time.sleep(0.001)  # Small sleep to prevent CPU hogging

        writer.release()

    def stop(self):
        print("[INFO] Waiting for all frames to be written...")
        # Wait until all frames are written
        while len(self.frames) > 0:
            time.sleep(0.01)

        self.is_running = False
        if self.thread:
            self.thread.join()
        print("[INFO] Video writing complete.")
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time object detection with ByteTrack')
    parser.add_argument('--source', type=str, default='0', 
                        help='Source for video (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda, cpu)')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Input resolution (WxH)')
    parser.add_argument('--process-resolution', type=str, default='384x384',
                        help='Processing resolution (WxH)')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='Number of frames to skip (0=no skip, 1=every other frame, etc.)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save output video')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output video')
    parser.add_argument('--display', action='store_true', default=True,
                        help='Display real-time detection')
    
    return parser.parse_args()

def preprocess_frame(frame, target_size):
    """Preprocess frame for model inference"""
    # Resize frame preserving aspect ratio
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create target image with padding
    target_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Paste resized image into target image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    target_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    # Convert to float32 and normalize
    img = target_img.astype(np.float32) / 255.0
    
    return img, (scale, x_offset, y_offset)

def draw_tracks(frame, tracks, classes_dict, show_lost=True):
    """Draw tracks on frame"""
    # Define colors for different track states
    colors = {
        'new': (0, 255, 0),      # Green for new tracks
        'tracked': (255, 255, 0), # Yellow for tracked objects
        'lost': (0, 0, 255)       # Red for lost tracks
    }
    
    for track in tracks:
        if track['state'] == 'removed':
            continue
            
        if track['state'] == 'lost' and not show_lost:
            continue
            
        # Get track info
        track_id = track['id']
        bbox = track['bbox']
        cls_name = classes_dict[track['class']]
        state = track['state']
        
        # Get color based on state
        color = colors.get(state, (255, 255, 0))  # Default to yellow
        
        # Draw bounding box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        
        # Draw label
        label = f"{cls_name} ID:{track_id}"
        if state == 'new':
            label += " NEW"
        elif state == 'lost':
            label += " MISSING"
            
        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1])-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                   
    return frame

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if needed
    if args.save_video and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Performance optimization settings
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking for faster conv operations
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
    cv2.setNumThreads(8)  # Optimize OpenCV threading
    
    # Parse resolutions
    width, height = map(int, args.resolution.split('x'))
    process_width, process_height = map(int, args.process_resolution.split('x'))
    process_size = (process_width, process_height)
    
    # Load YOLOv8 model
    print(f"Loading model {args.model} on {args.device}...")
    model = YOLO(args.model).to(args.device)
    
    # Use TensorRT if available
    if args.device == 'cuda' and hasattr(model, 'export') and hasattr(model.export, 'engine'):
        try:
            print("Attempting to use TensorRT acceleration...")
            model_engine = model.export.engine(half=True)
            if model_engine:
                model = model_engine
                print("TensorRT acceleration enabled")
        except Exception as e:
            print(f"TensorRT acceleration failed: {e}")
    
    # Use half precision if on CUDA
    if args.device == 'cuda':
        model.model.half()  # Use float16 (half precision) - reduces memory usage and improves speed
    
    # Initialize video capture
    print(f"Opening video source: {args.source}")
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    # Set input resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default FPS if not available
    
    # Initialize video writer if saving
    video_writer = None
    if args.save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = os.path.basename(args.source) if not args.source.isdigit() else "webcam"
        output_filename = f"{args.output_dir}/detection_{source_name}_{timestamp}.mp4"
        video_writer = AsyncVideoWriter(output_filename, fps, (width, height))
        video_writer.start()
        print(f"Saving output to {output_filename}")
    
    # Initialize ByteTracker
    tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Initialize variables
    current_frame = 0
    
    # For FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    fps_values = []  # Store all FPS values for final average
    fps_display_values = []  # Store recent FPS values for smoothed display
    
    # Skip frames to increase speed if needed
    skip_counter = 0
    
    # Create display window if needed
    if args.display:
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    
    print("Starting detection...")
    
    # Pre-allocate tensor for faster processing
    if args.device == 'cuda':
        # Pre-allocate CUDA tensors
        dummy_input = torch.zeros((1, 3, process_height, process_width), 
                                 device='cuda', dtype=torch.float16)
        _ = model(dummy_input)  # Warm up model
    
    # Main loop
    with torch.no_grad():  # Disable gradient calculation for inference
        while True:
            # Skip frames if needed
            if args.frame_skip > 0:
                skip_counter += 1
                if skip_counter % (args.frame_skip + 1) != 0:
                    ret, _ = cap.read()  # Read but don't process
                    if not ret:
                        break
                    continue
                
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame += 1
            
            # Calculate FPS
            new_frame_time = time.time()
            if prev_frame_time > 0:
                current_fps = 1/(new_frame_time-prev_frame_time)
                fps_values.append(current_fps)  # Store all FPS values for final average
                fps_display_values.append(current_fps)  # For display smoothing
                if len(fps_display_values) > 15:  # Keep only last 15 values for display
                    fps_display_values.pop(0)
                fps = sum(fps_display_values) / len(fps_display_values)  # Smoothed display FPS
            else:
                fps = 0
            prev_frame_time = new_frame_time
            
            # Resize and preprocess for faster processing
            preprocessed_frame, (scale, x_offset, y_offset) = preprocess_frame(frame, process_size)
            
            # Convert to tensor and send to device
            if args.device == 'cuda':
                input_tensor = torch.from_numpy(preprocessed_frame).permute(2, 0, 1).unsqueeze(0).half().to('cuda')
            else:
                input_tensor = torch.from_numpy(preprocessed_frame).permute(2, 0, 1).unsqueeze(0).float().to(args.device)
                
            # Run YOLOv8 detection
            results = model(input_tensor, conf=args.conf)
            
            # Process results
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Scale back to original frame
                    x1 = (x1 - x_offset) / scale
                    y1 = (y1 - y_offset) / scale
                    x2 = (x2 - x_offset) / scale
                    y2 = (y2 - y_offset) / scale
                    
                    # Create detection object
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': int(box.cls[0].item()),
                        'conf': float(box.conf[0].item())
                    })
            
            # Update tracker
            tracks = tracker.update(detections)
            
            # Draw tracks on frame
            draw_tracks(frame, tracks, model.names)
            
            # Display FPS (add color based on target)
            fps_color = (0, 255, 0) if fps >= 37 else (0, 165, 255) if fps >= 30 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 2)
            
            # Save frame to video if requested
            if args.save_video and video_writer is not None:
                video_writer.add_frame(frame)
            
            # Display the result
            if args.display:
                cv2.imshow("Detection", frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Calculate and display average FPS
    if len(fps_values) > 0:
        # Remove first few frames as they're often slower during initialization
        if len(fps_values) > 10:
            fps_values = fps_values[10:]
        
        avg_fps = sum(fps_values) / len(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
        
        print(f"\n===== Performance Statistics =====")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Minimum FPS: {min_fps:.2f}")
        print(f"Maximum FPS: {max_fps:.2f}")
        print(f"Processed Frames: {len(fps_values)}")
        print(f"Target FPS achieved: {'Yes' if avg_fps >= 35 else 'No'}")
        print(f"================================")
    
    # Clean up
    cap.release()
    if video_writer is not None:
        video_writer.stop()
    if args.display:
        cv2.destroyAllWindows()
    
    print("Detection complete.")
    return avg_fps if len(fps_values) > 0 else 0

if __name__ == "__main__":
    main()