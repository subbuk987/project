import os
import shutil
import tempfile
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define upload directory
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use a trained model if available (e.g., 'best.pt')

# Waste categories mapping
WASTE_CATEGORIES = {
    'plastic': [39, 41],  # bottle, cup
    'paper': [73, 74],    # book, paper
    'metal': [76, 77],    # can, container
    'other': [0]          # default category
}

def process_video(video_path):
    """Processes the video and applies YOLOv8 + ByteTrack tracking."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {'error': 'Could not open video file'}

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a temporary file for the processed video
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_output.name

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Dictionaries to track objects and unique detections
    unique_detections = defaultdict(set)
    tracked_objects = {}

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # Perform detection with tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        if results is None or len(results) == 0:
            continue  # Skip if no detections
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract box details
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Ensure conversion from tensor
                cls = int(box.cls[0].cpu().numpy())  # Convert class ID from tensor
                conf = float(box.conf[0].cpu().numpy())  # Convert confidence score
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None  # Convert track ID

                # Process only high-confidence detections with tracking ID
                if conf > 0.3 and track_id is not None:
                    # Determine waste category
                    category = 'other'
                    for cat, ids in WASTE_CATEGORIES.items():
                        if cls in ids:
                            category = cat
                            break
                    
                    # Track object history
                    if track_id not in tracked_objects:
                        tracked_objects[track_id] = {
                            'category': category,
                            'class_id': cls,
                            'first_seen': frame_count,
                            'last_seen': frame_count
                        }
                    else:
                        tracked_objects[track_id]['last_seen'] = frame_count
                    
                    # Add object to unique detections
                    unique_detections[category].add(track_id)
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Display label with tracking ID
                    label = f"{category} #{track_id}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Write frame to output video
        out.write(frame)
    
    cap.release()
    out.release()

    # Process tracking statistics
    tracking_stats = {
        category: {
            'count': len(ids),
            'objects': [{'id': obj_id, 'duration': tracked_objects[obj_id]['last_seen'] - tracked_objects[obj_id]['first_seen']}
                        for obj_id in ids]
        }
        for category, ids in unique_detections.items()
    }
    
    return output_path, tracking_stats

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    """Handles video upload and processing."""
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    file.save(video_path)
    
    # Process the video
    output_path, tracking_stats = process_video(video_path)
    
    if output_path is None:
        return jsonify({'error': 'Failed to process video'}), 500
    
    # Move processed video to static folder
    processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')
    shutil.move(output_path, processed_video_path)
    
    return jsonify({
        'video_path': f'/static/uploads/processed_video.mp4',
        'tracking_stats': tracking_stats
    })

@app.route('/static/uploads/<filename>')
def serve_processed_video(filename):
    """Serves the processed video file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
