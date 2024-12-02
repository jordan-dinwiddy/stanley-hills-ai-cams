import cv2
from ultralytics import YOLO
import logging
import time
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load the YOLO model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # YOLOv8 Nano for fast inference

# Directory to save frames
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)
latest_frame_file = os.path.join(output_dir, "frame.jpg")  # Always contains the latest frame

def resize_frame(frame, target_width):
    """Resize frame to target width while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    return cv2.resize(frame, (target_width, new_height))

def process_rtsp_stream(rtsp_url):
    """Ingest RTSP stream, run YOLO detection on frames once per second, and save frames."""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"Failed to open RTSP stream: {rtsp_url}")
        return

    frame_count = 0
    last_inference_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame. Reconnecting...")
                time.sleep(1)
                continue

            frame_count += 1
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Unique timestamp for the frame

            # Run inference once per second
            if current_time - last_inference_time >= 1.0:
                last_inference_time = current_time
                logging.info(f"Running inference on frame {frame_count}...")

                # Run the frame through the YOLO model
                results = model(frame, conf=0.5)

                # Overlay detections on the frame
                for detection in results:
                    for cls, bbox, conf in zip(detection.boxes.cls, detection.boxes.xyxy, detection.boxes.conf):
                        class_name = detection.names[int(cls)]
                        x1, y1, x2, y2 = map(int, bbox.tolist())
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green color
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Add label with class name and confidence
                        label = f"{class_name} ({conf:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        logging.info(f"Detected: {class_name} with confidence {conf:.2f} at {bbox.tolist()}")

                # Save the high-resolution frame for inference but resize it before saving
                resized_frame = resize_frame(frame, target_width=640)

                # Save the frame with a unique timestamp
                frame_file = os.path.join(output_dir, f"{timestamp}.jpg")
                cv2.imwrite(frame_file, resized_frame)
                logging.info(f"Saved frame to {frame_file}")

                # Update the latest frame file
                cv2.imwrite(latest_frame_file, frame)
                logging.info(f"Updated latest frame to {latest_frame_file}")

    except KeyboardInterrupt:
        logging.info("Stream processing stopped by user.")
    finally:
        cap.release()
        logging.info("RTSP stream closed.")

if __name__ == "__main__":
    # Replace with your camera's RTSP URL
    rtsp_url = "rtsps://10.1.1.134:7441/3W12O36llGYVR68L?enableSrtp"

    # Print log messages to the console
    logging.info(f"Opening RTSP stream: {rtsp_url}")

    process_rtsp_stream(rtsp_url)