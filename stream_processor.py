import cv2
from ultralytics import YOLO
import logging
import time
import os
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load the YOLO model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # Replace with fine-tuned model if available

# Directory to save frames
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)
latest_frame_file = os.path.join(output_dir, "latest_frame.jpg")  # Always contains the latest frame

executor = ThreadPoolExecutor(max_workers=2)

# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=1, nms_max_overlap=0.5)

def update_tracks_async(detections, frame):
    return tracker.update_tracks(detections, frame=frame)

def resize_frame(frame, target_width):
    """Resize frame to target width while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    return cv2.resize(frame, (target_width, new_height))

def emit_event(object_id, object_class):
    """Emit an event for a newly detected object."""
    logging.info(f"Event: New {object_class} detected with ID {object_id}!")

def process_rtsp_stream(rtsp_url):
    """Ingest RTSP stream, run YOLO detection on frames, and track objects."""
    cap = cv2.VideoCapture(rtsp_url)
    #cap.set(cv2.CAP_PROP_FPS, 10)  # Set RTSP stream FPS to 10
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        logging.error(f"Failed to open RTSP stream: {rtsp_url}")
        return

    frame_count = 0
    last_inference_time = 0

    seen_track_ids = {}  # Store track IDs with their last seen timestamp
    EXPIRY_TIME = 5  # Time in seconds before a track can trigger a new event again

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame. Reconnecting...")
                cap.release()
                time.sleep(2)  # Wait before attempting reconnection
                cap = cv2.VideoCapture(rtsp_url)
                continue

            frame_count += 1
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Unique timestamp for the frame

            # Run inference once per second
            if current_time - last_inference_time >= 0.5:
                last_inference_time = current_time
                logging.info(f"Running inference on frame {frame_count}...")

                # Run the frame through the YOLO model
                results = model(frame, conf=0.5)

                # Prepare detections for DeepSORT
                detections = []
                classes = []
                is_new_detection = False

                for detection in results:
                    for cls, bbox, conf in zip(detection.boxes.cls, detection.boxes.xyxy, detection.boxes.conf):
                        class_name = detection.names[int(cls)]
                        classes.append(class_name)  # Store class names

                        # Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]
                        x1, y1, x2, y2 = map(float, bbox.tolist())
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1

                        detections.append([[x, y, w, h], conf.item(), class_name])  # Append in the required format
                
                logging.info(f"Detections array: {detections}")

                # Update DeepSORT tracker
                start_time = time.time()
                #tracked_objects = tracker.update_tracks(detections, frame=frame)

                future = executor.submit(update_tracks_async, detections, frame)
                tracked_objects = future.result()

                elapsed_time = (time.time() - start_time)*1000
                print(f"Time taken to update tracks: {elapsed_time:.2f} ms")

                # Annotate and log tracked objects
                for track in tracked_objects:
                    if not track.is_confirmed():  # Skip unconfirmed tracks
                        continue

                    track_id = track.track_id
                    class_name = track.get_det_class()  # Retrieve class name directly from the detection
                    x1, y1, x2, y2 = map(int, track.to_tlbr())  # Get track bounding box (top-left to bottom-right)

                    # Draw bounding box
                    color = (0, 255, 0)  # Green for tracked objects
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Add label with track ID and class
                    label = f"{class_name} (ID: {track_id})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    current_time = time.time()
                    if track_id not in seen_track_ids or current_time - seen_track_ids[track_id] > EXPIRY_TIME:
                        is_new_detection = True
                        emit_event(track_id, class_name)  # Emit event for new or expired track
                    
                    seen_track_ids[track_id] = current_time  # Update last seen timestamp

                # Update the latest frame file
                cv2.imwrite(latest_frame_file, frame)
                logging.info(f"Updated latest frame to {latest_frame_file}")

                # if is_new_detection is false
                #if is_new_detection == False:
                    # Save the high-resolution frame for inference but resize it before saving
                    #resized_frame = resize_frame(frame, target_width=640)

                    # Save the frame with a unique timestamp
                    #frame_file = os.path.join(output_dir, f"{timestamp}.jpg")
                    #cv2.imwrite(frame_file, resized_frame)
                   # logging.info(f"Saved frame to {frame_file}")
               # else:
                if is_new_detection == True:
                    # Save the frame with a unique timestamp
                    frame_file = os.path.join(output_dir, f"{timestamp}-NEW_DETECTION.jpg")
                    cv2.imwrite(frame_file, frame)
                    logging.info(f"Saved new detection frame to {frame_file}")

            


    except KeyboardInterrupt:
        logging.info("Stream processing stopped by user.")
    finally:
        cap.release()
        logging.info("RTSP stream closed.")

if __name__ == "__main__":
    # Replace with your camera's RTSP URL
    #rtsp_url = "rtsps://10.1.1.134:7441/3W12O36llGYVR68L?enableSrtp"
    rtsp_url = "rtsps://10.1.1.134:7441/yRVYjAD4IBbBr9kp?enableSrtp"
    #rtsp_url = "rtsps://10.1.1.134:7441/eXnZyAaCSt4E6RGq?enableSrtp"

    # Print log messages to the console
    logging.info(f"Opening RTSP stream: {rtsp_url}")

    process_rtsp_stream(rtsp_url)