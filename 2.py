

from ultralytics import YOLO
import cv2
import math
import numpy as np 


# Load YOLO model   
model = YOLO("helmet.pt")

# results = model("traffic-2.mp4", save=True, show=True)  


# Load video (one for detection, one for clean frames)
cap_detect = cv2.VideoCapture("traffic-2.mp4")
cap_original = cv2.VideoCapture("traffic-2.mp4")
frame_width, frame_height = int(cap_detect.get(3)), int(cap_detect.get(4))

# Output video writers
out = cv2.VideoWriter("traffic_helmet_detected.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      int(cap_detect.get(cv2.CAP_PROP_FPS)), 
                      (frame_width, frame_height))

out_no_helmet = cv2.VideoWriter("no_helmet_detections.mp4", 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                int(cap_detect.get(cv2.CAP_PROP_FPS)), 
                                (800, 800))  # Adjust frame size for dynamic grid layout

# Define colors for helmet and no-helmet
helmet_color = (0, 255, 0)   # Green for helmet
no_helmet_color = (0, 0, 255)   # Red for no helmet

# Sliding window parameters
step_size_divisor = 2
window_size_divisor = 2
window_size = (frame_width // window_size_divisor, frame_height // window_size_divisor)
step_size = (frame_width // step_size_divisor, frame_height // step_size_divisor)

# Set fixed size for each crop
fixed_crop_size = (100, 100)  # Width, Height for each cropped image


while cap_detect.isOpened() and cap_original.isOpened():
    ret_detect, frame_detect = cap_detect.read()
    ret_original, frame_original = cap_original.read()
    
    if not ret_detect or not ret_original:
        break

    height, width = frame_detect.shape[:2]
    no_helmet_crops = []  # Store cropped images of "no helmet" detections

    # Slide window over the frame
    for y in range(0, height, step_size[1]):
        for x in range(0, width, step_size[0]):
            x_end = min(x + window_size[0], width)
            y_end = min(y + window_size[1], height)
            sub_frame = frame_detect[y:y_end, x:x_end]

            # Run YOLO model on each sub-frame
            results = model(sub_frame)

            # Process detection boxes, labels, and confidence scores
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                labels = r.boxes.cls.cpu().numpy()  # Class labels
                confidences = r.boxes.conf.cpu().numpy()  # Confidence scores for each detection

                for box, label, confidence in zip(boxes, labels, confidences):
                    # Set a confidence threshold to filter detections
                    if confidence >= 0:
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1 + x), int(y1 + y), int(x2 + x), int(y2 + y)

                        if label == 0:  # Helmet
                            color = helmet_color
                            text = f"Helmet ({confidence:.2f})"  
                        else:  # No helmet
                            color = no_helmet_color
                            text = f"No Helmet ({confidence:.2f})"  
                            
                            # Only add to no-helmet crops if confidence is above the threshold
                            if confidence >= 0.4:
                                # Crop and resize no-helmet detection from the original frame (no bounding boxes)
                                no_helmet_crop = frame_original[y1:y2, x1:x2]
                                no_helmet_crop_resized = cv2.resize(no_helmet_crop, fixed_crop_size)
                                no_helmet_crops.append(no_helmet_crop_resized)
                        
                        # Draw bounding box and label with confidence on the detected frame (frame_detect)
                        cv2.rectangle(frame_detect, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_detect, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame with helmet detection
    cv2.imshow("Helmet Detection", frame_detect)
    out.write(frame_detect)

    # Display and save the cropped "no helmet" detections if any
    if no_helmet_crops:
        # Calculate grid size based on the number of crops
        num_crops = len(no_helmet_crops)
        grid_size = math.ceil(math.sqrt(num_crops))  # To create a square-like grid

        # Prepare a blank canvas for the grid
        grid_frame_size = (grid_size * fixed_crop_size[0], grid_size * fixed_crop_size[1])
        grid_frame = np.zeros((grid_frame_size[1], grid_frame_size[0], 3), dtype=np.uint8)

        # Place each crop in the grid
        for idx, crop in enumerate(no_helmet_crops):
            row = idx // grid_size
            col = idx % grid_size
            y1, y2 = row * fixed_crop_size[1], (row + 1) * fixed_crop_size[1]
            x1, x2 = col * fixed_crop_size[0], (col + 1) * fixed_crop_size[0]
            grid_frame[y1:y2, x1:x2] = crop

        # Display and save the grid of no-helmet detections
        cv2.imshow("No Helmet Detections", grid_frame)
        out_no_helmet.write(grid_frame)  # Save to the "no helmet" video file

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_detect.release()
cap_original.release()
out.release()
out_no_helmet.release()
cv2.destroyAllWindows()