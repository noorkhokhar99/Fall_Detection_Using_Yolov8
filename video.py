from ultralytics import YOLO
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# model = YOLO("./runs/detect/train/weights/best.pt")
model = YOLO("fall_det_1.pt")

video_path = "production_id_5039468 (2160p).mp4";
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,conf=0.5)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# model.track("https://www.youtube.com/watch?v=GTCCMG8zUlU",show=True,tracker="bytetrack.yaml")