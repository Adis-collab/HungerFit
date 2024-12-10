import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, capture_index):
        # Initialize the capture index for the webcam/video feed and check device
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        # Load the YOLO model
        self.model = self.load_model()

    def load_model(self):
        # Load the YOLO model with custom weights
        model = YOLO("/Users/adi/Desktop/HungerFit/datasets/runs/detect/train2/weights/best.pt")
        model.fuse()  # Fuse layers to optimize inference
        return model

    def predict(self, frame):
        # Perform prediction using the YOLO model
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        # Extract bounding box information and plot it on the frame
        xyxys = []
        confidences = []
        class_ids = []
        class_thali = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs


            # Draw bounding boxes on the frame
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if class_id==0:
                    class_thali.append("Chapati")
                elif class_id==1:
                    class_thali.append("Rice")

                x1, y1, x2, y2 = map(int, box)
                color = (255, 0, 0)  # Set bounding box color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{class_thali[0]} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                class_thali.clear()

        return frame

    def __call__(self):
        # Open the webcam/video feed
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), f"Failed to open capture device {self.capture_index}"

        # Frame rate calculation
        start_time = time()

        while True:
            # Capture frame-by-frame from the video feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize frame for better processing speed (optional)
            frame_resized = cv2.resize(frame, (640, 480))

            # Perform object detection on the frame
            results = self.predict(frame_resized)

            # Draw bounding boxes on the frame
            frame_with_bboxes = self.plot_bboxes(results, frame_resized)

            # Calculate FPS (Frames Per Second)
            end_time = time()
            fps = 1 / (end_time - start_time)
            start_time = end_time
            cv2.putText(frame_with_bboxes, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Display the frame with bounding boxes
            cv2.imshow('Object Detection', frame_with_bboxes)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()


# Run the object detection
if __name__ == "__main__":
    detector = ObjectDetection(capture_index=0)  # Change capture_index to 0 for webcam or path for a video file
    detector()
