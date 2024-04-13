import cv2
import torch
import numpy as np


path='C:/Users/KIIT/Downloads/bone fracture.v2-release.yolov5pytorch/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)

cap=cv2.VideoCapture('bone_fracture.mp4')
count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,600))
    results=model(frame)
    frame=np.squeeze(results.render())
    results=model(frame)
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
'''import cv2
import glob
import torch
import numpy as np

# Load the YOLOv5 model
path = 'C:/Users/KIIT/Downloads/bone fracture.v2-release.yolov5pytorch/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

# Function to perform object detection on an image
def detect_objects(image):
    # Resize the image if needed
    image = cv2.resize(image, (1020, 600))
    # Perform object detection
    results = model(image)
    # Render the results
    rendered_image = np.squeeze(results.render())
    return rendered_image

# Process single image or a series of images
def process_images(image_paths):
    for image_path in image_paths:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image at path: {C:/Users/KIIT/Downloads/bone fracture.v2-release.yolov5pytorch}")
            continue
        # Perform object detection
        detected_image = detect_objects(frame)
        # Display the detected image
        cv2.imshow("Detected Objects", detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Directory containing images
image_directory = 'C:/Users/KIIT/Downloads/bone fracture.v2-release.yolov5pytorch/'

# Get a list of all image files in the directory
image_paths = glob.glob(image_directory + '*.jpg')  

# Process the images
process_images(image_paths)'''


