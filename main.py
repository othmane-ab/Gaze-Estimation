import cv2
import numpy as np

# Load all the models
face_net = cv2.dnn.readNet('face-detection-adas-0001.xml', 'face-detection-adas-0001.bin')
landmarks_net = cv2.dnn.readNet('landmarks-regression-retail-0009.xml', 'landmarks-regression-retail-0009.bin')
head_pose_net = cv2.dnn.readNet('head-pose-estimation-adas-0001.xml', 'head-pose-estimation-adas-0001.bin')
gaze_net = cv2.dnn.readNet('gaze-estimation-adas-0002.xml', 'gaze-estimation-adas-0002.bin')

# Load image and get its height and width
image = cv2.imread('image.jpg')
height, width = image.shape[:2]

# Prepare the image for inference by converting its color (BGR->RGB)
blob = cv2.dnn.blobFromImage(image, size=(672, 384))

# Perform face detection
face_net.setInput(blob)
out = face_net.forward()

# Get face detected
face = out.reshape(-1, 7)
for detection in face:
    confidence = float(detection[2])
    xmin = int(detection[3] * width)
    ymin = int(detection[4] * height)
    xmax = int(detection[5] * width)
    ymax = int(detection[6] * height)

    if confidence > 0.5:
        face_image = image[ymin:ymax, xmin:xmax]

# Perform landmarks detection
face_blob = cv2.dnn.blobFromImage(face_image, size=(48, 48))
landmarks_net.setInput(face_blob)
landmarks = landmarks_net.forward()

# Perform head pose estimation
head_pose_blob = cv2.dnn.blobFromImage(face_image, size=(60, 60))
head_pose_net.setInput(head_pose_blob)
yaw, pitch, roll = head_pose_net.forward()

# Prepare input for gaze estimation model
eyes_blob = cv2.dnn.blobFromImage(face_image, size=(60, 60))
gaze_net.setInput({'left_eye_image': eyes_blob, 'right_eye_image': eyes_blob, 'head_pose_angles': np.array([yaw, pitch, roll])})
gaze_vector = gaze_net.forward()

# Gaze vector gives the direction of gaze
print("Gaze Vector: ", gaze_vector)
