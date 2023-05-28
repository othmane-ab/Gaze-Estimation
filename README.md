# Gaze-Estimation
A project to estimate the gaze of person in the image


This script performs face detection, landmarks detection, head pose estimation, and gaze estimation. The models used are pre-trained models provided by Intel and can be downloaded from their Open Model Zoo.

Please replace 'image.jpg' with the path to your image. Also, replace the model names with the path to your models if they're not in the same directory as this script. The path to the models are assumed to be in the same directory as the Python script in this example.

This script only considers the first face detected in the image for simplicity. If there are multiple faces in the image and you want to estimate the gaze for all faces, you will have to modify the code to loop over all detected faces.

Please note that you may need to adjust the blob sizes based on your models. Make sure to replace 'face-detection-adas-0001.xml', 'landmarks-regression-retail-0009.xml', 'head-pose-estimation-adas-0001.xml', and `'gaze-estimation