# YOLO
## Introduction
YOLO (You Only Look Once) is a real-time object detection algorithm that identifies and localizes multiple objects in an image with a single forward pass of a neural network. Unlike traditional methods that use region proposals followed by classification, YOLO treats detection as a regression problem, directly predicting bounding boxes and class probabilities from the entire image. It is widely used in applications like surveillance, autonomous driving, and robotics due to its balance of accuracy and performance.  
  
**76.5% accuracy in traffic management systems (top 1) and 93.3% accuracy in vehicle detection (top 5).**  
  
***Mean Average Prediction (mAP)*** - metric used to evaluate object detection models based on sub metrics such as Confusion Matrix, Intersection over Union (IoU), Recall, and Precision. **YOLO has a mAP of 57.9% on the COCO dataset, higher than any other model.**

### Key features
**Speed:** Extremely fast, suitable for real-time applications.  
  
**End-to-end architecture:** A single neural network handles detection, making it efficient.  
  
**Versions:** Multiple versions exist (e.g., YOLOv3, YOLOv4, YOLOv5, YOLOv8), each improving speed, accuracy, and usability.  
  
## COCO dataset
YOLO synergizes well with the COCO dataset and it's common to see YOLO models trained with it.  
  
COCO (Common Objects in Context) is a large scale dataset created to train and evaluate object detection algorithms. It has more detail per image than any other dataset and supports many tasks such as Detection, Segmentation, Keypoints and Captioning. It uses the PyTorch framework.  
  
### Content
- 330k images  
- 200k labeled images  
- 1.5M object instances
- 80 object types (car, dog, person, etc)
- segmentaion masks for each object  
  
### Division
- Train2017 (118k images)
- Val2017 (5k images)
- Test2017 (unlabeled images)

___