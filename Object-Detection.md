# HotWheels - Object Detection 🚏 🚶‍♂️
## Introduction 📘
Apart from the Lane Detection the second major module in the program was the Object Detection because being able to correctly identify obstacles and avoid danger is a critical feature of any self-driving vehicle.  
  
We were required to develop a model capable of detecting objects and the street signs of the track in the lab and for that we used the *YOLO* algorithm, a very famous object detection tool, and trained it with images from the *COCO* dataset.  
  
### YOLO
YOLO (You Only Look Once) is a real-time object detection algorithm that identifies and localizes multiple objects in an image with a single forward pass of a neural network.  
  
Unlike traditional methods that use region proposals followed by classification, YOLO treats detection as a regression problem, directly predicting bounding boxes and class probabilities from the entire image. Thanks to that it's widely used in applications like surveillance, autonomous driving, and robotics due to it's balance of accuracy and performance.  
  
***Mean Average Prediction (mAP)*** - metric used to evaluate object detection models based on sub metrics such as Confusion Matrix, Intersection over Union (IoU), Recall, and Precision.  
  
***YOLO has a mAP of 57.9% on the COCO dataset, higher than any other model, 76.5% accuracy in traffic management systems (top 1) and 93.3% accuracy in vehicle detection (top 5).***  

#### [Key features]
- **Speed:** Extremely fast, suitable for real-time applications.  
  
- **End-to-end architecture:** A single neural network handles detection, making it efficient.  
  
- **Versions:** Multiple versions exist (e.g., YOLOv3, YOLOv4, YOLOv5, YOLOv8), each improving speed, accuracy, and usability.  
  
YOLO synergizes well with the COCO dataset and it's common to see YOLO models trained with it.  
    
### COCO dataset 
COCO (Common Objects in Context) is a large scale dataset created to train and evaluate object detection algorithms. It has more detail per image than any other dataset and supports many tasks such as Detection, Segmentation, Keypoints, Captioning, etc and uses the PyTorch framework.  
  
#### Content
- 330k images  
- 200k labeled images  
- 1.5M object instances
- 80 object types (car, dog, person, etc)
- segmentaion masks for each object  
  
#### Division
- Train2017 (118k images)
- Val2017 (5k images)
- Test2017 (unlabeled images)

## Development 🛠️
In the frist train that we did we used all of the images and classes that the COCO dataset provided, being able to detect all of the different object types. However, in our case, there was no need to detect that many classes, so we shrunk that number to fit only the necessary ones that were present in the lab track.  
  
Having completed the training with this new filtered dataset, we tested it with real images from the lab and the resuls were not satisfactory. Even though the model was capable of identifying objects in the images from the COCO dataset, it was not performing well at all with images from the track. This problem occured because the characteristics of the camera in the JetRacer were very different from the camera used in the dataset (angle, fov, color pigmentation. etc) making it difficult for the model to make it's predictions. On top of that, the track also had details that the dataset did not cover such as different collored lanes.  
  
In order to solve this problem we were forced to create our own custom dataset using only images taken from our car in the lab track to guarantee accurate predictions. For this we used **Roboflow**, an online platform that allows you to import images and classify them with your custom classes and providing pre-processing/post-processing tools to improve the quality of your dataset. This process was very repetitive and exhausting but caused a huge improvement in our results proving to be worth the time and dedication.  
  
**[IMPORTANT COMMANDS]**  
Below are some useful commands, to develop your own object detection model, that are executed inside the YOLO folder.  
  
Train the model:  

	python train.py --img 640 --batch 16 --epochs 50 --data ./datasets/final_data.yaml --weights best.pt


Test the model with a video:  

	python detect.py --weights ./runs/train/exp30/weights/best.pt --source ./data/videos/lab_signs.mp4


Export to ONNX:  

	python export.py --weights ./runs/train/exp30/weights/best.pt --include onnx --opset 12 --simplify --img 640 --batch 1 --device 0
  
## Results 📊
After several adjustments the final model consisted of only 10 classes, each one representing a specific street sign in the track, and in the video below we can see them being accurately identified.

https://github.com/user-attachments/assets/4556ca30-7604-4370-a6ee-9804e0810a38

___