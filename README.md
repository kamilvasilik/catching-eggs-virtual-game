# catching-eggs-virtual-game
openCV, python 

### Idea
object detection (in our case 'banana') in real time using webcam 

### Versions
catching_eggs_01.py 
  - object detection using YOLO method 
  - need to download model configuration file and weights from https://pjreddie.com/darknet/yolo/ \
    (yolov3-320.cfg, yolov3-320.weights or yolov3-tiny.cfg, yolov3-tiny.cfg)
  
catching_eggs_02.py 
 - object detection using custom cascade classifier
 - cascade classifier has been trained using Cascade-Trainer-GUI https://amin-ahmadi.com/cascade-trainer-gui/
 
### Files
catching_eggs_library.py 
  - functions used in main files

three versions of haarcascade classifier: \
haarcascade / haarcascade_banana_01.xml \
haarcascade / haarcascade_banana_02.xml \
haarcascade / haarcascade_banana_03.xml 
