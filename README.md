# MTCNN_face_detection_alignment
Tensorflow version of "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks"
## Requirement
1. Python 3.5
2. Tensorflow_1.10.0 or later
3. Cuda (if use nvidia gpu)
## Training Data
pnet: 400W 
pos : part : neg = 1 : 1 : 3
space: 3G

rnet: 2000W
pos : part : neg = 1 : 1 : 3
space: 50G

onet: 1600W
pos : part : neg : landmark = 1 : 0.3 : 2 : 2
space: 144G
## Results
![](https://github.com/daikankan/MTCNN_face_detection_alignment/blob/master/results/lfw.jpg) 

![](https://github.com/daikankan/MTCNN_face_detection_alignment/blob/master/results/widerface.jpg) 

### FDDB
![](https://github.com/daikankan/MTCNN_face_detection_alignment/blob/master/results/discROC.png) 

![](https://github.com/daikankan/MTCNN_face_detection_alignment/blob/master/results/discROC-pnet.png) 
## Matlab implementation
[kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
## License
This code is distributed under MIT LICENSE
## Contact
daikank@163.com
