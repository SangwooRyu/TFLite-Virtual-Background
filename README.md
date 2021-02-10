# TFLite-Virtual-Background


Virtual Background for Web.

Currently only making Segmentation Mask using C++ works.


Some Code from
https://github.com/Qengineering/TensorFlow_Lite_Segmentation_RPi_32-bit


## Reference
https://ai.googleblog.com/2020/10/background-features-in-google-meet.html

https://github.com/floe/deepbacksub


## Environment
 - Ubuntu 20.04.1 LTS (GNU/Linux 5.4.72-microsoft-standard-WSL2 x86_64)
 

## Requirement
You should install..
 - Tensorflow Lite
    - You should build Tensorflow Lite for your machine and place library inside TFLite-Virtual-Background/lib.
    - You should copy [Tensorflow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite "Tensorflow Github") to TFLite-Virtual-Background
 - Flatbuffers
    - You should build [Flatbuffers](https://github.com/google/flatbuffers "Flatbuffers Github") and copy it to TFLite-Virtual-Background 
 - OpenCV
    - You should install OpenCV on your machine now (plan to remove)
