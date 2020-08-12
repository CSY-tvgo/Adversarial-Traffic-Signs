# Adversarial-Traffic-Signs  

Supplementary Material: Click this [Dropbox link](https://www.dropbox.com/s/folonu3j8ss01yt/Supplements-of-Repo-Adversarial-Traffic-Signs.zip?dl=0) to download. It contains 3 files and you can place them into corresponding directories to run the code.  

## Contents  
+ `1.CTSRD_Classification`: Train a Chinese traffic sign classification model using [CTSRD Dataset](http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html). It uses ResNet101 in Keras.  
+ `2.DARTS-Attack-on-CTSRD`: Modified [DARTS pipeline](https://github.com/inspire-group/advml-traffic-sign), it can generate adversarial Chinese traffic signs.  
+ `3.TT100K_ObjectDetection`: Using an object detection model pre-trained by [TT100K Dataset](https://cg.cs.tsinghua.edu.cn/traffic-sign/) to do test of adversarial traffic signs in real world. It uses Faster R-CNN-ResNet101 in TensorFlow Object Detection API.  
+ `4.Examples of Adversarial Traffic Signs`: Some examples.  

## Acknowledgement  
This project is what I do in my thesis of B.Eng. in Electronic and Information Engineering.  
I appreciate support from VeriMake, issues are also welcome on [VeriMake BBS](https://verimake.com/) *(in Chinese)*.  

## Reference  
+ [Chinese Traffic Sign Database](http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html)  
+ [Keras API Reference](https://keras.io/api/)  
+ [DARTS: Deceiving Autonomous Cars with Toxic Signs](https://github.com/inspire-group/advml-traffic-sign)  
+ [TensorFlow Guide](https://tensorflow.google.cn/guide)  
+ [Tsinghua-Tencent 100K Dataset](https://cg.cs.tsinghua.edu.cn/traffic-sign/) 
+ [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
