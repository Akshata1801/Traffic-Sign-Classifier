# Traffic-Sign-Classifier
### Overview
In this project, I used deep neural networks and three classic convolutional neural network architectures(LeNet, AlexNet and GoogLeNet) to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out my model on images of German traffic signs that I find on the web.

### The goals / steps of this project are the following:
* Load and explore the data set.
* Realize LeNet architecture and use `ReLu`, `mini-batch gradient descent` and `dropout`. 
* Realize AlexNet and make some modifications, use `learning rate decay`, `Adam optimization` and `L2 regulization`. 
* Use GoogLeNet to classify traffic signs and make some modifications, use `inception` and `overlapping pooling` and `average pooling`. 
* Analyze the softmax probabilities of the new images
* Summarize the results

### Dependencies
python3.5  
matplotlib (2.1.1)  
opencv-python (3.3.1.11)  
numpy (1.13.3)  
tensorflow-gpu (1.4.1)  
sklearn (0.19.1)  

### Dataset
Download the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which the images are already resized to 32x32. It contains a training, validation and test set.

[Data pre-process.ipynb](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/Data%20pre-process.ipynb)
---

I used the numpy library to calculate summary statistics of the traffic signs data set:
* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32 ,3)
* The number of unique classes/labels in the data set is: 43

[GoogLeNet.ipynb]
---
[GoogLeNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) was the winner of the ILSVRC 2014, it main contribution was the development of `Inception Module` that dramatically reduced the number of parameters in the network.   
![alt text][inception]  
Additionally, this paper uses `Average Pooling` instead of `Fully connected layer` at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much. The overall architecture of GoogLeNet is as the following table.  

![alt text][googlenet]  

The original architecture of GoogLeNet is a little hard to train by my GPU, so I choose to reduce the number of layers from 22 to 14, the details of network is showing in the following table.

| Type          | Kernel/Stride	| Output    | Parameters  |
|:-------------:|:-------------:|:---------:|:-----------:|
| conv          | 3x3/2x2       | 16x16x64  | 1,792       |
| inception(2a) |               | 16x16x256 | 137,072     |
| inception(2b)	|               | 16x16x480 | 388,736     |
| max pool    	| 3x3/2x2      	| 7x7x480   |             |
| inception(3a) |  	            | 7x7x512   | 433,792     |
| inception(3a) |  	            | 7x7x512   | 449,160     |
| max pool 	    | 3x3/2x2  	    | 3x3x512   |             |
| inception(4a) |  	            | 3x3x832   | 859,136     |
| inception(4a) |  	            | 3x3x1024  | 1,444,080   |
| avg pool 	    | 3x3/1x1  	    | 1x1x1024  |             |
| flatten	    | 864			| 1024      |             |
| full		    | 43            | 43        | 44,032      |

Some details for this architecture is as following:
- Inception Module  
The inception module is the core of this architecture, it is driven by two disadvantage of previous architecture: a large amount of parameters which lead to overfitting and dramatically use of computational resources. It's navie implement doesn't have 1x1 conv before/after 3x3 conv, 5x5 conv and max pooling layer. The reason why adding 1x1 convolutional layer is that it can reduce the depth of the output from previous layer, therefore, the amount of operations can be significantly reduced. More details can be found in [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf). Since max pooling will reduce the shape of input feature map, so I realize it by padding with zeros and another implement can look [here](https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/).
- Overlapping pooling  
The normal pooling operation is with kernel size = 2 and stride = 2, and the overlapping pooling means kernel size > stride, like kernel size = 3 and stride = 2, thus there will be overlapping fields. According to [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://219.216.82.193/cache/13/03/yann.lecun.com/a46bf8e4b17c2a9e46a2a899a68a0a0d/sermanet-ijcnn-11.pdf), overlapping pooling can slightly reduce the error rates compared to non-overlapping and make the model more difficult to overfit. 

### Training 
I have turned the following three hyperparameters to train my model.
* LEARNING_RATE = 2e-4
* EPOCHS = 100
* BATCH_SIZE = 64
* keep_prop = 0.5

The results are:
* accuracy of training set: 100.0%
* accuracy of validation set: 98.5%
* accuracy of test set: 98.1% 

Summary
---
In this project, I use three classific CNN architecture to recognize traffic signs from GTSRB, they are LeNet, AlexNet and GoogLeNet. Since the original architecture may no be suit for images from GRSRB, so I made some changes to them. In addition, I use some methods and tricks to train the model, like mini-batch gradient descent, Adam optimization, L2 regularization, learning rate decay and so on. 
