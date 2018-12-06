---
title: "CPE 646 Final project"
author: "Zhenyue Liu"
date: "November 27, 2018"
output: html_document
---

# 20-Newsgroup Classification using KNN, MLP and CNN
## Introduction
<font size="3">In this project, I am going to classify the 20-Newsgroup dataset which contains 18,000 rows of news files into 7 large groups. To implemente this goal, I will use three different classfier model: K-nearest neighbor, Multilayer Perceptron and Convolution Neural Network, and compare them to each other.   
This project is helping me to understand how these three models work, how to optimize result and what are their advantages and disadvantages.   
 
The model K-nearest neighbor is representative in simple classifiers, and two neural network Multilayer Perceptron and Convolution Neural Network are also widely used in Natural Langauge Processing. And I will use accuray and consuming time as two parameters to evaluate these models.  

## Data Sourse and reference
### Data source  
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 small newsgroups, and 7large groups.  
> http://qwone.com/~jason/20Newsgroups/   
<br>
Sample:    

![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/sample1.png)   

### GloVe Corpora:     
Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 200d)  
>https://nlp.stanford.edu/projects/glove/  

### Reference:   
Google Guide of text classification   
> https://developers.google.com/machine-learning/guides/text-classification/

### Complete document 
Complete document could be found on my github:   
> https://github.com/lzysh9506/Pattern_Recog_and_Classfic  


## Process
### 1. Load data:   
The original data is a *.tar.gz file, so I directly read all news data from this file, and use the first part name of folder to be their labels.   
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/targz_doc.png)  

### 2. Clean Data:  
In case that some useful texts are cleaned, I just did some simple steps to clean data:   
- complete all shorthand (like it's -> it is, Tom'll -> Tom will)  
- lower case all words  
- replace all punctuations and digits with white space  
The clean data sample looks like:   
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/sample2.png)   


### 3. Split data
I shuffle and split original 18,828 rows of data and their labels into training set, cross validation set and test set, the proportion is 0.6:0.2:0.2. 
But since the KNN model doesn't need cross validaton set to avoid overfitting, so I just used training set and test set in this model.
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/Split_data.png)  

### 4. Vectorize data
The KNN and MLP model used n-grams model, and I chose 2000 features with highest f_classif score using tf-idf encoding.  
For CNN model, I used word embedding to vectorize data as sequence, and chose 20,000 most frequent features.  
<br>
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/vectorize_data.png)   
<br>
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/vectorize_data_cnn.png)   

### 5. K-nearest Neighbor
In KNN model, I chose three K values 11, 21, 31 to see if they have difference in result. When implementing the KNN algorithm, I used the training set to build a KDTree structure, and query each test sample to find k nearest neighbor points, it could be much faster than calculating the distance from test sample to each point in training set. 

### 6. Multilayer Perceptron
First I set the last layer of the model. The unit in this layer is the same as the number of classes, and the values from all units sum to be 1. And I used `softmax` as activation function.  
Then I defined a two-layer MLP model, used `relu` activation function in each layer and added dropout layers for regularization to avoid overfitting.   
When training the model, the argument is as following:  
`learning rate = 1e-3`  
`epoch = 100`  
`batch size = 128`  
`dropout_rate = 0.2`  
`loss_function = 'sparse_categorical_crossentropy'`  
`optimizer = 'Adam'`  
After that, the cross validation set is used to validate the model.

### 7. Convolution Neural Network
The last layer is the same as in MLP model. The first layer in CNN models is an embedding layer, which learns the relationship between the words in a dense vector space.  
In the model, there are four convolution layers, four max pooling layers and one full connection layer at last.   
I also used `relu` as activation function and zero padding to fill the layer.
When training the model, the argument is as following:  
`learning rate = 1e-3`  
`epoch = 100`  
`batch size = 128`  
`blocks = 2`  
`filter = 64`  
`dropout_rate = 0.2`  
`embedding_dim = 200`
`kernel_size = 3`  
After that, the cross validation set is used to validate the model.

### 8. Embedding pre-trained corpora to fine-tuned CNN
Words in a given dataset are most likely not unique to that dataset. So the CNN model could learn relationship between the words in my dataset using other existed dataset. Using a pre-trained embedding gives the model a head start in the learning process.  
In this project, I used GloVe corpora which contains 400,000 vocabulary from Wikipedia 2014 + Gigaword 5 to pre-trained CNN model. The consuming time must increase, but as well as the accuracy.  
I do this in two stages:  
1. In the first run, with the embedding layer weights frozen, allow the rest of the network to learn. At the end of this run, the model weights reach a state that is much better than their uninitialized values.   
2. For the second run, allow the embedding layer to also learn, making fine adjustments to all weights in the network.

## Result
### KNN
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/knn_result.png) 
<br>  

### MLP
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/mlp_result.png)   
<br>
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/acc_mlp.png)   
Cumulative consuming of MLP: 192 seconds   

### CNN without pre-train
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/cnn_result.png)
<br>
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/acc_cnn.png)  

Cumulative consuming of CNN: 1609 seconds   

### CNN with GloVe pre-trained
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/pre_trained_cnn_result.png)
<br>
![Image 1](https://github.com/lzysh9506/Pattern_Recog_and_Classfic/raw/master/Screenshoots/acc_pre_trained_cnn.png)  
Cumulative consuming of pre-trained CNN: 2013 seconds 

## Discussion

|                |K-nearest Neighbor|Multilayer Perceptron|Convolution Neural Network|Pre-trained Convolution Neural Network|
|:---------------|-----------------:|--------------------:|-------------------------:|-------------------------------------:|
|Accuracy        |0.6567(k=11)      |0.9334               |0.7343                    |0.9095                                |
|Consuming time  |126               |192                  |1609                      |2013                                  |  

Based on the table above, we could see that MLP has the highest accuray with also very low consuming time. It is the best model without any doubt.  
When using KNN, we found that the result decreases when the k goes higher. I guess the main reason might be the training set is in quite great clustering. When counting more samples to do the vote, there's high possibility that include samples from other groups, resulting the accuracy to decrease.  
In contrast to KNN, the CNN must be a much more complicated model, but the performance is not good enough. Though the pre-trained CNN also has a high accuracy, it costs ten times time compared to MLP. The reason is that it need first train the model with corpora, and actually we could see that pre-trained CNN only experience four epoch on training set before converging. 

</font>
