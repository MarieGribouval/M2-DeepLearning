# Deep Bee

Deep Learning Project of Master 2 of Data Science
\[Project Report\] | [Article Summary](./RESUME.md)

Marie Gribouval, Léo Boule and Leshanshui YANG from University of Rouen


## 📑 Project Introduction
Since the survival of bees is important to the environment of the planet, it is important to find a non-destructive solution to monitor the health of bees. To this end, we studied the article "The Application of Convolutional Neural Network for Pollen Bearing Bee Classification" in this project.

The article is concerned about whether or not bees carry pollen. For our project, we are more interested in the protection of bees by **predicting the state of health of the bee**, especially if it carries mites (varroas).

Varroa mites are parasites that settle on the bee's body which would weaken it. In some regions of the world it is the leading cause of bee mortality.
When we found our database, other information about each image was given (like the subspecies for example). So we decided to classify this information as well to test our neuron network (by **applying our CNN to different classification tasks**.)

We used a [data set from kaggle by jenny18](https://www.kaggle.com/jenny18/honey-bee-annotated-images) which fits our theme very well. We achieved the classification of bee species and health through a network with less than 10K parameters. We tested the model mentioned in the article and proposed some changes and experiment.

(For more details of the article, please find our summary of the article (Résumé de l'article) in [RESUME.md](./RESUME.md).


## 🚩 Targets
The targets of this project is to:

+ [x] implement the **data pre-processing and model building** for reproducing the experiment

+ [x] test the **performance of RGB and HSV inputs** under very few layers of network

+ [x] compare **the speed of CPU with Jetson GPU**

+ [x] **record all misclassified data** to facilitate model improvement


## 📊 Discussion of Results


With very few effective iterations, ***we obtained the results very close to the article with classification accuracy rates (90.89%) in terms of subspecies and (91.71%) in terms of health.***

### Data Set

  The [original dataset](https://www.kaggle.com/jenny18/honey-bee-annotated-images) we used consists of 4744 rows of data, where **X is a photo of the bee** and **Y is the label** of the photo (about the region, health level, honey carried, subpecies, etc.).
  
  In our experiments, we used 4098 pieces of data smaller than 100 * 100 pixels. We selected **health and subpecies** as the Y to predict(classify) in 2 independent models.
  
  We used keras' **stratified data splitting** function to ensure that the label proportion (distribution) of the training set and the test set is basically the same.
    
    | Type          | Data          |
    | ------------- | ------------- |
    | Total         | 4098          |
    | Train         | 2582 (63%)    |
    | Valid         |  287 (07%)    |
    | Test          | 1220 (30%)    |


### Model and Experimental Parameters

The model we used (and proposed by the article) is shown in the figure:
![](https://github.com/TilkeyYANG/M2-DeepLearning/raw/master/imgs/model.jpg)


After testing various parameters in the experiment, we selected the following parameters and training trick:

```python
	#...
    	batch_size = 8
    	epoch = 100
	#...
    	checkpoint = ModelCheckpoint('models\\model_%s_%s.h5'%(Xname, ycol[:4]), monitor='val_loss', 
								 save_best_only=True, mode='auto')  
    	earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
	#...
```


### Experimental Results
During training, we obtained some curves as shown in the figure:

![](https://github.com/TilkeyYANG/M2-DeepLearning/raw/master/imgs/rgb2layers.jpg)

It is worth noting that since we set Early Stopping of 20 Epochs and only save the optimal model (Loss of Validation is the smallest). In that case, **we actually used the model stored in the 6th Epoch during the test phase, instead of the model after "overfitting".**

With the same method (**checkpoint + earlystopping**), we obtained the following results:

![](https://github.com/TilkeyYANG/M2-DeepLearning/raw/master/imgs/accuracy.jpg)

> *`RGB-enh` means we adjusted the saturation and brightness of the image in order to enhance the image quality.*

> *`3 layers` of HSV means we added another convolution layer, which is another trick mentionned in the article.*

### Jetson Nano GPU vs i7 CPU

If we compare the processing time on a i7-8750 CPU and on the Jetson GPU, we have these results:

![](https://github.com/TilkeyYANG/M2-DeepLearning/raw/master/imgs/Jetson_results.png)
![](https://github.com/TilkeyYANG/M2-DeepLearning/raw/master/imgs/CPU_results.png)

We can see that the speed of the CPU is higher for a 2 layers model (RGB), but is lower if it is a 3 layers model(HSV). It is probably because the model HSV is more complex, so the GPU will be the best solution in this case.
Furthermore, the RGB prediction model is faster than HSV one, because it is a much simple model.

Moreover, if we compare the fps for 20 and 200 images prediction (with scalling the 200 images one on the 20 images one), we can see that it is taking much time. It is probably because the Jetson has a lack of memory with a lot of images to predict, so it makes the prediction running slowly. However, it seems to be the same thing for the CPU.


With the Jetson results, it proves that it would be possible if we want to use this program and the jetson as an embedded application. Processing time is enough fast to analyze if bees have any problem in real time.

## ⌨️ Flow Chart of Code

![](https://github.com/TilkeyYANG/M2-DeepLearning/raw/master/imgs/flowchart.jpg)
