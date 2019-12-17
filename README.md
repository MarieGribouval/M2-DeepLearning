# Deep Bee

Deep Learning Project of Master 2 of Data Science
\[Project Report\] | [Article Summary](./RESUME.md)

Marie Gribouval, Léo Boule and Leshanshui YANG from University of Rouen


## 📑 Project Introduction
Since the survival of bees is important to the environment of the planet, it is important to find a non-destructive solution to monitor the health of bees. To this end, we studied the article "The Application of Convolutional Neural Network for Pollen Bearing Bee Classification" in this project.

We use the same data set from kaggle as in the article. We achieved the purpose of bee species and health classification through a network with less than 10K parameters. We tested the model mentioned in the article and proposed some changes and experiment.

(For more details of the article, please find our summary of the article (Résumé de l'article) in [RESUME.md](./RESUME.md).

## 🚩 Target
The purpose of this project is to implement the **data pre-processing and model building** for reproducing the experiment from the original article, and to **test the performance of RGB and HSV** inputs under very few layers of network, comparing **the speed of CPU with Jetson GPU**.

## 📊 Discussion of Results


With very few effective iterations, ***we obtained the results very close to the article with classification accuracy rates (90.89%) in terms of subspecies and accuracy (91.71%) in terms of health.***

### Data set

  The original dataset we used consists of 4744 rows of data, where **X is a photo of the bee** and **Y is the label** of the photo (about the region, health level, honey carried, subpecies, etc.).
  
  In our experiments, we used 4098 pieces of data smaller than 100 * 100 pixels. We selected **health and subpecies** as the Y to predict(classify) in 2 independent models.
  
  We used keras' **stratified data splitting** function to ensure that the label proportion (distribution) of the training set and the test set is basically the same.
    
    | Type          | Data          |
    | ------------- | ------------- |
    | Total         | 4098          |
    | Train         | 2582 (63%)    |
    | Valid         |  287 (07%)    |
    | Test          | 1220 (30%)    |


### Model and experimental parameters

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

It is worth noting that since we set Early Stopping of 20 Epochs and only save the optimal model (Loss of Validation is the smallest). In that case, we actually used the model stored in the sixth Epoch during the test phase, instead of the model after "overfitting".

With the same method (checkpoint + earlystopping), we obtained the following results:

![](https://github.com/TilkeyYANG/M2-DeepLearning/raw/master/imgs/accuracy.jpg)

> *`RGB-enh` means we adjusted the saturation and brightness of the image in order to enhance the image quality.*
> *`3 layers` of HSV means we added another convolution layer, which is another trick mentionned in the article.*


## ⌨️ Code Description




