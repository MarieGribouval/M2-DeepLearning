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

### Data set

  The original dataset we used consists of 4744 rows of data, where **X is a photo of the bee** and **Y is the label** of the photo (about the region, health level, honey carried, subpecies, etc.).
  
  In our experiments, we used 4098 pieces of data smaller than 100 * 100 pixels. We selected **health and subpecies** as the Y to predict(classify) in 2 independent models.
  
  We used keras' **strtified data splitting** function to ensure that the label proportion (distribution) of the training set and the test set is basically the same.
  
    | Type          | Data lines    |
    
    | ------------- | ------------- |
   
    | Total         | 4098          |
    
    | Train         | 2582          |
    
    | Valid         |  287          |
    
    | Test          | 1220          |





## ⌨️ Code Description




