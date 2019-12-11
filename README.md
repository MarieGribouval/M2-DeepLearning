# M2-DeepLearning

## Résumé de l'article "The Application of Convolutional Neural Network for Pollen Bearing Bee Classification" de Tomyslav Sledevic, 2018 IEEE 6th Workshop on Advances in Information, Electronic and Electrical Engineering


## Introduction

  La survie des abeilles est cruciale puisqu'en plus de produire par exemple du miel, elles permettent la pollinisation d’une grande partie des plantes à fleurs. Sans elles l’agriculture serait en péril. Afin de surveiller la bonne santé d’une ruche, les agriculteurs l’ouvrent pour vérifier plusieurs paramètres comme le nombre d’œufs, la présence de varroa,… Néanmoins cette vérification est une source de stress importante pour la ruche et peut même lui être néfaste. Dans le climat actuel où la survie des abeilles devient un problème mondial, il est important de trouver une solution non invasive pour surveiller la santé de la ruche.

Cet article utilise le critère de la quantité de pollen afin d’évaluer l’état de la ruche. En effet, si cette quantité est insuffisante les abeilles manquent de nourriture, et la colonie est affaiblie. Pour cela des images d’abeilles à l’entrée de la ruche sont exploitées. Il est plus simple d’extraire des données à partir de l’extérieur de la ruche par rapport à l’intérieur car dans le dernier cas il est plus complexe de déplacer la ruche.

Les auteurs ont pour objectif de pouvoir à partir d’une image, prédire si l’abeille porte du pollen ou non. Pour cela ils ont comparé plusieurs méthodes de classification. Leurs meilleurs résultats ont été obtenu avec  un réseau de neurone convolutionnel (CNN), à partir de là, ils ont cherché les meilleurs hyper paramètres.

La 2ème section décrit la base de données utilisée, ensuite, la configuration du CNN est détaillée dans la section 3, puis dans la dernière section, les auteurs présentent les résultats qu’ils ont obtenus. 

## Partie 2 - Pollen Bearing Bee Dataset 
![Model proposed in article Sledevič, Tomyslav. "The application of convolutional neural network for pollen bearing bee classification." In 2018 IEEE 6th Workshop on Advances in Information, Electronic and Electrical Engineering (AIEEE), pp. 1-4. IEEE, 2018.](https://github.com/TilkeyYANG/M2-DeepLearning/raw/master/model.jpg)
  Des images d’entrée de ruches ont été prises, sur ces images de base des images de taille 100x100 pixels sont extraites à l’endroit où une abeille est détectée. Grâce à cela une base de donnée contenant 1000 images avec des abeilles portant du pollen, et 1000 autres sans est créée. 
L’orientation des abeilles n’a pas été modifiée suite à l’extraction dans l’éventualité où cela est étudié ultérieurement. Sur certaines images les abeilles sont partiellement visibles à cause des ombres ou de l’entrée de la ruche.

## Partie 3 - Training of the CNN

Pour l’apprentissage du CNN, 10% des données de train sont utilisées pour la validation. La couche d’entrée reçoit des images en couleur RGB ou HSV. Le nombre de couche cachée varie de un à trois afin de comparer l’impact de ce paramètre. 
Chaque couche est composée d’une convolution qui permet d’extraire plusieurs caractéristiques (selon le nombre de filtres choisi) puis d’un max polling de taille 2x2 qui diminue les dimensions des images obtenues à la sortie de la convolution. Entre les deux, une normalisation des données est réalisée. A l’issu du dernier max polling, toutes nos images construites dans les couches cachées sont réunies en un unique vecteur, et, enfin ce vecteur est réduit à un scalaire. Cela est représenté par “Fully Connected” dans le schéma.
La couche de sortie classifie chaque image comme ayant ou non du pollen selon le scalaire.

## Partie 4 - Classification results

  Différents paramètres de l’architecture du CNN impactent le taux de bonne classification : le nombre de couche cachée, la taille du filtre et le nombre de filtre. Indépendamment de ces paramètres, le nombre d’époque est également important car il en faut une centaine afin d’obtenir un taux de bonne classification stable.
Le CNN n’est pas totalement robuste à l’arrière-plan car un CNN travaillant avec les images d’une même entrée de ruche donne de meilleurs résultats que l’utilisation de toutes les données.
Des filtres de taille variables ont été utilisés dans les couches convolutionnelles, leur taille optimale dépend des images initiales utilisées. Ces dernières font aussi varier le nombre de couches cachées puisque avec des images HSV, les meilleurs résultats (92-94%) sont obtenus avec 3 couches cachées, alors qu’il en faut 2 pour obtenir des résultats similaires avec les images RGB.
Finalement, pour l’application FPGA, l’architecture est la suivante : 3 couches cachées ; pour la première 7 filtres, chacun de taille 7x7, pour la deuxième 5 filtres 5x5, pour la troisième trois filtres 3x3. C’est l’architecture avec trois couches cachées qui est choisie car elle est moins gourmande en calculs que celle avec deux couches cachées pour des résultats similaires. 

## Conclusion

  Différentes architectures de CNN ont pu être comparé dans ce papier. Cela permet de choisir le CNN optimal à utiliser dans l’application pour détecter le pollen sur les abeilles. Dans un contexte réel, il faut au préalable détecter les régions d’intérêt des images prises.
