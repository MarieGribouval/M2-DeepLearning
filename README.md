# M2-DeepLearning

Introduction :

  La survie des abeilles est cruciale puisque en plus de produire par exemple du miel, elles permettent la pollinisation d’une grande partie des plantes à fleurs. Sans elles l’agriculture serait en péril. Afin de surveiller la bonne santé d’une ruche, les agriculteurs l’ouvrent pour vérifier plusieurs paramètres comme le nombre d’œufs, la présence de varroa,… Néanmoins cette vérification est une source de stress importante pour la ruche et peut même lui être néfaste. Dans le climat actuel où la survie des abeilles devient un problème mondial, il est important de trouver une solution non invasive pour surveiller la santé de la ruche.
Cet article utilise le critère de la quantité de pollen afin d’évaluer l’état de la ruche. En effet, si cette quantité est insuffisante les abeilles manquent de nourriture, et la colonie est affaiblie. Pour cela des images d’abeilles à l’entrée de la ruche sont exploitées. Il est plus simple d’extraire des données à partir de l’extérieur de la ruche par rapport à l’intérieur car sinon il est plus complexe de déplacer la ruche.
Les auteurs ont pour objectif de pouvoir à partir d’une image, prédire si l’abeille porte du pollen ou non. Pour cela ils ont comparé plusieurs méthodes de classification. Leurs meilleurs résultats ont été obtenu avec  un réseau de neurone convolutionnel (CNN), à partir de là ils ont cherché les meilleurs hyper paramètres.
La 2e section décrit la base de données utilisée, ensuite, la configuration du CNN est détaillée dans la section 3, puis dans la dernière section, les auteurs présentent les résultats qu’ils ont obtenu.

