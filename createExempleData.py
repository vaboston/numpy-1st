#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt


# tableau une dimension
a = np.array([1,2,3])
# print("a :", a)

# tableau 2 dimensions, type float
b = np.array([(1.4,2.4,5.5,6.6), (3.5,6.2,7.4,8.8)], dtype = float)
print("b :", b)

b = b.reshape(8 ,)
print("b :", b)


bb = np.array([(1.4,2.4), (3.5,6.2),(4.4,5.6)], dtype = float)
# print("bb :", bb)

# placeholder = espace reserve


# genere les lignes (rows) avec un 
# numpy.random.randn(10, 10) : array 2d de 10 x 10 nombres dune distribution gaussienne standard(moyenne 0, ecart-type 1).

# Pour les nombres [1,2,3], la moyenne est de 2, la variance est 0,667
# [(1 - 2)2 + (2 - 2)2 + (3 - 2)2] ÷ 3 = 0,667
# [somme de l'écart au carré] ÷ nombre d'observations = variance

# Variance, S2 = moyenne de l ecart au carre de valeurs par rapport a la moyenne
# Comme le calcul de la variance se fait a partir des carres des ecarts, les unites de mesure ne sont pas les memes que celles des observations originales. Par exemple, les longueurs mesurees en metres (m) ont une variance mesuree en metres carres (m2).
# La racine carree de la variance nous donne les unites utilisees dans l echelle originale.

# ecart-type (S) = Racine carree de la variance
# L ecart-type est la mesure de dispersion la plus couramment utilisee en statistique lorsqu on emploie la moyenne pour calculer une tendance centrale. Il mesure donc la dispersion autour de la moyenne. En raison de ses liens etroits avec la moyenne, l ecart-type peut etre grandement influence si cette derniere donne une mauvaise mesure de tendance centrale.


# ecart type:  

# numpy.random.randn(nombre_de_rows, nombre_de_dimension) 

aa = np.random.randn(10, 10)
# print("aa :", aa)

ab = np.random.randn(10, 2)
# print("ab :", ab)



# extrait
row_per_class = 2

c = np.random.randn(row_per_class, 2)
# print("c :", c)

d = np.random.randn(row_per_class, 2)
# print("d :", d)

e = c + np.array([-2, -2])
# print("e :", e)

f = d + np.array([2, 2])
# print("f :", f)




# Generate rows
sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
sick_2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
healthy_2 = np.random.randn(row_per_class, 2) + np.array([2, -2])
# print("sick: ", sick)
# print("sick_2: ", sick_2)
# print("healthy: ", healthy)
# print("healthy_2: ", healthy_2)

features = np.vstack([sick, sick_2, healthy, healthy_2])
# print("features: ", features)

features2 = np.concatenate([sick, sick_2, healthy, healthy_2])
# print("features2: ", features2)

g = np.array([[1,2,3],[4,5,6]])
h = np.array([[11,12,13],[14,15,16]])
i = np.vstack([g,h]) # 
j = np.concatenate((g,h),axis=0)
# print("g :", g)
# print("h :", h)
# print("i :", i)
# print("j :", j)





def get_dataset():
    """
        Method used to generate the dataset
    """
    # Numbers of row per class
    row_per_class = 10
    # Generate rows
    # np.random.randn(row_per_class, 2) = genere des points proches de 0, et l'addition de l'array place dans le bon cadran
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    sick_2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    healthy_2 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([sick, sick_2, healthy, healthy_2])
    targets = np.concatenate((np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))
    targets2 = np.arange(40)
    print("features:", features)
    print("targets:", targets)


    targets = targets.reshape(40,)

    return features, targets

if __name__ == '__main__':
    features, targets = get_dataset()
    # Plot points
#    plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.scatter(features[:, 0], features[:, 1], c=targets, cmap=plt.cm.Spectral)

plt.show()




# import numpy as np








### CODE ###

# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np

# def get_dataset():
#     """
#         Method used to generate the dataset
#     """
#     # Numbers of row per class
#     row_per_class = 100
#     # Generate rows
#     np.random.randn(row_per_class, 2) = genere des points proches de 0, et l'addition de l'array place dans le bon cadran
#     sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
#     sick_2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

#     healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
#     healthy_2 = np.random.randn(row_per_class, 2) + np.array([2, -2])

#     features = np.vstack([sick, sick_2, healthy, healthy_2])
#     targets = np.concatenate((np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

#     targets = targets.reshape(400,) # nombre de coordonnée généré avant, reshape modifie la dimension, cela va créer un tableau de 40 éléments dont les 20 premiers sont 0 et les autres 1, les datas étant 
# classés par ordre (malade puis safe), les malades seront en bleu et les autres en rouge (selon la cmap)

#     return features, targets

# if __name__ == '__main__':
#     features, targets = get_dataset()
#     # Plot points
#     plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral) # scatter : nuage de pointe, [:, 0] = colomne 0 de feature, colomone 1 de feature
# plt.show()

####