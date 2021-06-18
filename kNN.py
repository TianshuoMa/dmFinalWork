import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x_test = pd.read_csv("data_test.csv",sep=",")
y_test = pd.read_csv("target_test.csv",sep=",")
x_train = pd.read_csv("data_train.csv",sep=",")
y_train = pd.read_csv("target_train.csv",sep=",")

knn = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto', p=2, metric='minkowski')

knn.fit(x_train, y_train)

test_accuracy =[]
neighbors_settings = range(1, 101)
for n_neighbors in neighbors_settings:
    clf= KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(x_train,y_train)
    test_accuracy.append(clf.score(x_test,y_test))
plt.plot(neighbors_settings,test_accuracy, label="accuracy of the testing data")
plt.ylabel('Accuracy')
plt.xlabel('Number of Neibhbors')
plt.show()

print(knn.predict(x_test[:1]))
print(y_test[:1])
print(knn.score(x_test, y_test))