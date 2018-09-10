import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))
#Class labels: [0 1 2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
#Labels counts in y: [50 50 50]
print('Labels counts in y_train:', np.bincount(y_train))
#Labels counts in y_train: [35 35 35]
print('Labels counts in y_test:', np.bincount(y_test))
#Labels counts in y_test: [15 15 15]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

from mlxtend.plotting import plot_decision_regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, y_combined, clf=knn)

import matplotlib.pyplot as plt
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

y_pred = knn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

print('Accuracy: %.2f' % knn.score(X_test_std, y_test),'\n')

#try K=1 through K=25 and record testing accuracy
from sklearn.metrics import accuracy_score
k_range=range(1,26)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    y_pred=knn.predict(X_test_std)
    scores.append(accuracy_score(y_test, y_pred))
score_opt=max(scores)
opt_list=list(enumerate(scores, start=1)) 
print ('The best choice of K for this data: ',[i for i, x in opt_list if x == score_opt])
print('The accurary of the best choice of K for this data: ',score_opt,'\n')

print("My name is QI GUO")
print("My NetID is: qiguo3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################



