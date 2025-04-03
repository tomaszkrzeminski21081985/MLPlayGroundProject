
import cmath
from scipy.stats import f
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist['data'], mnist['target']


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')
   
some_digit=X[0]
# plot_digit(some_digit)
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

print(np.unique(y_train))

sgd_clf=SGDClassifier(random_state=45)
sgd_clf.fit(X_train, y_train_5)


predict=sgd_clf.predict([some_digit])
print(predict)

score2=sgd_clf.decision_function([some_digit])
print(score2)

score=cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(score)

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) 
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

f1_score = f1_score( y_train_5, y_train_pred)
print(f1_score)


y_scores=cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

roc_xxx=roc_auc_score(y_train_5, y_scores)
print(roc_xxx)

forest_clf=RandomForestClassifier(random_state=42)
y_scores=cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')