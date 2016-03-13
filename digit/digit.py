from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
import csv


X = []
Y = []
n_cases = 4000
i = 0
with open('dataset/train.csv', 'rb') as csvfile:
    digitdata = csv.reader(csvfile, delimiter=',')
    next(digitdata, None)
    for r in digitdata:
        r = [float(o) for o in r]
        Y.append(r[0])
        X.append(r[1:])
        i += 1
        if i >= n_cases:
            break


X = preprocessing.scale(X)
train_X = []
train_Y = []
test_X = []
test_Y = []

for i in range(n_cases):
    if i%4:
        train_X.append(X[i])
        train_Y.append(Y[i])
    else:
        test_X.append(X[i])
        test_Y.append(Y[i])

print "learning cases {}".format(len(X))
print "test cases {}".format(len(test_X))

clf = svm.SVC()
clf.fit(X, Y) 

print clf

n_infractions = 0

preds = clf.predict(test_X)
accuracy_score = metrics.accuracy_score(preds, test_Y)

print "incorrect predictions {}".format(n_infractions)
print "accuracy {}".format(accuracy_score)
