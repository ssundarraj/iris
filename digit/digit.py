from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
import csv


train_X = []
train_Y = []
test_X = []
test_Y = []
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
        # i += 1
        # if i >= n_cases:
        #     break

with open('dataset/test.csv', 'rb') as csvfile:
    digitdata = csv.reader(csvfile, delimiter=',')
    next(digitdata, None)
    for r in digitdata:
        r = [float(o) for o in r]
        test_Y.append(r[0])
        test_X.append(r)

X = preprocessing.scale(X)
test_X = preprocessing.scale(test_X)

for i in range(n_cases):
    train_X.append(X[i])
    train_Y.append(Y[i])

print "learning cases {}".format(len(X))
print "test cases {}".format(len(test_X))

clf = svm.SVC()
clf.fit(X, Y) 

print clf

preds = clf.predict(test_X)
with open('dataset/submission.csv', 'w') as opfile:
    for pred in preds:
        pred = int(pred)
        print pred
        opfile.write(str(pred))

# accuracy_score = metrics.accuracy_score(preds, test_Y)
# print "accuracy {}".format(accuracy_score)

