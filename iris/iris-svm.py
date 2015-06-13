from sklearn import svm
import csv


flower_names = {
    'Iris-setosa': 1, #(1, 0, 0),
    'Iris-versicolor': 2, #(0, 1, 0),
    'Iris-virginica': 3, #(0, 0, 1)
}

X = []
Y = []

test_set = []

i = 0
with open('iris.csv', 'rb') as csvfile:
    irisdata = csv.reader(csvfile, delimiter=' ')
    for r in irisdata:
        t = r[0].split(",")
        if i % 40 in range(39):
            test_set.append(t)
        else:
            Y.append(flower_names[t[4]])
            X.append(t[0:4])
        i += 1

# print X
# print Y
print "learning cases {}".format(len(X))
print "test cases {}".format(len(test_set))

clf = svm.SVC()
clf.fit(X, Y) 

n_infractions = 0
for test in test_set:
    if not clf.predict([test[0:4]]) == flower_names[test[4]]:
        n_infractions += 1

print "incorrect predictions {}".format(n_infractions)
