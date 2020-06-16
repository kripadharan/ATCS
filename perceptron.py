import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import Perceptron

iris_data = pd.read_csv("iris_data.csv" ) # , names=column_names )

"""Demonstrate Perception on Iris Data
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris-setosa
      -- Iris-versicolor
      -- Iris-virginica
British statistician and biologist Ronald Fisher
"The use of multiple measurements in taxonomic problems" 
    Annual Eugenics, 7, Part II, 179-188 (1936)
"""

iris_data.info()
print(iris_data.head())
print(iris_data['class'].value_counts())

def iris_to_color(iris):
   if iris=='Iris-setosa':
      return "red"
   elif iris=='Iris-versicolor':
      return "blue"
   else:
      return "green"

"""
plt.scatter(iris_data['sepal width'], iris_data['sepal length'], c=iris_data['class'].apply(iris_to_color))
plt.xlabel("sepal width")
plt.ylabel("sepal length")
plt.title("Red = Iris-setosa, Blue = Iris-versicolor, Green = Iris-virginica")
plt.show()
plt.scatter(iris_data['petal width'], iris_data['petal length'], c=iris_data['class'].apply(iris_to_color))
plt.xlabel("petal width")
plt.ylabel("petal length")
plt.title("Red = Iris-setosa, Blue = Iris-versicolor, Green = Iris-virginica")
plt.show()
"""


f, axarr = plt.subplots(2,3)
axarr[0,0].scatter(iris_data['sepal width'], iris_data['sepal length'], c=iris_data['class'].apply(iris_to_color))
axarr[0,0].set_title("sepal length vs sepal width")
axarr[0,1].scatter(iris_data['petal width'], iris_data['petal length'], c=iris_data['class'].apply(iris_to_color))
axarr[0,1].set_title("petal length vs petal width")
axarr[1,0].scatter(iris_data['sepal width'], iris_data['petal length'], c=iris_data['class'].apply(iris_to_color))
axarr[1,0].set_title("petal length vs sepal width")
axarr[1,1].scatter(iris_data['sepal length'], iris_data['petal length'], c=iris_data['class'].apply(iris_to_color))
axarr[1,1].set_title("petal length vs sepal length")
axarr[0,2].scatter(iris_data['sepal length'], iris_data['petal width'], c=iris_data['class'].apply(iris_to_color))
axarr[0,2].set_title("petal width vs sepal length")
axarr[1,2].scatter(iris_data['sepal width'], iris_data['petal width'], c=iris_data['class'].apply(iris_to_color))
axarr[1,2].set_title("petal width vs sepal width")
plt.title("Red = Iris-setosa, Blue = Iris-versicolor, Green = Iris-virginica")
plt.show()




# What about our buddy the Perceptron?

def percey_demo(iris, column, epochs):
    def class_to_targets(target):
        if target == 'Iris-' + iris:
            return 1
        else:
            return 0

    percey = Perceptron()
    print("Training a Perceptron to classify Iris-" + iris + " using " + column + " width and " + column + " length.")
    iris_inputs = iris_data[ [column + ' width', column + ' length'] ]
    iris_targets = iris_data['class'].apply(class_to_targets)

    xmin=min(iris_inputs[iris_inputs.columns[0]])
    xmax=max(iris_inputs[iris_inputs.columns[0]])
    xnums = np.arange(xmin, xmax,(xmax-xmin)/100)

    for x in range(epochs):
        #print(np.unique(iris_targets))
        percey.partial_fit(iris_inputs, iris_targets, classes=np.unique(iris_targets))
        weights = percey.coef_[0]
        threshold = percey.intercept_
        # print(threshold)
        # print(weights)

        def makeline(xval):
            return (-threshold-weights[0]*xval)/weights[1]

        plt.scatter(iris_data[column + " width"], iris_data[column + " length"], c=iris_data['class'].apply(iris_to_color))
        plt.plot(xnums,makeline(xnums),c="orange")
        plt.xlabel(column + " width")
        plt.ylabel(column + " length")
        plt.axis(ymin=min(iris_inputs[iris_inputs.columns[1]])-0.5, ymax=max(iris_inputs[iris_inputs.columns[1]])+0.5)
        plt.title("Training Perceptron to identify Iris-" + iris + " on epoch=" + str(x) +
                    "\nRed = Iris-setosa, Blue = Iris-versicolor, Green = Iris-virginica")
        plt.show()

# percey_demo("setosa", "petal", 3)   
# percey_demo("virginica", "petal", 10) 


def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("Confusion Matrix:")
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))


def class_to_targets(target):
    if target == 'Iris-virginica':
        return 1
    else:
        return 0


# Full fit with two inputs
iris_inputs = iris_data[ ['petal width', 'petal length'] ]
iris_targets = iris_targets = iris_data['class'].apply(class_to_targets)

percey = Perceptron()

print("Training a Perceptron to classify Iris-Virginica using petal inputs.")
percey.fit(iris_inputs, iris_targets)
percey_outputs = percey.predict(iris_inputs)

print("Found weights=" + str(percey.coef_) + " and threshold: " + str(percey.intercept_) + " in " + str(percey.n_iter_) + " epochs.")
print("Mean accuracy:", percey.score(iris_inputs, iris_targets))
print_conf_matrix(iris_targets, percey_outputs)
print("Precision = TP / (TP + FP) = ", precision_score(iris_targets, percey_outputs))
print("Recall = TP / (TP + FN) = ", recall_score(iris_targets, percey_outputs))


print("\n\nLet's make it train longer:")
percey2 = Perceptron(n_iter_no_change=10)
percey2.fit(iris_inputs, iris_targets)
percey2_outputs = percey2.predict(iris_inputs)

print("Found weights=" + str(percey2.coef_) + " and threshold: " + str(percey2.intercept_) + " in " + str(percey2.n_iter_) + " epochs.") 
print("Mean accuracy:", percey2.score(iris_inputs, iris_targets))
print_conf_matrix(iris_targets, percey2_outputs)
print("Precision = TP / (TP + FP) = ", precision_score(iris_targets, percey2_outputs))
print("Recall = TP / (TP + FN) = ", recall_score(iris_targets, percey2_outputs))

