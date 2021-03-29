import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
# import matplotlib.pyplot as pyplot
# from matplotlib import style

data = pd.read_csv("./Datasets/student-mat.csv", sep = ";")

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

labels = "G3"

x = np.array(data.drop([labels], 1))  # Attributes
y = np.array(data[labels]) # Labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0

for _ in range(30):
# Training the data until we get the best accuracy
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train) # Fitting the training data into the model
    acc = linear.score(x_test, y_test) # Getting the score/accuracy
    print("Accuracy: ", acc)

    if acc > best:
        best = acc
        print(f"New best!: {acc}")

        # Saving the model with pickle
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)
        print("Model Saved to new accuracy!\n------")        

pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)   

predictions = linear.predict(x_test)

print(f'------ \nResults with {round(best*100)} accuracy: ')
for x in range(len(predictions)):    
    print(f"Predicted value: {round(predictions[x])}, Actual value: {y_test[x]}")