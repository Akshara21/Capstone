import numpy as np
import pandas as pd
from numpy import random

class Perceptron:
    def __init__(self, learning_rate=0.01, MaxIter=20):
        self.MaxIter = MaxIter
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None   

    def fit(self, X,y):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0                                                # Input the total number of iterations to be performed in the training phase as “MaxIter, here maximum iterations = 20 as given.

        for i in range(self.MaxIter):
            np.random.seed(i)
            r = np.arange(len(y))
            np.random.shuffle(r)
            X = X[r]
            y = y[r]
            for i, x in enumerate(X):   
                act_val = np.dot(x, self.weights) + self.bias          # Calculating the activation function by using the formula, a = (W*X) + bias.Reference Used: Perceptron Lecture Notes 
                y_predicted = np.where(act_val >= 0, 1, -1)
                upd = self.learning_rate * (y[i] - y_predicted)    # If activation function value, (act_val)<= 0 then the classification is incorrect, Adjust the weights and bias by modifying the formulas wi = wi + y * xi &  bias = bias + y.
                self.weights  = self.weights + upd * x
                self.bias = self.bias + upd                
        return self.weights
    def predict(self, X):                                         # Pass the test feature set as input to the predict function and get the value of predicted value 'y_predicted' 
        act_val = np.dot(X , self.weights) + self.bias
        y_predicted = np.where(act_val >= 0, 1, -1)
        return y_predicted,act_val


train= np.array(pd.read_csv("./train.data",header=None))  # given i/p
test = np.array(pd.read_csv("./test.data",header=None))
X_train = train[:,0:4]               # train data
y_train = train[:,4]
X_test = test[:,0:4]                 # test data
y_test = test[:,4]

def split(X_train,y_train,a,b,c):      # func to split train data 
    y1 = np.array([1 if i == 'class-'+str(a) else -1 if i == 'class-'+str(b) else 0 if i == 'class-'+str(c) else '' for i in y_train])
    data = np.column_stack((X_train,y1))
    final_train= np.empty((0,5))
    for i in range(0,len(data)):
        if (data[i][4]) == 0: 
            pass
        else:
            final_train = np.vstack((final_train,data[i]))
    return final_train

def test_split(X_test,y_test,a,b,c):     # func to split TEST data 
    y2 = np.array([1 if i == 'class-'+str(a) else -1 if i == 'class-'+str(b) else 0 if i == 'class-'+str(c) else '' for i in y_test])
    temp = np.column_stack((X_test,y2))
    final_test= np.empty((0,5))
    for i in range(0,len(temp)):
        if (temp[i][4]) == 0: 
            pass
        else:
            final_test = np.vstack((final_test,temp[i]))
    return final_test

p = Perceptron(MaxIter=20)


final_train_1= np.array((split(X_train,y_train,1,2,3)))          #CLASS 1 VS CLASS 2
X_train_1 = final_train_1[:,0:4]
y_train_1 = final_train_1[:,4]
final_test_1= np.array((test_split(X_test,y_test,1,2,3)))
X_test_1 = final_test_1[:,0:4]
y_test_1 = final_test_1[:,4]
p.fit(X_train_1,y_train_1)
class_train_1_2 =p.predict(X_train_1)
class_test_1_2 = p.predict(X_test_1)
accuracy_test_1_2 = np.sum(y_test_1 == class_test_1_2[0]) / len(y_test_1)
accuracy_train_1_2 = np.sum(y_train_1 == class_train_1_2[0]) / len(y_train_1)


final_train_2= np.array((split(X_train,y_train,2,3,1)))         #CLASS 2 VS CLASS 3
X_train_2 = final_train_2[:,0:4]
y_train_2 = final_train_2[:,4]
final_test_2= np.array((test_split(X_test,y_test,2,3,1)))
X_test_2 = final_test_2[:,0:4]
y_test_2= final_test_2[:,4]

p.fit(X_train_2,y_train_2)
class_test_2_3 = p.predict(X_test_2)
class_train_2_3 = p.predict(X_train_2)
accuracy_train_2_3 = np.sum(y_train_2 == class_train_2_3[0]) / len(y_train_2)
accuracy_test_2_3 = np.sum(y_test_2 == class_test_2_3[0]) / len(y_test_2)


final_train_3= np.array((split(X_train,y_train,1,3,2)))         #CLASS 1 VS CLASS 3
X_train_3 = final_train_3[:,0:4]
y_train_3 = final_train_3[:,4]
final_test_3= np.array((test_split(X_test,y_test,1,3,2)))
X_test_3 = final_test_3[:,0:4]
y_test_3= final_test_3[:,4]

p.fit(X_train_3,y_train_3)
class_test_1_3 =p.predict(X_test_3)
class_train_1_3 =p.predict(X_train_3)
accuracy_test_1_3 = np.sum(y_test_3 == class_test_1_3[0]) / len(y_test_3)
accuracy_train_1_3 = np.sum(y_train_3 == class_train_1_3[0]) / len(y_train_3)

print("Binary Classification:")
print("-----------------------\n")
print(f"Accuracy for class_1_2 test data: {accuracy_test_1_2*100}")
print(f"Accuracy for class_1_2 train data: {accuracy_train_1_2*100}")
print(f"Accuracy for class_2_3 test data: {accuracy_test_2_3*100}")
print(f"Accuracy for class_2_3 train data: {accuracy_train_2_3*100}")
print(f"Accuracy for class_1_3 test data: {accuracy_test_1_3*100}")
print(f"Accuracy for class_1_3 train data: {accuracy_train_1_3*100}")

final_rest= np.empty((0,5))
# func to split train data 
def split_rest(X,y,a,b,c):
    y1 = np.array([1 if i == 'class-'+str(a) else -1 if i == 'class-'+str(b) else -1 if i == 'class-'+str(c) else '' for i in y])
    final_rest = np.column_stack((X,y1))
    return final_rest
# split_rest(X,y,1,2,3)
# given i/p -- test

final_rest_train= np.array((split_rest(X_train,y_train,1,2,3)))
final_rest_test= np.array((split_rest(X_test,y_test,1,2,3)))

# Train Data
X_rest_train_1 = final_rest_train[:,0:4]
y_rest_train_1 = final_rest_train[:,4]

# Test Data
X_rest_test_1 = final_rest_test[:,0:4]
y_rest_test_1 = final_rest_test[:,4]

# Train, Predict , Accuracy
p.fit(X_rest_train_1,y_rest_train_1)
class_rest_train_1 =p.predict(X_rest_train_1)
class_rest_test_1 =p.predict(X_rest_test_1)

final_rest_train_2 = np.array((split_rest(X_train,y_train,2,1,3)))
final_rest_test_2 = np.array((split_rest(X_test,y_test,2,1,3)))

# Train Data
X_rest_train_2 = final_rest_train_2[:,0:4]
y_rest_train_2 = final_rest_train_2[:,4]

# Test Data
X_rest_test_2 = final_rest_test_2[:,0:4]
y_rest_test_2= final_rest_test_2[:,4]


p.fit(X_rest_train_2,y_rest_train_2)
class_rest_train_2 =p.predict(X_rest_train_2)
class_rest_test_2 =p.predict(X_rest_test_2)

final_rest_train_3 = np.array((split_rest(X_train,y_train,3,1,2)))
final_rest_test_3 = np.array((split_rest(X_test,y_test,3,1,2)))

# Train Data
X_rest_train_3 = final_rest_train_3[:,0:4]
y_rest_train_3 = final_rest_train_3[:,4]

# Test Data
X_rest_test_3 = final_rest_test_3[:,0:4]
y_rest_test_3= final_rest_test_3[:,4]

# Train, Predict , Accuracy class 2 and class 3

p.fit(X_rest_train_3,y_rest_train_3)
class_rest_train_3 =p.predict(X_rest_train_3)
class_rest_test_3 =p.predict(X_rest_test_3)


# Test Data 
class_predict = np.array([])
for i in range(len(class_rest_test_1[1])):
    if max(class_rest_test_1[1][i],class_rest_test_2[1][i],class_rest_test_3[1][i]) == class_rest_test_1[1][i]:
        class_predict = np.hstack((class_predict, "class-1"))
    elif max(class_rest_test_1[1][i],class_rest_test_2[1][i],class_rest_test_3[1][i]) == class_rest_test_2[1][i]:
        class_predict = np.hstack((class_predict, "class-2"))
    elif max(class_rest_test_1[1][i],class_rest_test_2[1][i],class_rest_test_3[1][i]) == class_rest_test_3[1][i]:
        class_predict = np.hstack((class_predict, "class-3"))
class_predict    
# Accuracy calculation for test classes
accuracy_test_class_pred = np.sum(y_test == class_predict) / len(y_test)
accuracy_test_class_pred
print("\n Multiclass Classification using the one vs rest approach:")
print("------------------------------------------------------------\n")
print("Accuracy value of Test Class - One vs Rest : ", round((accuracy_test_class_pred*100),2))

# Train Data 
class_train_predict = np.array([])
for i in range(len(class_rest_train_1[1])):
    if max(class_rest_train_1[1][i],class_rest_train_2[1][i],class_rest_train_3[1][i]) == class_rest_train_1[1][i]:
        class_train_predict = np.hstack((class_train_predict, "class-1"))
    elif max(class_rest_train_1[1][i],class_rest_train_2[1][i],class_rest_train_3[1][i]) == class_rest_train_2[1][i]:
        class_train_predict = np.hstack((class_train_predict, "class-2"))
    elif max(class_rest_train_1[1][i],class_rest_train_2[1][i],class_rest_train_3[1][i]) == class_rest_train_3[1][i]:
        class_train_predict = np.hstack((class_train_predict, "class-3"))
class_train_predict
# Accuracy calculation for train classes
accuracy_train_class_pred = np.sum(y_train == class_train_predict) / len(y_train)
accuracy_train_class_pred

print("Accuracy value of Train Class - One vs Rest : ", round((accuracy_train_class_pred*100),2),"\n")
# L2 Reg

print("Adding l2 regularisation term to multi-class classifier:")
print("--------------------------------------------------------\n")
class Perceptron_reg:
    def __init__(self, learning_rate=0.01, MaxIter=20):
        self.MaxIter = MaxIter
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None   

    def fit(self, X,y,l):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0                                                # Input the total number of iterations to be performed in the training phase as “MaxIter, here maximum iterations = 20 as given.

        for i in range(self.MaxIter):
            np.random.seed(i)
            r = np.arange(len(y))
            np.random.shuffle(r)
            X = X[r]
            y = y[r]
            for i, x in enumerate(X):   
                act_val = np.dot(x, self.weights) + self.bias          # Calculating the activation function by using the formula, a = (W*X) + bias.Reference Used: Perceptron Lecture Notes 
                y_predicted = np.where(act_val >= 0, 1, -1)
                upd = self.learning_rate * (y[i] - y_predicted)    # If activation function value, (act_val)<= 0 then the classification is incorrect, Adjust the weights and bias by modifying the formulas wi = wi + y * xi &  bias = bias + y.
                self.weights  = (1 - 2*l) * self.weights + upd * x
                self.bias = self.bias + upd                
        return self.weights
    def predict(self, X):                                         # Pass the test feature set as input to the predict function and get the value of predicted value 'y_predicted' 
        act_val = np.dot(X , self.weights) + self.bias
        y_predicted = np.where(act_val >= 0, 1, -1)
        return y_predicted,act_val

l2 = Perceptron_reg(MaxIter=20,learning_rate=0.01)

for index in  [.01, 0.1, 1.0, 10.0, 100.0]:

    l2.fit(X_rest_train_1,y_rest_train_1,index)
    l2_train_1 =p.predict(X_rest_train_1)
    l2_test_1 =p.predict(X_rest_test_1)

    l2.fit(X_rest_train_2,y_rest_train_2,index)
    l2_train_2 =p.predict(X_rest_train_2)
    l2_test_2 =p.predict(X_rest_test_2)

    l2.fit(X_rest_train_3,y_rest_train_3,index)
    l2_train_3 =p.predict(X_rest_train_3)
    l2_test_3 =p.predict(X_rest_test_3)
    
    l2_test_predict_1 = np.array([])
    for i in range(len(l2_test_3[1])):
        if max(l2_test_3[1][i],l2_test_2[1][i],l2_test_1[1][i]) == l2_test_1[1][i]:
            l2_test_predict_1 = np.hstack((l2_test_predict_1, "class-1"))
        elif vmax(l2_test_3[1][i],l2_test_2[1][i],l2_test_1[1][i]) == l2_test_2[1][i]:
            l2_test_predict_1 = np.hstack((l2_test_predict_1, "class-2"))
        elif vmax(l2_test_3[1][i],l2_test_2[1][i],l2_test_1[1][i]) == l2_test_3[1][i]:
            l2_test_predict_1 = np.hstack((l2_test_predict_1, "class-3"))
    l2_test_predict_1    
    # Accuracy calculation for test classes
    accuracy_l2_test_1 = round(np.sum(y_test == l2_test_predict_1) / len(y_test),2)
    accuracy_l2_test_1

    print("Accuracy of test class predicted with lambda : ",index," is", accuracy_l2_test_1*100)

    l2_train_predict_1 = np.array([])
    for i in range(len(l2_train_1[1])):
        if max(l2_train_1[1][i],l2_train_2[1][i],l2_train_3[1][i]) == l2_train_1[1][i]:
            l2_train_predict_1 = np.hstack((l2_train_predict_1, "class-1"))
        elif max(l2_train_1[1][i],l2_train_2[1][i],l2_train_3[1][i]) == l2_train_2[1][i]:
            l2_train_predict_1 = np.hstack((l2_train_predict_1, "class-2"))
        elif max(l2_train_1[1][i],l2_train_2[1][i],l2_train_3[1][i]) == l2_train_3[1][i]:
            l2_train_predict_1 = np.hstack((l2_train_predict_1, "class-3"))
    l2_train_predict_1    
    # Accuracy calculation for test classes
    accuracy_l2_train_1 = round(np.sum(y_train == l2_train_predict_1) / len(y_train),2)

    print("Accuracy of train class predicted with lambda : ",index," is" ,accuracy_l2_train_1*100,"\n")









