"""
Shou-Cheng Wu
"""


from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
from sklearn import preprocessing
# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
	# DEFINE __init function
        self.W = W
        self.b = b
    def forward(self, x):
	# DEFINE forward function
        self.x = x 
        return np.dot(self.x, self.W) + self.b
    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.001,
    ):
        # DEFINE backward function
        dw = np.dot(self.x.T, grad_output)
        dx = np.dot(grad_output, self.W.T)
        db = np.sum(grad_output)
        return dw, dx, db
	
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
	# DEFINE forward function
        self.r_out = np.maximum(0, x)
        #print("shape",self.r_out.shape)
        return self.r_out
    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.001,
    ):
        for i in range(0,len(self.r_out)):
            for j in range(0,len(self.r_out[i])):
                if self.r_out[i][j]>0:
                    self.r_out[i][j]=1
        dx = np.multiply(self.r_out, grad_output)
        return dx
    # DEFINE backward function
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self):
        self.sigmoid = None
    def forward(self, x, y):
		# DEFINE forward function
         self.sigmoid = 1.0 / (1.0 + np.exp(-x))
         loss = -1 * (y * np.log(self.sigmoid) + (1.0 - y) * np.log(1.0 - self.sigmoid))
         return self.sigmoid, loss
    def backward(
	    self,
        y, 
	    grad_output, 
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.001
	):
        # DEFINE backward function
        delta = self.sigmoid - y
        return delta
		
# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units

    #initial the weights between -1 and 1 
        # W :input -> hidden
        self.w1 = np.random.uniform(-1.0, 1.0, size=(input_dims, hidden_units))
        # b1 :1 -> hidden
        self.b1 = np.random.uniform(-1.0, 1.0, size=(1, hidden_units))
        # W2 : hidden -> 1
        self.w2 = np.random.uniform(-1.0, 1.0, size=(hidden_units, 1))
        # b2 : 1 -> 1
        self.b2 = np.random.uniform(-1.0, 1.0, size=(1, 1))


        # call the class object
        self.L1 = LinearTransform(self.w1, self.b1)
        self.L2 = LinearTransform(self.w2, self.b2)
        self.relu = ReLU()
        self.SCE = SigmoidCrossEntropy()

        #initial momentum
        self.w1_d = 0
        self.b1_d = 0
        self.w2_d = 0
        self.b2_d= 0


    def train(
        self, 
        train_x, 
        train_y, 
        learning_rate = 0.005, 
        momentum = 0.6,
        l2_penalty =0.0,
        batch_size = 1,
        b = 1,
    ):
        #compute the batch (x and y)
        x_batch = train_x[b * batch_size: (b + 1) * batch_size, :]
        y_batch = train_y[b * batch_size: (b + 1) * batch_size, :]
        # INSERT CODE for training the network
        #forward pass  (L1->relu) (L2->sigmoid)
        z1 =  self.relu.forward(self.L1.forward(x_batch))
        z2, SCE_loss = self.SCE.forward(self.L2.forward(z1),y_batch)
        #backward pass 
        delta = self.SCE.backward(y_batch, 1)
        dw2, dx2, db2 = self.L2.backward(delta)
        dx2_r = self.relu.backward(dx2)
        dw1, dx1, db1 = self.L1.backward(dx2_r)

        #compute the change of each weights and update it
        self.w1_d = momentum * self.w1_d - learning_rate * dw1
        self.w1 += self.w1_d
        self.b1_d = momentum * self.b1_d - learning_rate * db1
        self.b1 += self.b1_d
        self.w2_d = momentum * self.w2_d - learning_rate * dw2
        self.w2 += self.w2_d
        self.b2_d = momentum * self.b2_d - learning_rate * db2
        self.b2 += self.b2_d

        regularization_loss = SCE_loss + l2_penalty/2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        acc_b = (np.round(z2) == y_batch)
        return acc_b, regularization_loss

    def evaluate(self, x, y,l2_penalty):
	# INSERT CODE for testing the network   
        z1 =  self.relu.forward(self.L1.forward(x))
        z2, val_SCE_loss = self.SCE.forward(self.L2.forward(z1),y)
        y2_prediction = np.round(z2)
        val_loss = val_SCE_loss + l2_penalty / 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        val_loss = np.round(np.mean(val_loss),2)
        val_acc = np.mean(y2_prediction == y)
        return val_loss, val_acc
# ADD other operations and data entries in MLP if needed

def normalization(x):
    x_max = np.max(x)
    x_min = np.min(x)
    x_nor = (x-x_min)/(x_max-x_min)
    return x_nor

def nor_test(x,xmax,xmin):
    x_nor = (x-xmin)/(xmax-xmin)
    return x_nor

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
	    data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='latin1')

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
	
    # Data normalization (1)
    #train_x = normalization(train_x)
    #train_y = normalization(train_y)
    #max_x = np.max(train_x)
    #min_x = np.min(train_x)
    #test_x = nor_test(test_x,max_x,min_x)
    #test_y = nor_test(test_y,max_x,min_x)

    #Data normalization (2)
    train_x = preprocessing.normalize(train_x)
    train_y = preprocessing.normalize(train_y)
    test_x = preprocessing.normalize(test_x)
    test_y = preprocessing.normalize(test_y)

    num_examples, input_dims = train_x.shape
    #(num_example, input_dims)=(10000,3072)
    #test2 => 2000 examples
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 50
 
    batch_size = 32
    num_batches = num_examples // batch_size
    # construct the model
    mlp = MLP(input_dims=3072, hidden_units=32)

    train_loss_list = []
    train_accuracy_list = []
    max_test_acc= 0.

    for epoch in range(num_epochs):
	# INSERT YOUR CODE FOR EACH EPOCH HERE
        total_loss = 0.0
        total_acc = 0.0
        for b in range(num_batches):
			# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
			# MAKE SURE TO UPDATE total_loss
            
            train_acc, train_loss = mlp.train(train_x = train_x, train_y = train_y, batch_size = batch_size, b = b)
            train_acc = train_acc * 100
            train_loss_list.append((train_loss))
            train_accuracy_list.append((train_acc))
        
        # we compute the loss and acc for "num_batches" times. We need to mean them
        total_loss = round(np.mean(train_loss_list), 2)
        total_acc = round(np.mean(train_accuracy_list), 2)
        eval_loss_out, eval_acc = mlp.evaluate(test_x,test_y,0.0)
        eval_acc = round(100 * eval_acc,2)
        max_test_acc = max (max_test_acc, eval_acc)
        print("Epoch:", epoch+1, "       Train_loss:", total_loss,"        Total_accuracy", total_acc,"%",     "       Validation_loss",eval_loss_out, "      validation_accuracy", eval_acc)

    eval_loss_out, eval_acc = mlp.evaluate(test_x,test_y,0.0)
    eval_acc = round(100 * eval_acc,2)
    print("Maximum_testing_accuracy:", max_test_acc)
    print("Testing accuracy:",eval_acc)
    