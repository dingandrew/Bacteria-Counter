'''
We need to import these libraries to examine and manipulate our data, and to 
create our model.

tensorflow: a machine learning and deep learning library that contains keras
keras: a very high level library that allows us to build a model with few lines
       of code
matplotlib.pyplot: a graphing library that we are using to display images from
		   the dataset
numpy: library that supports large multi-dimensional arrays and matrices and 
	many high level mathematical functions
'''
#############################################################
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
print(tf.VERSION)
#############################################################


'''
The mnist dataset contains images of handwritten digits(0-9). These images 
contain a variety of different handwriting stlyes and sizes. This is a classic 
machine learning problem because we want to teach our model to recognize a 
number regardless of how it was written.

Each image in the dataset is 28 X 28 pixels which is represented by a 2 dimensional
array. There are 70,000 images in this dataset. We want to split the dataset
into a training and test sets. We test our models accuracy with a seperate test set
to make sure that our model has not just "memorized" or overfit to our traning 
data. The training set contains 60,000 images and the test set contains 10,000
images. Each set contains an x_ and y_ ,the x_ contains the actual image data and 
the y_ contains the actual number value for the coresponidng image. x_train,
y_train, x_test, y_test, store the data inside an array.

Next we want to normalize our image data. All images are black and white so each
value in the 2-dimensional array is between 0 - 255. To normalize data means that
we are shifting these values to be between 0 - 1, this makes training the model
much more effective. 

'''
#############################################################
mnist = tf.keras.datasets.mnist #load datatset into variable mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data() #seperate the data

#show what the data looks like in the x any y training sets
#what happens when you dont use cmap=plt.cm.binary
print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print(y_train[0])

#show what the data looks like in the x any y test sets
print(x_test[0])
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
print(y_test[0])

#normalizing the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#show what normalized data looks like
print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
##############################################################

'''
Currently our images are stored by a 2 dimensional array but our neural network
can only handle 1-dimensional input. So we have have to reshape our image arrays
into a 1-dimensional array.
'''
###############################################################
#flatten our image arrays
x_trainflat = x_train.reshape(x_train.shape[0], -1)
x_testflat = x_test.reshape(x_test.shape[0], -1)

print(x_trainflat[1])
################################################################

'''
First we intialize a sequential model which is a type of nerual network where
we have layers of nodes starting with an input layer and ending with a output 
layer. Models with many layers are called deep neural networks. Each of our 
layers are dense layers which mean every node is connected to every node in the 
next and previous layers. To create a layer we need to pass in the number of nodes
and an activation function. A activation function is essentially the function that
determines if that node will fire given some input data. 
'''
##################################################################
# intialize the model
model = tf.keras.models.Sequential() 

#adding layers to our neural network
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#build the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
	                    metrics=['accuracy'])

#train the model with data
model.fit(x_trainflat, y_train, epochs=5)

print(model.summary())
#################################################################

'''
This will tell us the accuracy and loss of our model for our testing data

Try changing the amount of nodes in the input and middle layers, the number of
epochs, and try training the model without normalizing the data. Notice how
these variables affect the accuracy. 
'''
#################################################################
val_loss, val_acc = model.evaluate(x_testflat, y_test)
print(val_loss)
print(val_acc)
#################################################################

'''
Change the imageIndx to see if the model is correct for different images
'''
##################################################################
predictions = model.predict(x_testflat)

#the index of the image we want to test
imageIndx = 1

print(predictions)
print(predictions[imageIndx])
print(np.argmax(predictions[imageIndx]))

plt.imshow(x_test[imageIndx])
plt.show()
###################################################################

'''
Saves the model as .hd5h file format in the current directory
'''
#################################################################
model.save('numreader.model')
###############################################################3

