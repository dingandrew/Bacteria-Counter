##################################################################
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
###################################################################


'''
###################################################################
# Create a VideoCapture object assigned to the correct port (in this case "0")
# READ MORE: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
cap = cv2.VideoCapture(0)
cap.set(4, 28)

# Start a loop to repeatedly capture frames and display them
# READ MORE: https://docs.python.org/3/reference/compound_stmts.html#while
while (1):
	# Assign 'frame' to a captured frame from VideoCapture object created in line 9
	# 'ret' is a boolean(True/False statement) that indicates if the frame is ready to be shown
	ret, frame = cap.read()

	# Show the image frame
	# READ MORE: https://docs.opencv.org/3.1.0/dc/d2e/tutorial_py_image_display.html?highlight=imshow
	cv2.imshow('frame', frame)

	# Show the frame for 1 ms and wait to see if user pressed 'q'
	# READ MORE: https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=waitkey
	if cv2.waitKey(1) & 0xFF == ord('y'):
		# If user presses 'y', take picture exit out of the loop
		image = cv2.resize(frame, (28, 28))
		cv2.imwrite('num.png', image)
		break
		
# Make sure to release the camera
cap.release()

# Close all windows
# READ MORE: https://docs.opencv.org/3.1.0/dc/d2e/tutorial_py_image_display.html?highlight=destroyallwindows
# READ MORE: https://docs.opencv.org/3.1.0/d7/dfc/group__highgui.html#ga6b7fc1c1a8960438156912027b38f481
cv2.destroyAllWindows()
#############################################################################
'''


#############################################################################
#path = r'C:\Users\dinga\PycharmProjects\firstOpenCV\venv\Include\num.png'
path = './num.png'
image = cv2.imread(path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('numBlackWhite.png', image_gray)

plt.imshow(image_gray, cmap=plt.cm.binary)
plt.show()
print(image_gray)

cv2.fastNlMeansDenoising(image_gray, image_gray, 15)

image_gray = cv2.bitwise_not(image_gray)
plt.imshow(image_gray, cmap=plt.cm.binary)
plt.show()
print(image_gray)

for x in range(0, 28):
	for y in range(0, 28):
		if(image_gray[x][y] < 157 ):
			image_gray[x][y] = 0

cv2.imwrite('numBlackWhite.png', image_gray)

plt.imshow(image_gray, cmap=plt.cm.binary)
plt.show()
print(image_gray)

imageNorm = tf.keras.utils.normalize(image_gray, axis=1)
imageFlatNorm = np.reshape(imageNorm, -1)
print(imageFlatNorm)



##############################################################################



#############################################################################
new_model = tf.keras.models.load_model('numreader.model')

#need to change the shape of the array tob 2D
#These 2 lines are very hacky probably a better solution
a = np.zeros(shape=(1,784))
a[0] = imageFlatNorm

predictions = new_model.predict(a)

print("Probability of each digit: ", predictions)
print("Predicted number is: ",np.argmax(predictions))
#############################################################################
