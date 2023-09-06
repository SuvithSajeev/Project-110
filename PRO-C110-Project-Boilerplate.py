# import the opencv library
import cv2
import tensorflow as tf
import numpy as np

# define a video capture object
camera = cv2.VideoCapture(0)

model = tf.keras.models.load_model("Projects/Project-110/rock-paper-scissor_keras_model.h5")


while(True):
    
    # Capture the video frame by frame
    status, frame = camera.read()
    frame = cv2.flip(frame,1)
    img = cv2.resize(frame,(224,224))
    testimg = np.array(img,dtype = np.float32)
    testimg = np.expand_dims(testimg,axis = 0)
    normalizeimg = testimg / 255
  
    prediction = model.predict(normalizeimg)
    print("Prediction: ",prediction)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
camera.release()

# Destroy all the windows
cv2.destroyAllWindows()