# TM002!
![스크린샷 2025-03-31 165637](https://github.com/user-attachments/assets/25b4d4f0-6e26-44e8-a9d5-ac51a77e4731)

from keras.models import load_model  # TensorFlow is required for Keras to work  
import cv2   
#Install opencv-python  
import numpy as np  

#Disable scientific notation for clarity
np.set_printoptions(suppress=True)

#Load the model
model = load_model("model/keras_Model.h5", compile=False)

#Load the labels
class_names = open("model/labels.txt", "r").readlines()

#CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    #Grab the webcamera's image.
    ret, image = camera.read()

    #Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    #Create a copy for prediction
    image_to_predict = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image_to_predict = (image_to_predict / 127.5) - 1

    #Predicts the model
    prediction = model.predict(image_to_predict)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    #Add text to the image
    text = f"Class: {class_name[2:].strip()}"
    score = f"Confidence: {str(np.round(confidence_score * 100))[:-2]}%"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, score, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    #Show the image in a window
    cv2.imshow("Webcam Image", image)

    #Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
