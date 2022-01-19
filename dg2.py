import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Python Imaging Library (PIL) - external library adds support for image processing capabilities
from PIL import Image
import PIL.ImageOps


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

print(pd.Series(y).value_counts())

classes = ['0', '1', '2','3', '4','5', '6', '7', '8','9']
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
xtrainscaled = xtrain/255
xtestscaled = xtest/255

clf = LogisticRegression(solver= "saga", multi_class= 'multinomial').fit(xtrainscaled, ytrain)
ypred = clf.predict(xtestscaled)
acc = accuracy_score(ytest, ypred)
print(acc)

#Starting the camera
cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #Drawing a box in the center of the video
        height,width = gray.shape
        upperleft = (int(width / 2 - 56), int(height / 2 - 56))
        bottomright = (int(wdith / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upperleft, bottomright, (0,255,0), 2)
        roi = gray[upperleft[1]:bottomright[1], upperleft[0]: bottomright[0]]
        #Converting cv2 image to pil format so that the interpreter understands
        impil = Image.fromarray(roi)
        imagebw = impil.convert('L')
        iamgebwresize  = imagebw.resize( (28,28), Image.ANTIALIAS)
        imgInverted = PIL.ImageOps.invert(iamgebwresize)
        pixelfilter = 20
        #percentile() converts the values in scalar quantity
        minpixel = np.percentile(imgInverted, pixelfilter)
        
        #using clip to limit the values betwn 0-255
        imgInverted_scaled = np.clip(imgInverted - minpixel, 0, 255)
        maxpixel = np.max(imgInverted)
        imgInverted_scaled = np.asarray(imgInverted_scaled)/maxpixel
          #converting into an array() to be used in model for prediction
        testsample = np.array(imgInverted_scaled).reshape(1,784)
        testpred = clf.predict(testsample)
        
        print("Predicted class is: ", test_pred)
        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()