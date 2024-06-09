import cv2
from keras.models import load_model
import numpy as np
from PIL import Image as PILImage
model=load_model('BrainTumor10Epochs.h5')

image = cv2.imread('C:\\Users\\Pushkar\\PycharmProjects\\Braintumor detection\\archive\\pred\\pred0.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img=PILImage.fromarray(image)

img = img.resize((64, 64), PILImage.LANCZOS)

img=np.array(img)
img_input = np.expand_dims(img, axis=0)
#print(img)
result=model.predict(img_input)
# predicted_class = np.argmax(result, axis=1)
# print(predicted_class)
print(result)


