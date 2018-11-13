from keras.models import load_model
model = load_model('models/mnistCNN.h5')

from PIL import Image
import numpy as np

def main():
    print("Enter the image file name : ")
    index=input()
    img = Image.open('data/' + str(index) + '.png').convert("L")
    img = img.resize((28,28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)
    # Predicting the Test set results
    y_pred = model.predict(im2arr)
    # print(y_pred)
    answer=y_pred[0]
    for i in range(len(answer)):
        if(answer[i]==1):
            print(i)
main()