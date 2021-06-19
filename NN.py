import os
import cv2
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np



def displySample(img):
    plt.figure(figsize=(0.5,0.5))
    plt.imshow(img, cmap='gray')
    plt.show()


def UploadDataset():
   data=[]
   labels=[]
   print("[INFO] loading images...")
   imagePaths = list(paths.list_images("Dataset"))
   for imagePath in imagePaths:
       label = imagePath.split(os.path.sep)[-2]
       image = load_img(imagePath)
       image = img_to_array(image)
       image = preprocess_input(image)
       data.append(image)
       labels.append(label)

   ans=[data,labels]
   return ans

def UploadDatasetForSVM_withoutpreprocessimage():
    data = []
    labels = []
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images("Dataset"))
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = load_img(imagePath)
        image = img_to_array(image)
        data.append(image)
        labels.append(label)

    ans = [data, labels]
    return ans


def Convert_To_Gray_Scale(Images):
    print("[INFO] loading Convert Images to gray scale...")
    GrayScaledData=[]
    for Image in Images:
        gray_image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        GrayScaledData.append(gray_image)
    return GrayScaledData



def Normalize_DataSet(Images):
    print("[INFO] loading Normalize Input...")
    data=[]
    for Image in Images:
        Image=Image/255.0
        data.append(Image)
    data=np.array(data)
    data=data.reshape(2059,10000)
    return data


def Preprocess_Labels(Labels):
    Labels = np.array(Labels)
    newarr = Labels.reshape(len(Labels), 1)
    newarr=newarr.astype(np.int) + 1
    ohe = OneHotEncoder()
    y_encoded = ohe.fit_transform(newarr).toarray()

    return y_encoded



def NN_Model_TwohiddenLayers():
    X,labels=UploadDataset()
    X=Convert_To_Gray_Scale(X)
    X=Normalize_DataSet(X)
    y_encoded=Preprocess_Labels(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.10,random_state=42)
    model = Sequential()
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    trained = model.fit(X_train, y_train, epochs=25, batch_size=20)
    Result=model.evaluate(X_test, y_test,verbose=0)
    print("[INFO] NN_Model_TwohiddenLayers.....")
    print("Test Loss:", Result[0])
    print("Test Accuracy", Result[1])
    result=[Result[0],Result[1]]
    return result




def NN_Model_ThreehiddenLayers():
    X,labels=UploadDataset()
    X=Convert_To_Gray_Scale(X)
    X=Normalize_DataSet(X)
    y_encoded=Preprocess_Labels(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20,random_state=42)
    model = Sequential()
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    trained = model.fit(X_train, y_train, epochs=25, batch_size=20)
    Result=model.evaluate(X_test, y_test,verbose=0)
    print("[INFO] NN_Model_ThreehiddenLayers....")
    print("Test Loss:", Result[0])
    print("Test Accuracy", Result[1])
    result = [Result[0], Result[1]]
    return result


def SVM_Model():
    X, labels = UploadDatasetForSVM_withoutpreprocessimage()
    Labels = np.array(labels)
    Labels = Labels.astype(np.int) + 1
    X = Convert_To_Gray_Scale(X)
    X = Normalize_DataSet(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Labels, test_size=0.20, random_state=42)
    svm_model_linear = SVC(kernel='linear', C=1).fit(x_train, y_train)
    y_pred = svm_model_linear.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print("[INFO] SVM_Model....")
    print("Test Accuracy:",accuracy)
    return accuracy

def compareModelAccuracy(names,results):
    print("[INFO] loading start plotting and Compare between the Models...")
    x_pos = np.arange(len(names))
    plt.bar(x_pos,results,color = ['black', 'red','blue'],width=0.2)
    plt.xticks(x_pos, names)
    plt.show()


def Run():
    model1res = NN_Model_TwohiddenLayers()
    model2res = NN_Model_ThreehiddenLayers()
    model3res = SVM_Model()
    res=[model1res[1]*100,model2res[1]*100,model3res]
    names=("NN_Ty1","NN_Ty2","SVM")
    compareModelAccuracy(names,res)


Run()