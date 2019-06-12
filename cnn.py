import io
import os
import pandas as pd
import numpy as np
from keras import optimizers
from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.datasets import mnist
from keras.utils import to_categorical
from matplotlib.pyplot import figure

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def datatochart():
    df = pd.read_csv(os.path.join('../Stocks/Forex', 'DAT_MT_EURUSD_M1_2018.csv'), delimiter=',',
                     usecols=['Date', 'Open', 'High', 'Low', 'Close'])

    pattern_size = 15  # 앞 데이터의 길이
    predict_size = 5  # 예측하는 뒤 데이터의 길이
    batch_size = pattern_size + predict_size  # 각 batch 의 크기
    batch_num = int(df.shape[0] / batch_size)  # batch 의 갯수

    figure(num=None, figsize=(0.1 * 16, 0.1 * 16), dpi=80, facecolor='w', edgecolor='k')  # 8 pixel per 0.1
    fo = open("label.txt", "w+")
    fo.write("index,label\n")

    highly_increasing_c = 0
    increasing_c = 0
    neutral_c = 0
    highly_decreasing_c = 0
    decreasing_c = 0

    for i in range(batch_num):
        batch_df = df.iloc[i * batch_size:(i + 1) * batch_size, 2:5]  # 각 batch 의 분리
        #print(batch_df)
        train_df = batch_df.iloc[0:pattern_size]  # train df
        # print(batch_df.shape)
        #plt.plot(range(15), train_df['High'])
        #plt.plot(range(15), train_df['Low'])
        plt.plot(range(15), train_df['Close'])
        plt.xticks([])
        plt.yticks([])

        label_df = batch_df.iloc[pattern_size:batch_size, 2]  # label df
        current_price = batch_df.iloc[pattern_size - 1, 2]  # most current price of train data
        label_mean = label_df.mean()  # mean value of our label data
        pip_value = 0.00001  # 1 pip, 20 pip = 1% increase
        value_diff = label_mean - current_price
        pip_diff = value_diff / pip_value
        #  pip_diff_sign = np.sign(pip_diff)

        # print(pip_diff)
        # 0 highly increasing 1 increasing 2 neutral 3 decreasing 4 highly decreasing
        if -10 < pip_diff and pip_diff < 10:  # neutral
            y_label = 2
            neutral_c += 1
        elif pip_diff > 30:
            y_label = 0
            highly_increasing_c += 1
        elif pip_diff > 10:
            y_label = 1
            increasing_c += 1
        elif pip_diff < -30:
            y_label = 4
            highly_decreasing_c += 1
        elif pip_diff < -10:
            y_label = 3
            decreasing_c += 1

        str = '%d,%d\n' % (i,y_label)
        fo.write(str)

        # plt.xlabel('Time(m)', fontsize=18)
        # plt.ylabel('Close Price', fontsize=18)

        plt.savefig('charts/chart%d.jpg' %i)
        print('chart%d created' %i)

        if y_label == 0 and neutral_c-highly_increasing_c > 0:
            for j in range(neutral_c-highly_increasing_c):
                plt.savefig('charts/chart%d-%d.jpg' % (i, j))
                str = '%d,%d\n' % (i, y_label)
                fo.write(str)
                print('subchart%d-%d created' % (i, j))
                highly_increasing_c += neutral_c-highly_increasing_c
        elif y_label == 1 and neutral_c-increasing_c > 0:
            for j in range(neutral_c-increasing_c):
                plt.savefig('charts/chart%d-%d.jpg' % (i, j))
                str = '%d,%d\n' % (i, y_label)
                fo.write(str)
                print('subchart%d-%d created' % (i, j))
                increasing_c += neutral_c-increasing_c
        elif y_label == 3 and neutral_c-decreasing_c > 0:
            for j in range(neutral_c-decreasing_c):
                plt.savefig('charts/chart%d-%d.jpg' % (i, j))
                str = '%d,%d\n' % (i, y_label)
                fo.write(str)
                print('subchart%d-%d created' % (i, j))
                decreasing_c += neutral_c-decreasing_c
        elif y_label == 4 and neutral_c-highly_decreasing_c > 0:
            for j in range(neutral_c-highly_decreasing_c):
                plt.savefig('charts/chart%d-%d.jpg' % (i, j))
                str = '%d,%d\n' % (i, y_label)
                fo.write(str)
                print('subchart%d-%d created' % (i, j))
                highly_decreasing_c += neutral_c-highly_decreasing_c

        #if y_label == 1 or y_label == 3: # 3배
         #   for j in range(3-1):
          #      plt.savefig('charts/chart%d-%d.jpg' %(i, j))
           #     str = '%d,%d\n' % (i,y_label)
            #    fo.write(str)
             #   print('subchart%d-%d created' %(i,j))
            #img = load_img('charts/chart%d.jpg' %i)
            #data = img_to_array(img)
            #samples = expand_dims(data, 0)
            #datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
            #it = datagen.flow(samples, batch_size = 1)

            #for i in range(9):
             #   plt.clf()
              #  batch = it.next()
               #plt.imshow(image)
                #plt.show()

       # elif y_label == 0 or y_label == 4: # 8배
        #    for j in range(8-1):
         #       plt.savefig('charts/chart%d-%d.jpg' % (i, j))
          #      str = '%d,%d\n' % (i,y_label)
           #     fo.write(str)
            #    print('subchart%d-%d created' % (i, j))


        if i%100 == 0:
            print('h_i_c: %d, i_c: %d, n_c: %d, d_c: %d, h_d_c: %d'
                  %(highly_increasing_c,increasing_c, neutral_c, decreasing_c, highly_decreasing_c))
        plt.clf()
        if i == 4500:
            break
    fo.close()

# TODO:: 앞의 15분 데이터 뽑고 뒤의 5분으로 레이블 판정 - 0 = decrease 1 = neutral 2 = increase

def y_label(file_path):
    label_df = pd.read_csv(os.path.join(file_path, 'label.txt'), delimiter=',', usecols=['label'])
    return label_df

# output: X_train, y_train, X_test, y_test (test for either validation set or accuracy)
def load_data(validation_percent):
    all_images = []
    counter = 0
    for image_path in os.listdir('charts'):
        #print(image_path)
        img = io.imread('charts/%s' % image_path, as_grey=True)
        # io.imshow(img)
        # io.show()
        img = img.reshape([128, 128, 1])
        all_images.append(img)
        counter += 1
    data = np.array(all_images)

    validation_size = int(data.__len__()*validation_percent/100)

    return data[:data.__len__()-validation_size], labels[:data.__len__()-validation_size], \
           data[data.__len__()-validation_size:], labels[data.__len__()-validation_size:]

def count_labels(labels):
    highly_increasing_c = 0
    increasing_c = 0
    neutral_c = 0
    highly_decreasing_c = 0
    decreasing_c = 0

    for i in range(labels.__len__()):
        if labels.iloc[i, 0] == 0:
            highly_increasing_c += 1
        elif labels.iloc[i, 0] == 1:
            increasing_c += 1
        elif labels.iloc[i, 0] == 2:
            neutral_c +=1
        elif labels.iloc[i, 0] == 3:
            decreasing_c += 1
        elif labels.iloc[i, 0] == 4:
            highly_decreasing_c += 1

    print('h_i_c: %d, i_c: %d, n_c: %d, d_c: %d, h_d_c: %d'
          % (highly_increasing_c, increasing_c, neutral_c, decreasing_c, highly_decreasing_c))

    print('h_i_c: %f%%, i_c: %f%%, n_c: %f%%, d_c: %f%%, h_d_c: %f%%'
          % (highly_increasing_c/labels.__len__(), increasing_c/labels.__len__(),
             neutral_c/labels.__len__(), decreasing_c/labels.__len__(), highly_decreasing_c/labels.__len__()))



#========================================MAIN====================================#

#datatochart() # 차트 생성에 사용

labels = y_label('')
#print(labels.iloc[1,0])
count_labels(labels)
#h_i_c: 1141, i_c: 3526, n_c: 9339, d_c: 3453, h_d_c: 1171 중간빼고 2.5배, 8배
#h_i_c: 0.061245%, i_c: 0.189265%, n_c: 0.501288%, d_c: 0.185346%, h_d_c: 0.062856%

X_train, y_train, X_test, y_test = load_data(20)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#count_labels(y_train)
#count_labels(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[0])

#datagen = ImageDataGenerator()

#create model
model = Sequential()

#add model layers
model.add(Conv2D(128, kernel_size=3, strides=(2,2), padding='valid', activation='relu', input_shape=(128,128,1)))
model.add(Conv2D(128, kernel_size=3, strides=(2,2), padding='valid',activation='relu'))
#model.add(Conv2D(128, kernel_size=3, strides=(2,2), padding='valid',activation='relu'))
#model.add(Conv2D(128, kernel_size=3, strides=(2,2), padding='valid',activation='relu'))

model.add(Flatten())
#model.add(Dense(12, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

#  compile model using accuracy as a measure of model performance
#  sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
ADAM = optimizers.Adam(lr= 0.001)
model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
# learning rate does not really matters
model.summary()

#train model
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)

#show predictions for the first 100 images in the test set
print(model.predict(X_test[:100]))

#show actual results for the first 100 images in the test set
print(y_test[:100])


#  Convolutional Neural Network Summary
#  CNN uses feature extracting method from the given input as RGB (N x N x 3) or Grey scaled images (N x N x 1). It uses filters to find corresponding weights which is to figure out the importance of the pixels in the images that matters.
#  There are some technics such as Padding, Strides, Pooling layers, CNN Backpropagation
#  Third party Nets such as classical Net (LeNet-5, AlexNet, VGG), newer nets ResNet, Inception Net
#  ResNet '2015 (in plain network, increasing # of layers should decrease trainning error but actaully increases because of vanishing gradient. Resblocks allows the gradient to be directly backpropagated to earlier layers and solve problem of "plain network". Resblocks learns identity function easily(the one with less important weights to be gone) -> this harms less on trainning set performance.
#  Inception Network '2014 (1x1 convolution: shrink number of channels, bottleneck layer: uses 1x1 conv before applying big fillters to reduce the computation cost. then uses bottleneck to stack up different channels of the filters. ex) 28x28x1 + 28x28x3 + 28x28x5 +...+ maxpooling layer with same padding. side branch with FCL and softmax helps that we are going in right way)
#  Data Augmentation: It should be implemented on-fly and parallel(saves memory) rotation, random cropping, mirroring, shearing, local warping, color shifting, PCA color augmentation.
#  Ensembling: train several different network independently and uses average of their outputs
#  Transfer Learning: nn + weights from other sources. 1. freeze all and just replace last layers with SM 2.precompute last frozen layer activation and convert X through all fixed layers and save to disk