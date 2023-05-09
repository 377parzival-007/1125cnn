import keras
from keras.datasets import mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(train_X,train_Y), (test_X,test_Y) = keras.datasets.mnist.load_data()

train_X = train_X.reshape(-1, 28,28,1)
test_X = test_X.reshape(-1, 28,28,1)

print(train_X[7, 6, 10,0])

train_X[7, 6, 10,0] = 138

print(train_X[7, 6, 10,0])
print(train_X.shape)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255

print(train_X[7, 6, 10])

plt.imshow(test_X[0].reshape(28, 28), cmap = plt.cm.binary)
plt.show()

threshold = .5

# for k in range(len(train_X)):
for k in range(5):
    image = train_X[k]
    if k%1000==0:
        print(k)
    killables = []
    for i in range(1,27):
        for j in range(1,27):
            goodneighbors = 0
            if image[i,j,0] > threshold:
                goodneighbors += 1
            else:
                continue
            if image[i-1,j,0] > threshold:
                goodneighbors += 1
            if image[i+1,j,0] > threshold:
                goodneighbors += 1
            if image[i+1,j-1,0] > threshold:
                goodneighbors += 1
            if image[i+1,j+1,0] > threshold:
                goodneighbors += 1
            if image[i-1,j+1,0] > threshold:
                goodneighbors += 1
            if image[i-1,j-1,0] > threshold:
                goodneighbors += 1
            if image[i,j-1,0] > threshold:
                goodneighbors += 1
            if image[i,j+1,0] > threshold:
                goodneighbors += 1
            if goodneighbors < 4:
                continue
            killables.append([i,j])
    if(len(killables)==0):
        continue
    tokills = np.random.randint(0,len(killables),3)
    for tokill in tokills:
        [i, j] = killables[tokill]
        image[i,j,0] = 0
        image[i-1,j,0] = 0
        image[i+1,j,0] = 0
        image[i+1,j-1,0] = 0
        image[i+1,j+1,0] = 0
        image[i-1,j+1,0] = 0
        image[i-1,j-1,0] = 0
        image[i,j-1,0] = 0
        image[i,j+1,0] = 0

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=5)

test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
print('Test loss', test_loss)
print('Test accuracy', test_acc)

predictions = model.predict(test_X)
print(np.argmax(np.round(predictions[0])))


plt.imshow(train_X[0].reshape(28, 28), cmap = plt.cm.binary)
plt.show()


plt.imshow(train_X[1].reshape(28, 28), cmap = plt.cm.binary)
plt.show()

np.savetxt('data.csv', train_X.flatten(), delimiter=',')

while True:
    gg = int(input('Input a number to view from training data:'))

    plt.imshow(train_X[gg].reshape(28, 28), cmap = plt.cm.binary)
    plt.show()

    bb = int(input('Input a number to see prediction for test data:'))

    predictions = model.predict(test_X)
    print(predictions[bb])
    print(np.argmax(np.round(predictions[bb])))

    plt.imshow(test_X[bb].reshape(28, 28), cmap = plt.cm.binary)
    plt.show()

    answer = input('Do you want to continue?:')
    if answer.lower().startswith("y"):
        print("ok, continue then")
    elif answer.lower().startswith("n"):
        print("ok, bye")
        exit()