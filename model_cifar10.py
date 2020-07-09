from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization

def create_model():
    model = Sequential([
         Conv2D(filters=32,input_shape=(32,32,3),kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
         BatchNormalization(),
         Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
         BatchNormalization(),
         MaxPooling2D(pool_size=(2,2)),
         Dropout(0.25),
         Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
         BatchNormalization(),
         Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
         BatchNormalization(),
         MaxPooling2D(pool_size=(2,2)),
         Dropout(0.25),
         Flatten(),

         Dense(512,activation='relu'),
         Dropout(0.5),
         Dense(units=10,activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
