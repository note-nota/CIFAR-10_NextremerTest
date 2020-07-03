from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

def create_model():
    model = Sequential([
         Conv2D(filters=32,input_shape=(32,32,3),kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
         Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
         MaxPooling2D(pool_size=(2,2)),
         Dropout(0.25),
         Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
         Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
         MaxPooling2D(pool_size=(2,2)),
         Dropout(0.25),
         # Dense層に渡すために、多次元配列を2次元配列に変換する
         Flatten(),

         # 全結合層を追加する
         Dense(512,activation='relu'),
         Dropout(0.5),
         # 出力層は10種類、softmax関数を設定することで、確率が最も高いもののみが発火するようにする
         Dense(units=10,activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
