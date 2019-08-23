import keras
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, add, GlobalAveragePooling2D, Activation


def CNN2(num_classes):
    input_image = Input(shape=(28, 28, 1))
    
    ## first layer
    x1_batch = BatchNormalization()(input_image)
    x1_conv = Conv2D(32, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform', padding = 'same')(x1_batch)
    x1_add = add([x1_batch, x1_conv])
    x1_pool = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x1_add)
                         
                         
    ## second layer
    x2_batch = BatchNormalization()(x1_pool)
    x2_conv = Conv2D(32, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform',padding = 'same')(x2_batch)
    x2_add = add([x2_batch, x2_conv])
    x2_pool = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x2_add)
    

     ## ouput layer
    x3_batch = BatchNormalization()(x2_pool)
     
    x = Flatten()(x3_batch)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
     
    model = Model(input_image, preds)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.summary()
    
    return model



def CNN2_dropout(num_classes):
    input_image = Input(shape=(28, 28, 1))
    
    ## first layer
    x1_batch = BatchNormalization()(input_image)
    x1_conv = Conv2D(64, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform',padding = 'same')(x1_batch)
    x1_add = add([x1_batch, x1_conv])
    x1_pool = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x1_add)
     
     
    ## second layer
    x2_batch = BatchNormalization()(x1_pool)
    x2_conv = Conv2D(64, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform', padding = 'same')(x2_batch)
    x2_dp = Dropout(0.3) (x2_conv)
    x2_add = add([x2_batch, x2_dp])
    x2_pool = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x2_add)
     
     
    ## ouput layer
    x3_batch = BatchNormalization()(x2_pool)
     
    x = Flatten()(x4_batch)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.7)(x)
    preds = Dense(num_classes, activation='softmax')(x)
     
    model = Model(input_image, preds)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.summary()
    
    return model



def CNN3_dropout(num_classes):
    input_image = Input(shape=(28, 28, 1))
    
    ## first layer
    x1_batch = BatchNormalization()(input_image)
    x1_conv = Conv2D(64, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform',padding = 'same')(x1_batch)
    x1_add = add([x1_batch, x1_conv])
    x1_pool = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x1_add)
     
     
    ## second layer
    x2_batch = BatchNormalization()(x1_pool)
    x2_conv = Conv2D(64, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform',padding = 'same')(x2_batch)
    x2_dp = Dropout(0.3) (x2_conv)
    x2_add = add([x2_batch, x2_dp])
    x2_pool = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x2_add)
     
     
    ## ouput layer
    x3_batch = BatchNormalization()(x2_pool)
     
    x = Flatten()(x3_batch)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.7)(x)
    preds = Dense(num_classes, activation='softmax')(x)
     
    model = Model(input_image, preds)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.summary()
    
    return model



def reduced_all_cnn(num_classes):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding = 'same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
    model.add(Conv2D(32, (1, 1), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(64, (3, 3), padding = 'same', activation='relu'))
    model.add(Conv2D(64, (1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(num_classes, (1, 1), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.summary()
    
    return model



def model5(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28, 1)))
    model.add(BatchNormalization(axis=1))
    
    model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
    model.add(BatchNormalization(axis=1))
    
    model.add(Dense(128, activation = 'softmax'))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(loss ='categorical_crossentropy', optimizer =  'adam', metrics =['accuracy'])
    #model.summary()
    
    return model
