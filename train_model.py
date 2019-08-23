import keras
import mnist_reader
import time, argparse, pdb
import matplotlib.pyplot as plt
import numpy as np

from keras.optimizers import SGD, Adam
from models import CNN2, CNN2_dropout, CNN3_dropout, reduced_all_cnn, model5


def data_preparation():
    
    #Reading the data
    x_train, y_train = mnist_reader.load_mnist('data/', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/', kind='t10k')
    
    #Recover dimensions
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    
    #Convert to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    #Normalize inputs from [0, 255] to [0, 1]
    x_train = x_train / 255
    x_test = x_test / 255
    
    #Convert class vectors to binary class matrices ("one hot encoding")
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    return x_train, y_train, x_test, y_test


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Fashion items classification using Fashion-MNIST dataset.')
    parser.add_argument('--model',default='model5', type=str)
    args = parser.parse_args()
    return args


def output_model(args, num_classes):
    if args.model == 'CNN2':
        model = CNN2(num_classes)
        num_epochs = 35
        batch = 256
    elif args.model == 'CNN2_dropout':
        model = CNN2_dropout(num_classes)
        num_epochs = 50
        batch = 256
    elif args.model == 'CNN3_dropout':
        model = CNN3_dropout(num_classes)
        num_epochs = 50
        batch = 64
    elif args.model == 'reduced_all_cnn':
        model = reduced_all_cnn(num_classes)
        num_epochs = 40
        batch = 64
    elif args.model == 'model5':
        model = model5(num_classes)
        num_epochs = 10
        batch = 256
    else:
        print('Error: not a valid model name')
    return model, num_epochs, batch


def make_plots(history, history2 = None):
    if history2:
        x1 = np.linspace(0,9,10)
        y1 = history.history['acc']
        plt.plot(x1,y1, c = 'lightblue', label = 'train lr= 0.01')
        y2 = history.history['val_acc']
        x2 = np.linspace(0,9,10)
        plt.plot(x2,y2, c = 'orange',  label = 'test lr= 0.01')
        x3 = np.linspace(9,30,30)
        y3 = history2.history['acc']
        plt.plot(x3,y3, c = 'green', label = 'train lr= 0.001')
        y4 = history2.history['val_acc']
        x4 = np.linspace(9,30,30)
        plt.plot(x4,y4, c = 'purple', label = 'test lr= 0.001')

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.xlim((0,30))
        plt.show()

        x1 = np.linspace(0,9,10)
        y1 = history.history['loss']
        plt.plot(x1,y1, c = 'lightblue', label = 'train lr= 0.01')
        y2 = history.history['val_loss']
        x2 = np.linspace(0,9,10)
        plt.plot(x2,y2, c = 'orange',  label = 'test lr= 0.01')
        x3 = np.linspace(9,30,30)
        y3 = history2.history['loss']
        plt.plot(x3,y3, c = 'green', label = 'train lr= 0.001')
        y4 = history2.history['val_loss']
        x4 = np.linspace(9,30,30)
        plt.plot(x4,y4, c = 'purple', label = 'test lr= 0.001')

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.xlim((0,30))
        plt.show()

    else:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc = 'upper left')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc = 'upper left')
        plt.show()


def main():
    
    #Initialize timer
    start_time = time.time()
    
    #Parse arguments
    args = parse_args()
    
    #Prepares the data
    x_train, y_train, x_test, y_test = data_preparation()

    #Define num classes
    num_classes = y_train.shape[1]
    
    #Build model and get params
    model, num_epochs, batch = output_model(args, num_classes)

    #Train the model
    if args.model == 'model5':
        
        #Set initial learning rate
        model.optimizer.lr=0.01
        history = model.fit(x_train,y_train, validation_data= (x_test, y_test), epochs=num_epochs, batch_size=batch, shuffle = True)
        
        #Update learning rate
        model.optimizer =Adam(lr = 0.001)
        history2 = model.fit(x_train, y_train, validation_data= (x_test, y_test), epochs=20, batch_size=1024, shuffle = True)

    else:
        
        #Fit the model compute the scores and print the model accuracy
        history = model.fit(x_train, y_train, validation_data= (x_test, y_test), epochs=num_epochs, batch_size=batch, shuffle = True)
    
    #Print running time
    print('Model took {0} s to run'.format(str(time.time()-start_time)))
    
    #Compute accuracy of the model
    scores = model.evaluate(x_test, y_test)
    print('Class accuracy: '+str(scores[1]*100))


    #Save model
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
    #Save weights
    model.save_weights("model_weights.h5")

    #Make plots
    if args.model == 'model5':
        make_plots(history, history2)
    else:
        make_plots(history)



if __name__ == '__main__':
    main()

