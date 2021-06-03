#%%
import os
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras import backend, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GaussianNoise, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.client import device_lib
from vit_keras import vit
from utils import*
import collections
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

class ThreeGroupsClassfier():
    def __init__(self):
        self.train_path = "input/train"
        self.valid_path = "input/valid"
        self.test_path = "input/test"
        self.util = Utils()
        self.util.allocate_gpu_memory()
        device_lib.list_local_devices()
        self.model=None
        self.train_generator=None 
        self.validation_generator=None 
        self.test_generator=None
        self.DataAugment()
        self.InitModel()

    def InitModel(self):
        finetune_at = 28
        vit_model = vit.vit_l32(
            image_size=TARGETSIZE,
            pretrained=True,
            include_top=False,
            pretrained_top=False)

        # fine-tuning
        for layer in vit_model.layers[:finetune_at - 1]:
            layer.trainable = False

        num_classes = len(self.validation_generator.class_indices)
        # Add GaussianNoise layer for robustness
        noise = GaussianNoise(0.01, input_shape=(TARGETSIZE, TARGETSIZE, 3))
        # Classification head
        head = Dense(num_classes, activation="softmax")
        self.model = Sequential()
        self.model.add(noise)
        self.model.add(vit_model)
        self.model.add(head)

    def DataAugment(self):
        train_datagen = ImageDataGenerator(
            rescale=1/255,
            horizontal_flip=True,
            rotation_range=15,
            zoom_range=0.1)
        valid_datagen = ImageDataGenerator(rescale=1/255)
        test_datagen  = ImageDataGenerator(rescale=1/255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(TARGETSIZE, TARGETSIZE),
            batch_size=4,
            color_mode='rgb',
            class_mode='sparse',
            shuffle=True)

        self.validation_generator = valid_datagen.flow_from_directory(
            self.valid_path,
            target_size=(TARGETSIZE, TARGETSIZE),
            batch_size=4,
            color_mode='rgb',
            class_mode='sparse')

        self.test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(TARGETSIZE, TARGETSIZE),
            batch_size=4,
            color_mode='rgb',
            class_mode='sparse')
    
    def Train(self):
        backend.clear_session()
        # vit_model = vit.vit_l32(
        #     image_size=TARGETSIZE,
        #     pretrained=True,
        #     include_top=False,
        #     pretrained_top=False)
        # print(len(vit_model.layers))
        # print(vit_model.layers)

        reduceLr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            verbose=1,
            mode='max',
            min_lr=0.0001)

        filepath="model_{epoch:02d}-{val_loss:.2f}.hdf5"
        save_dir = "/home/chenhsi/Projects/Xihelm/models"
        checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), 
                                        monitor='val_loss',#val_loss
                                        verbose=1,
                                        save_best_only=False, 
                                        save_weights_only=False)

        # finetune_at = 28
        # # fine-tuning
        # for layer in vit_model.layers[:finetune_at - 1]:
        #     layer.trainable = False
        # num_classes = len(validation_generator.class_indices)
        # # Add GaussianNoise layer for robustness
        # noise = GaussianNoise(0.01, input_shape=(TARGETSIZE, TARGETSIZE, 3))
        # # Classification head
        # head = Dense(num_classes, activation="softmax")
        # model = Sequential()
        # model.add(noise)
        # model.add(vit_model)
        # model.add(head)
        self.model.compile(optimizer=optimizers.Adam(),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])                      
        history = self.model.fit(
                                self.train_generator,
                                epochs=40,
                                validation_data=self.validation_generator,
                                verbose=1, 
                                shuffle=True,
                                # callbacks=[
                                #             reduceLr,
                                #             checkpoint]
                                callbacks=None
                                )
        #save model
        # model.save('/home/chenhsi/Projects/Xihelm/myModel.h5')
        # model.save_weights('/home/chenhsi/Projects/Xihelm/myModel_weights.h5')
        # modelSave=keras.models.load_model('/home/chenhsi/Projects/Xihelm/models/myModel')
        # model.load_weights('/home/chenhsi/Projects/Xihelm/models/model_02-0.96.hdf5')

        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        acc_values = history_dict["accuracy"]
        val_acc_values = history_dict["val_accuracy"]
        epochs = range(1, len(history_dict["accuracy"]) + 1)

        plt.plot(epochs, loss_values, "bo", label="train")
        plt.plot(epochs, val_loss_values, "b", label="valid")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(epochs, acc_values, "bo", label="train")
        plt.plot(epochs, val_acc_values, "b", label="valid")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        #predict image groups
        self.PredictGroups()

        print("best val_acc:", np.max(val_acc_values), "epoch:", np.argmax(val_acc_values))
        print("best val_loss:", np.min(val_loss_values), "epoch:", np.argmin(val_loss_values))
        test_loss, test_acc = self.model.evaluate(self.test_generator)
        print("Test Accuracy:", test_acc)
        print("==end==")

    def PredictGroups(self):
        folders=[
            '/home/chenhsi/Projects/Xihelm/input/test/animals',
            '/home/chenhsi/Projects/Xihelm/input/test/buildings',
            '/home/chenhsi/Projects/Xihelm/input/test/landscapes'
        ]
        for folder in folders:
            print("this folder: ",folder)
            arr=[]
            GrpTable={}
            for imgFile in os.listdir(folder):
                img = folder+"/"+imgFile
                imgTensor=self.util.LoadImg(img)
                pred=self.model.predict(imgTensor)
                classes = self.model.predict_classes(imgTensor)
                # print(pred)
                # print(classes)
                arr.append(classes[0])
                print("this is: ",detectionGroups[classes[0]])
            counter=collections.Counter(arr)
            for groupInd in counter:
                GrpTable[detectionGroups[groupInd]]=counter[groupInd]
            print(GrpTable)
    def Evaluate(self, filePath):
        self.InitModel()
        self.model.load_weights(filePath)
        #prediction
        self.PredictGroups()

