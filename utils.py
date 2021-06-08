from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from common import*
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class Utils():
    def __init__(self):
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)

    def allocate_gpu_memory(self, gpu_number=0):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        if physical_devices:
            try:
                print("Found {} GPU(s)".format(len(physical_devices)))
                tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')
                tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
                print("#{} GPU memory is allocated".format(gpu_number))
            except RuntimeError as e:
                print(e)
        else:
            print("Not enough GPU hardware devices available")

    def LoadImg(self, imgPath, show=True):
        img=load_img(imgPath,target_size=(TARGETSIZE,TARGETSIZE))
        imgTensor=img_to_array(img)
        imgTensor=np.expand_dims(imgTensor,axis=0)
        imgTensor /=255
        if show:
            plt.imshow(imgTensor[0])
            plt.axis('off')
            plt.show()
        return imgTensor