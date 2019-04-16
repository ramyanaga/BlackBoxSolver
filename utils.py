from keras.models import Model, model_from_json, Sequential
from PIL import Image

import tensorflow as tf
import os
import numpy as np
import gzip
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data
'''
#Based off Utils.extract_data from setup_mnist

'''
def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)
'''

def get_image_name(dirname="Images"):
    filesList = os.listdir(dirname)
    

'''
def load_image(image_id, dirname):
    image_name = os.listdir(dirname)[image_id]
    print("image_name", image_name)
    fpath = os.path.join(dirname, image_name)
    #im = Image.open(fpath)
    im = plt.imread(fpath)
    print("im type: ", type(im))
    print("imread return: ", im)
    print("image size: ", im.size)
    print("img shape: ", im.shape)
    #f = open(fpath, "rb").read()
    #print("f type: ", type(f))
    #f = bytes(bytearray(f))
    size = 32*32*3+1
   
    #labels = []
    #images = []
    #arr = np.fromstring(f[0*size:(0+1)*size],dtype=np.uint8)
    #arr = np.fromstring(im[0*size:(0+1)*size],dtype=np.uint8)
    #lab = np.identity(10)[arr[0]]
    print("arr[1:] shape: ",np.shape(arr[1:]))
    #img = arr[1:].reshape((3,32,32)).transpose((1,2,0))
    #labels.append(lab)
    img = img.reshape((3, 32, 32).transpose(1, 2, 0))
    #images.append((img/255)-.5)
    img = (img/255) - .5
    # TODO: FIND WAY TO CALCULATE LABEL FOR IMAGE
    return img
    #return np.array(images),np.array(labels)
'''

'''
def load_image(image_id, dirname):
    image_name = os.listdir(dirname)[image_id] 
    print("image_name", image_name)
    fpath = os.path.join(dirname, image_name)
    im = plt.imread(fpath)
    print("IMG BEFORE RESHAPE")
    print(im)
    img = im.reshape((3, 32, 32)).transpose((1, 2, 0))
    img = (img/255) - .5
    print("IMG: ")
    print(img)
    return img
'''

def load_image(image_id, dirname):
    image_name = os.listdir(dirname)[image_id]
    fpath = os.path.join(dirname, image_name)
    im = Image.open(fpath)
    r, g, b = im.split()
    #print("image size: ", im.size)
    #print("rgb channels: ", r, g, b)
    #arr = np.array([0 for i in range(3072)], dtype=np.uint8)
    arr = []
    for i in range(32):
        for j in range(32):
            #np.append(arr, r.getpixel((i, j)))
            arr.append(r.getpixel((i, j)))

    for i in range(32):
        for j in range(32):
            #np.append(arr, g.getpixel((i, j)))
            arr.append(g.getpixel((i, j)))

    for i in range(32):
        for j in range(32):
            #np.append(arr, b.getpixel((i, j)))
            arr.append(b.getpixel((i, j)))

    arr = np.array(arr)
    img = arr[0:].reshape((3, 32, 32)).transpose((1, 2, 0))
    #print(img)
    #print((img/255) - .5)
    return ((img/255) - .5) 

'''
def load_image(image_id, dirname):
    image_name = os.listdir(dirname)[image_id]
    print("image_name", image_name)
    image_path = os.path.join(dirname, image_name)
    image = Image.open(image_path)
    return image
'''

def model_prediction(model, inputs):
    prob = model.model.predict(inputs)
    predicted_class = np.argmax(prob)
    prob_str = np.array2string(prob).replace('\n', '')
    return prob, predicted_class, prob_str
    
        
def generate_data(data, id, target_label):
    inputs = []
    target_vec = []

    inputs.append(data.test_data[id])
    target_vec.append(np.eye(data.test_labels.shape[1])[target_label])

    inputs = np.array(inputs)
    target_vec = np.array(target_vec)

    return inputs, target_vec

