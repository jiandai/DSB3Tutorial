'''
forked from
https://github.com/booz-allen-hamilton/DSB3Tutorial
as
https://github.com/jiandai/DSB3Tutorial
hacked ver 20170323 by jian:
    - rewire the input/output from LUNA_segment_lung_ROI.py 
ver 20170330 by jian: use LUNA data to train
ver 20170331 by jian: dice coeff <.4 for 25 epoches, load weight and +105 epoches
ver 20170401sat by jian: debug preprocessing to create tr and test
ver 20170402sun.1 by jian: rerun using newer version of tr/test
ver 20170402sun.2 by jian: fork an experimental version, use v3 preproc, try multi-gpu
ver 20170403mon.1 by jian: fork an experimental version, use v3 preproc, use ref, not much diff in 20 epoch
https://www.kaggle.com/daij1492/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
ver 20170403mon.2 by jian: small sample test, /w batch size 4 OOM, shoot out on Epoch 41/100
ver 20170405wed by jian: revamp for optimization and whole tr set, 5150s / epch, loss not move
ver 20170406thu.1 by jian: fork from LUNA_train_unet_exp_archi.py for batch training
ver 20170406thu.2 by jian: fork for M60
ver 20170407fri by jian: modify loss, /w 1-6, 4 is the best

to-do:
review per-tr/tt normalization
'''

#from __future__ import print_function

import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers.core import Lambda
import tensorflow as tf

############################################################################################
# M60 test: batch number : 1-6
batch_char = sys.argv[1]
#batch_char = '1'
print('batch number = ',batch_char)
batch_num = int(batch_char)

working_path = "../../../../../luna16/processed/"
img_rows = 512
img_cols = 512
#smooth = 1.
smooth = 0.



TR_START=(batch_num-1)*500
TR_END=batch_num*500 #2843

TEST_START=0
TEST_END=40


#N_EPOCH = #20 #6
N_EPOCH = 260 #(~45 hr=>4/9/8pm for batch_num=4)+52

ngpus=1
batch_size =2*ngpus
learning_rate=(batch_num*2.0+1.0)*1.0e-6
#decay=.01
decay=.001

print 'Number of epoches = ',str(N_EPOCH)
print 'batch_size=',str(batch_size)

print 'learning_rate=',str(learning_rate)
print 'decay=',str(decay)


use_existing=True
#use_existing=False
print 'use_existing=',str(use_existing)

h5file = 'unet-m60-btch2'+batch_char+'.hdf5'
prediction_file = 'masksTestPredicted-m60-btch2'+batch_char+'.npy'






optr = Adam(lr=learning_rate,decay=decay) 
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
############################################################################################





def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [ shape[:1] // parts, shape[1:] ])
        stride = tf.concat(0, [ shape[:1] // parts, shape[1:]*0 ])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        return Model(input=model.inputs, output=merged)




def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
	inputs = Input((1, 512, 512))
	conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
	conv1 = Dropout(0.2)(conv1)
	conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
	conv2 = Dropout(0.2)(conv2)
	conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
	conv3 = Dropout(0.2)(conv3)
	conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
	conv4 = Dropout(0.2)(conv4)
	conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
	conv5 = Dropout(0.2)(conv5)
	conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

	up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
	conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
	conv6 = Dropout(0.2)(conv6)
	conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

	up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
	conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
	conv7 = Dropout(0.2)(conv7)
	conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

	up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
	conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
	conv8 = Dropout(0.2)(conv8)
	conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

	up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
	conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
	conv9 = Dropout(0.2)(conv9)
	conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

	conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

	model = Model(input=inputs, output=conv10)
	model.summary()
	model.compile(
		#optimizer=Adam(lr=1e-5), 
		#optimizer=Adam(lr=1e-6), 
		optimizer=optr,
		loss=dice_coef_loss, metrics=[dice_coef])

	return model



'''
#def unet_model():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    #model = make_parallel(model, ngpus)
    model.compile(optr, loss=dice_coef_loss, metrics=[dice_coef])

    return model
'''


def train_and_predict(use_existing):
    imgs_train = np.load(working_path+"trainImages-v3.npy").astype(np.float32) #2843
    imgs_train = imgs_train[TR_START:TR_END]
    #print imgs_train.shape #(2843, 1, 512, 512)

    imgs_mask_train = np.load(working_path+"trainMasks-v3.npy").astype(np.float32)
    imgs_mask_train = imgs_mask_train[TR_START:TR_END]
    #print imgs_mask_train.shape #(2843, 1, 512, 512)

    imgs_test = np.load(working_path+"testImages-v3.npy").astype(np.float32)
    imgs_test = imgs_test[TEST_START:TEST_END]
    #print imgs_test.shape #(710, 1, 512, 512)

    imgs_mask_test_true = np.load(working_path+"testMasks-v3.npy").astype(np.float32)
    imgs_mask_test_true = imgs_mask_test_true[TEST_START:TEST_END]
    #print imgs_mask_test_true.shape #(710, 1, 512, 512)

  
    mskLst = []
    imgLst = []
    for j in range(imgs_mask_train.shape[0]):
        if np.sum(imgs_mask_train[j].flatten())>0:
            mskLst.append(imgs_mask_train[j])
            imgLst.append(imgs_train[j])
    imgs_mask_train = np.stack(mskLst)
    imgs_train = np.stack(imgLst)
    print imgs_mask_train.shape
    print imgs_train.shape

    mskLst = []
    imgLst = []
    for j in range(imgs_mask_test_true.shape[0]):
        if np.sum(imgs_mask_test_true[j].flatten())>0:
            mskLst.append(imgs_mask_test_true[j])
            imgLst.append(imgs_test[j])
    imgs_mask_test_true = np.stack(mskLst)
    imgs_test = np.stack(imgLst)
    print imgs_mask_test_true.shape
    print imgs_test.shape



    # To be reviewed
    print np.histogram(imgs_train)
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std
    print np.histogram(imgs_train)

    print np.histogram(imgs_mask_train)

    print np.histogram(imgs_test)
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    imgs_test -= mean  # images should already be standardized, but just in case
    imgs_test /= std
    print np.histogram(imgs_test)





    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint(h5file, monitor='loss', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights(h5file) # modify the path
        
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training. 

    model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, nb_epoch=N_EPOCH, verbose=2, shuffle=True, callbacks=[model_checkpoint])


    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    num_test = len(imgs_test)
    print ('# in test set',num_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    mean = 0.0
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
        dice_c=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
        print i,dice_c
        mean+=dice_c
    np.save(prediction_file, imgs_mask_test)
    mean/=num_test
    print("Mean Dice Coeff : ",mean)

if __name__ == '__main__':
    train_and_predict(use_existing = use_existing)
