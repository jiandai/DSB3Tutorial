'''
forked from
https://github.com/booz-allen-hamilton/DSB3Tutorial
as
https://github.com/jiandai/DSB3Tutorial
hacked ver 20170323 by jian:
    - rewire the input/output from LUNA_segment_lung_ROI.py 
ver 20170330 by jian: use LUNA data to train
ver 20170331 by jian: dice coeff <.4 for 25 epoches, load weight and +105 epoches
ver 20170401 by jian: debug preprocessing to create tr and test
ver 20170402 by jian: rerun using v2 tr/test
ver 20170403 by jian: notice the training overshot on epoch 2 on job 709016:
Epoch 1/2
1628s - loss: -6.1547e-03 - dice_coef: 0.0062
Epoch 2/2
1626s - loss: -8.6933e-04 - dice_coef: 8.6933e-04
then job 709118 become useless

save unet-v2.hdf5 as unet-v2-2.hdf5 to rerun, Unstable as well on 722727
ver 20170405.1 by jian: common feature of 709016, 722727, and the new 764516 
np.sum(np.sum(np.sum(imgs_mask_test_true ,axis=3),axis=2),axis=1).min() gives ~87.7 so smooth~1 is ok
ver 20170405.2 by jian: test SGD, crashed on 8/14 + previous epch

to-do:
separate tr from predict
'''



#from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

#N_EPOCH = 14 on 1.0e-2 + 14 on 5.0e-3 + 4 on 1.0e-3 + 3 on 5.0e-4 + 2 on 4.0e-4 + 2 on 3.0e-4 + 2 on 2.0e-4 + 2 on lr=1.0e-4
#N_EPOCH = 48 +2+2

working_path = "../../../../../luna16/processed/"
#working_path = "./"


img_rows = 512
img_cols = 512

smooth = 1.


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

    model.compile(
        #optimizer=Adam(lr=1.0e-5), 
        #optimizer=SGD(lr=1.0e-4),
        #optimizer=SGD(lr=2.0e-4),
        #optimizer=SGD(lr=3.0e-4),
        #optimizer=SGD(lr=4.0e-4),
        #optimizer=SGD(lr=5.0e-4),
        #optimizer=SGD(lr=1.0e-3),
        #optimizer=SGD(lr=5.0e-3),
        optimizer=SGD(lr=1.0e-2),
        loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    #imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
    imgs_train = np.load(working_path+"trainImages-v2.npy").astype(np.float32)
    #imgs_train = imgs_train[:50]
    #print imgs_train.shape #(2843, 1, 512, 512)
    #imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path+"trainMasks-v2.npy").astype(np.float32)
    #imgs_mask_train = imgs_mask_train[:50]
    #print imgs_mask_train.shape #(2843, 1, 512, 512)
    #imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
    imgs_test = np.load(working_path+"testImages-v2.npy").astype(np.float32)
    imgs_test = imgs_test[:10]
    #print imgs_test.shape #(710, 1, 512, 512)
    #imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path+"testMasks-v2.npy").astype(np.float32)
    imgs_mask_test_true = imgs_mask_test_true[:10]
    #print imgs_mask_test_true.shape #(710, 1, 512, 512)
    
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std

    # Added normalization for test set as well
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    imgs_test -= mean  # images should already be standardized, but just in case
    imgs_test /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Saving weights to unet.hdf5 at checkpoints
    #model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    model_checkpoint = ModelCheckpoint('unet-v2.hdf5', monitor='loss', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        #model.load_weights('../unet.hdf5') # modify the path
        #model.load_weights('./unet.hdf5') # modify the path
        model.load_weights('./unet-v2.hdf5') # modify the path
        
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training. 
    #
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=N_EPOCH, verbose=1, shuffle=True, callbacks=[model_checkpoint])






    # loading best weights from training session
    #model.load_weights('./unet.hdf5')
    #model.load_weights('./unet-v2.hdf5')

    num_test = len(imgs_test)
    print ('# in test set',num_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    mean = 0.0
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
        dice=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
        print(i,dice)
        mean+=dice
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
    #np.save('masksTestPredicted-v2.npy', imgs_mask_test)

if __name__ == '__main__':
    #train_and_predict(False)
    train_and_predict(True)
