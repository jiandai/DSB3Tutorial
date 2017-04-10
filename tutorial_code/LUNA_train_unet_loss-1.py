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
ver 20170402 by jian: rerun using newer version of tr/test
ver 20170403.1 by jian: fork from "LUNA_train_unet.py", hypothesize we can keep training unet.hdf5, first find out where is the training loss, 0.1713 by the end of 25, 0.1722 here by +1 epoch
ver 20170403.2 by jian: hypothesize the training on v1 is converged
ver 20170404.1 by jian: split prediction to another script
ver 20170404.2 by jian: tune smooth par in dice coeff 1. => .1 starting from pre-trained h5, with hindsight, the tr loss of .17 is converged

review Dice coeff (f1-score)

first
>>> x = np.load ('../../../../../luna16/processed/trainMasks.npy')
>>> np.histogram(x)
(array([741765809,     56072,     49821,     46429,     43517,     42796,
           44646,     47742,     52937,   3125623]), array([ 0.        ,  0.00039216,  0.00078431,  0.00117647,  0.00156863,
        0.00196078,  0.00235294,  0.0027451 ,  0.00313726,  0.00352941,
        0.00392157]))
>>> y = np.load ('../../../../../luna16/processed/testMasks.npy')
>>> np.histogram(y)
(array([185272065,     13428,     12040,     11390,     10637,     10402,
           11067,     11624,     12653,    756934]), array([ 0.        ,  0.00039216,  0.00078431,  0.00117647,  0.00156863,
        0.00196078,  0.00235294,  0.0027451 ,  0.00313726,  0.00352941,
        0.00392157]))
re-train
>>> x = np.load ('trainMasks.npy')
>>> np.histogram(x)
(array([741852028,     55007,     49201,     46065,     42960,     42272,
           44396,     47248,     51988,   3044227]), array([ 0.        ,  0.00039216,  0.00078431,  0.00117647,  0.00156863,
        0.00196078,  0.00235294,  0.0027451 ,  0.00313726,  0.00352941,
        0.00392157]))
>>> y = np.load ('testMasks.npy')
>>> np.histogram(y)
(array([185185814,     14504,     12674,     11745,     11205,     10918,
           11301,     12124,     13616,    838339]), array([ 0.        ,  0.00039216,  0.00078431,  0.00117647,  0.00156863,
        0.00196078,  0.00235294,  0.0027451 ,  0.00313726,  0.00352941,
        0.00392157]))
Not a 0-1 mask

ver 20170405 by jian: fork from exp0, use tiny smooth 0.00001 (<0.00039216/30)
ver 20170406 by jian: use v3 and smooth=1, tune optimizer
ver 20170407 by jian: refine loss, remove smooth from numerator
to-do:

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

#N_EPOCH = #8 +6 +12 ~332 min
N_EPOCH = 20 #~9.1hr

working_path = "../../../../../luna16/processed/"


img_rows = 512
img_cols = 512

#smooth = 1.
#smooth = .1
#smooth = .00001
smooth = 1e-5
#http://tf-unet.readthedocs.io/en/latest/_modules/tf_unet/unet.html


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    #return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return (2. * intersection + 0.) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    #return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return (2. * intersection + 0.) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

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
        #optimizer=SGD(lr=5.0e-2), 
        #optimizer=SGD(lr=1.0e-1), 
        #optimizer=SGD(lr=1.0e-1,decay=.001), 
        optimizer=Adam(lr=5.0e-6,decay=.001), 
        loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    #imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
    #imgs_train = np.load(working_path+"trainImages-v2.npy").astype(np.float32)
    imgs_train = np.load(working_path+"trainImages-v3.npy").astype(np.float32)
    #imgs_train = imgs_train[:50]
    #print imgs_train.shape #(2843, 1, 512, 512)
    #imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)
    #imgs_mask_train = np.load(working_path+"trainMasks-v2.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path+"trainMasks-v3.npy").astype(np.float32)
    #imgs_mask_train = imgs_mask_train[:50]
    #print imgs_mask_train.shape #(2843, 1, 512, 512)
    #imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
    #imgs_test = np.load(working_path+"testImages-v2.npy").astype(np.float32)
    #imgs_test = imgs_test[:10]
    #print imgs_test.shape #(710, 1, 512, 512)
    #imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)
    #imgs_mask_test_true = np.load(working_path+"testMasks-v2.npy").astype(np.float32)
    #imgs_mask_test_true = imgs_mask_test_true[:10]
    #print imgs_mask_test_true.shape #(710, 1, 512, 512)
    #imgs_test = np.load(working_path+"pre-processed-images.npy").astype(np.float32)
    #imgs_test = np.load(working_path+"pre-processed-images-test-2.npy").astype(np.float32)
    
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std

    # Added normalization for test set as well
    #mean = np.mean(imgs_test)  # mean for data centering
    #std = np.std(imgs_test)  # std for data normalization
    #imgs_test -= mean  # images should already be standardized, but just in case
    #imgs_test /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint('unet-ss1.hdf5', monitor='loss', save_best_only=True)
    #model_checkpoint = ModelCheckpoint('unet-v2.hdf5', monitor='loss', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./unet-ss1.hdf5') # modify the path
        
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training. 
    #
    model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=N_EPOCH, verbose=1, shuffle=True, callbacks=[model_checkpoint])






    # loading best weights from training session
    #model.load_weights('./unet-ss1.hdf5')
    #model.load_weights('./unet-v2.hdf5')

    #num_test = len(imgs_test)
    #print ('# in test set',num_test)
    #imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    #for i in range(num_test):
    #    imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    #np.save('masksTestPredicted.npy', imgs_mask_test)
    #np.save('masksTestPredicted-v2.npy', imgs_mask_test)
    #np.save('masksTestPredicted-test-2.npy', imgs_mask_test)
    #mean = 0.0
    #for i in range(num_test):
    #    mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
    #mean/=num_test
    #print("Mean Dice Coeff : ",mean)

if __name__ == '__main__':
    train_and_predict(False)
    #train_and_predict(True)
