### Include all libraries
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# import cv2
from sklearn.model_selection import train_test_split

#from itertools import chain
#from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
#from skimage.morphology import label

import keras
#from keras import losses
from keras.models import Model, load_model#, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
#from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers, callbacks
from keras.callbacks import Callback

import tensorflow as tf

from keras.preprocessing.image import load_img#array_to_img, img_to_array, load_img#,save_img

import time
    
# Set the size of image and mask
img_size_ori = 101
img_size_target = 101

def upsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i

######
#### DEFINE LOSS FUNCTION
######
# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def bce_dice_mse_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred) + 10.0*keras.losses.mean_squared_error(y_true, y_pred)

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label1, pred):
    return tf.py_func(get_iou_vector, [label1, pred>0.5], tf.float64)

def my_iou_metric_2(label1, pred):
    return tf.py_func(get_iou_vector, [label1, pred >0], tf.float64)

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
    intersection = temp1[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

###### 
### DEFINE MODEL
######
def BatchActivate(x, actfn='relu'):
    x = BatchNormalization()(x)
    x = Activation(actfn)(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x, 'relu')
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput, 'relu')
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x, 'relu')
    return x

# Build model
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
    return output_layer

def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def display_training_history(history, outfile):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history["my_iou_metric"], label="Train score")
    ax_score.plot(history.epoch, history.history["val_my_iou_metric"], label="Validation score")
    ax_score.legend()
    ax_loss.set_xlabel("Epoch")
    ax_score.set_xlabel("Epoch")
    fig.savefig(outfile)
    
def get_best_threshold(y_valid, preds_valid):
    #Score the model and do a threshold optimization by the best IoU.
    ## Scoring for last model, choose threshold by validation data 
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    
    # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
    thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

    ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in thresholds])
    print('Intersection-over-Union values for different thresholds are listed below')
    print(ious)

    # instead of using default 0 as threshold, use validation data to find the best threshold.
    threshold_best_index = np.argmax(ious) 
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    # Display the ious as a function of threshold
    fig, ax = plt.subplots()
    ax.plot(thresholds, ious)
    ax.plot(threshold_best, iou_best, "xr", label="Best threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("IoU")
    plt.suptitle("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    ax.legend()
    fig.savefig('iou_threshold.png')
    
    return threshold_best

def main():
    # Get the start time
    t_start = time.time()

    # Define the name of files for saving intermediate models
    version = 1
    basic_name = f'Unet_resnet_v{version}'
    save_model_name1 = basic_name + '_1' + '.model'
    save_model_name2 = basic_name + '_2' + '.model'
    save_model_name3 = basic_name + '_3' + '.model'
    submission_file = basic_name + '.csv'

    # Loading of training/testing ids and depths
    train_df = pd.read_csv("train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    len(train_df)

    # Load images and masks
    train_df["images"] = [np.array(load_img("./train/images/{}.png".format(idx), color_mode="grayscale")) / 255 for idx in train_df.index]
    train_df["masks"] = [np.array(load_img("./train/masks/{}.png".format(idx), color_mode="grayscale")) / 255 for idx in train_df.index]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    
    # Make a plot for the coverage and coverage class
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    sns.distplot(train_df.coverage, kde=False, ax=axs[0])
    sns.distplot(train_df.coverage_class, bins=10, kde=False, ax=axs[1])
    plt.suptitle("Salt coverage")
    axs[0].set_xlabel("Coverage")
    axs[1].set_xlabel("Coverage class")
    fig.savefig('coverage_data.png')

    #Plotting the depth distributions
    fig, axs = plt.subplots()
    sns.distplot(train_df.z, label="Train")
    sns.distplot(test_df.z, label="Test")
    plt.legend()
    plt.suptitle("Depth distribution")
    fig.savefig('dept_distr.png')

    # Create train/validation split stratified by salt coverage (validation data size=0.2)
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)

    # Training Data augmentation: Flip image about the center in x-direction
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    print(x_train.shape)
    print(y_valid.shape)

    ### STAGE1
    print('Stage 1')
    # model1 - binary classification for each pixel - so use binary cross entropy as loss function
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model(input_layer, 16, 0.5)
    model1 = Model(input_layer, output_layer)
    c = optimizers.adam()
    model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

    # Display summary of model
    model1.summary()

    # Now set up the callbacks for handling stuff during model training
    early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name1,monitor='val_my_iou_metric', mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

    # Train the model using binary cross entropy(bce)
    epochs = 10
    my_batch_size = 32
    history = model1.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=epochs, batch_size=my_batch_size,callbacks=[ model_checkpoint,early_stopping, reduce_lr], verbose=2)

    # Display loss and my_iou_metric during model fitting
    display_training_history(history, 'train_history1.png')

    # Delete model1
    del model1
    gc.collect()
    
    ### STAGE 2
    print('Stage 2')

    # Load presaved model
    model2 = load_model(save_model_name1,custom_objects={'my_iou_metric': my_iou_metric, 
                                                    'dice_coef':dice_coef, 'bce_dice_loss':bce_dice_loss,
                                                   'bce_dice_mse_loss':bce_dice_mse_loss})


    # Compile model with bce_dice_loss as the loss function
    model2.compile(loss=bce_dice_loss, optimizer=c, metrics=[my_iou_metric])

    # Train model with bce_dice_loss
    epochs = 10
    my_batch_size = 32
    history = model2.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=my_batch_size,
                    callbacks=[ model_checkpoint,early_stopping, reduce_lr], 
                    verbose=2)

    # Display loss and my_iou_metric during model fitting
    display_training_history(history, 'train_history2.png')

    # Delete model2
    del model2
    gc.collect()
    
    ### STAGE 3
    print('Stage 3')

    # Load presaved model
    model3 = load_model(save_model_name2,custom_objects={'my_iou_metric': my_iou_metric, 
                                                    'dice_coef':dice_coef, 'bce_dice_loss':bce_dice_loss,
                                                   'bce_dice_mse_loss':bce_dice_mse_loss})

    # Compile model with bce_dice_mse_loss
    model3.compile(loss=bce_dice_mse_loss, optimizer=c, metrics=[my_iou_metric])

    epochs = 10
    my_batch_size = 32
    history = model3.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=my_batch_size,
                    callbacks=[ model_checkpoint,early_stopping, reduce_lr], 
                    verbose=2)

    # Display loss and my_iou_metric during model fitting
    display_training_history(history, 'train_history3.png')

    # Delete model3
    del model3
    gc.collect()

    # Load presaved model
    model4 = load_model(save_model_name3,custom_objects={'my_iou_metric': my_iou_metric, 'my_iou_metric': my_iou_metric, 
                                                    'dice_coef':dice_coef, 'bce_dice_loss':bce_dice_loss,
                                                   'bce_dice_mse_loss':bce_dice_mse_loss})
    # Make predictions on the validation data
    preds_valid = predict_result(model4,x_valid,img_size_target)
    
    # Get the threshold that maximizes the IOU
    threshold_best = get_best_threshold(y_valid, preds_valid)
    
    # Read in the test data and make predictions on it
    x_test = np.array([(np.array(load_img("./test/images/{}.png".format(idx), color_mode="grayscale"))) / 255 for idx in test_df.index]).reshape(-1, img_size_target, img_size_target, 1)
    preds_test = predict_result(model4,x_test,img_size_target)

    # Make predictions on the test data
    t1 = time.time()
    pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(test_df.index.values)}
    t2 = time.time()

    print(f"Usedtime = {t2-t1} s")

    # Save the predictions in a dataframe
    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(submission_file)

    # Print out the total execution time
    t_finish = time.time()
    print(f"Total run time = {(t_finish-t_start)/3600} hours")

# Execute main function
main()