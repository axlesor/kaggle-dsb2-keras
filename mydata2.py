from __future__ import print_function

import os
import numpy as np
import dicom
from scipy.misc import imresize
import segment
import re

img_resize = True
img_shape = (64, 64)

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def natural_key3(x):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x[0])]

def crop_resize(img):
    """
    Crop center and resize.

    :param img: image to be cropped and resized.
    """
    if img.shape[0] < img.shape[1]:
        img = img.T
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    img = crop_img
    img = imresize(img, img_shape)
    return img

def mycrop_resize(img, ctr, x, y):
    # crop image from center: first find the average of center
    c_x = 0.0
    c_y = 0.0
    for slice in range(img.shape[0]):
        c_x += ctr[slice][0]
        c_y += ctr[slice][1]
    c_x = int(c_x/img.shape[0])
    c_y = int(c_y/img.shape[0])  
    crop_img = img[:,:,0:64,0:64] 
    for slice in range(img.shape[0]):
        for period in range(img.shape[1]):
            # Add 10 cm both side of the centers, x/y is in mm
            xwidth = int(100/x)
            x_start = c_x - xwidth
            x_end   = c_x + xwidth
            if ( x_start < 0 ): x_start = 0
            if ( x_end > img.shape[3] ): x_end = img.shape[3]

            ywidth = int(100/y)
            y_start = c_y - ywidth
            y_end   = c_y + ywidth
            if ( y_start < 0 ): y_start = 0
            if ( y_end > img.shape[2] ): y_end = img.shape[2]
            # store the new 
            crop_img[slice,period] = imresize(img[slice,period,y_start:y_end,x_start:x_end], img_shape)
    return crop_img            

def load_images(from_dir, verbose=True):
    """
    Load images in the form study x slices x width x height.
    Each image contains 30 time series frames so that it is ready for the convolutional network.

    :param from_dir: directory with images (train or validate)
    :param verbose: if true then print data
    """
    print('-'*50)
    print('Loading all DICOM images from {0}...'.format(from_dir))
    print('-'*50)

    current_study_sub = ''  # saves the current study sub_folder
    current_study = ''  # saves the current study folder
    current_study_images = []  # holds current study images
    ids = []  # keeps the ids of the studies
    study_to_images = dict()  # dictionary for studies to images
    total = 0
    images = []  # saves 30-frame-images
    from_dir = from_dir if from_dir.endswith('/') else from_dir + '/'
    for subdir, _, files in sorted(os.walk(from_dir), key=natural_key3):
        subdir = subdir.replace('\\', '/')  # windows path fix
        subdir_split = subdir.split('/')
        study_id = subdir_split[-3]
        # DEBUG
        #try:
        #   cnt = int(study_id)
        #except:
        #   cnt = 0
        #if( cnt < 698 ):
        #    print ('SKIP')
        #    continue

        print ('Subdir: ' + str(subdir))
        if "sax" in subdir:
            #print ('List Files: ' + str(files))
            for f in sorted(files, key=natural_key):
                image_path = os.path.join(subdir, f)
                if not image_path.endswith('.dcm'):
                    continue

                image = dicom.read_file(image_path)
                (xpix, ypix) = image.PixelSpacing
                image = image.pixel_array.astype(float)
                image /= np.max(image)  # scale to [0,1]
                # Next two lines removes memory LEAK
                #if img_resize:
                #    image = crop_resize(image)

                if current_study_sub != subdir:
                    x = 0
                    try:
                        while len(images) < 30:
                            images.append(images[x])
                            x += 1
                        if len(images) > 30:
                            images = images[0:30]

                    except IndexError:
                        pass
                    current_study_sub = subdir
                    current_study_images.append(images)
                    #print ("Here: " + str(len(images)))
                    images = []

                if current_study != study_id:
                    # Now we have slice/time/x/y image, time to apply image resize
                    #kel = np.array(current_study_images)
                    #print (kel.shape, current_study, study_id)

                    #study_to_images[current_study] = np.array(current_study_images)
                    if current_study != "":
                        center = segment.calc_centers(np.array(current_study_images))
                        if img_resize:
                            current_study_images = mycrop_resize(np.array(current_study_images), center, oxpix, oypix)
                        study_to_images[current_study] = np.array(current_study_images)
                        ids.append(current_study)
                    current_study = study_id
                    current_study_images = []
                # Make sure all Images aligned with same respect ratio
                if image.shape[0] < image.shape[1]:
                    image = image.T
                images.append(image)
                # Save original pixel sizes for calculation
                (oxpix, oypix) = (xpix, ypix)
                if verbose:
                    if total % 1000 == 0:
                        print('Images processed {0}'.format(total))
                total += 1
    x = 0
    try:
        while len(images) < 30:
            images.append(images[x])
            x += 1
        if len(images) > 30:
            images = images[0:30]
    except IndexError:
        pass

    print('-'*50)
    print('All DICOM in {0} images loaded.'.format(from_dir))
    print('-'*50)

    
    current_study_images.append(images)
    #kel = np.array(current_study_images)
    #print (kel.shape)
    
    center = segment.calc_centers(np.array(current_study_images))
    
    if img_resize:
       fimages = mycrop_resize(np.array(current_study_images), center, oxpix, oypix)
    study_to_images[current_study] = fimages
    #study_to_images[current_study] = np.array(current_study_images)
    if current_study != "":
        ids.append(current_study)

    return ids, study_to_images


def map_studies_results():
    """
    Maps studies to their respective targets.
    """
    id_to_results = dict()
    train_csv = open('data/train.csv')
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, diastole, systole = item.replace('\n', '').split(',')
        id_to_results[id] = [float(diastole), float(systole)]

    return id_to_results


def write_train_npy():
    """
    Loads the training data set including X and y and saves it to .npy file.
    """
    print('-'*50)
    print('Writing training data to .npy file...')
    print('-'*50)

    study_ids, images = load_images('data/train')  # load images and their ids
    studies_to_results = map_studies_results()  # load the dictionary of studies to targets
    X = []
    y = []

    for study_id in study_ids:
        # Here study is in Slice/Time/x/y format in np.array
        study = images[study_id]
        outputs = studies_to_results[study_id]
        for i in range(study.shape[0]):
            X.append(study[i, :, :, :])
            y.append(outputs)

    X = np.array(X, dtype=np.uint8)
    y = np.array(y)
    np.save('data/X_train.npy', X)
    np.save('data/y_train.npy', y)
    print('Done.')


def write_validation_npy():
    """
    Loads the validation data set including X and study ids and saves it to .npy file.
    """
    print('-'*50)
    print('Writing validation data to .npy file...')
    print('-'*50)

    ids, images = load_images('data/validate')
    study_ids = []
    X = []

    for study_id in ids:
        study = images[study_id]
        for i in range(study.shape[0]):
            study_ids.append(study_id)
            X.append(study[i, :, :, :])

    X = np.array(X, dtype=np.uint8)
    np.save('data/X_validate.npy', X)
    np.save('data/ids_validate.npy', study_ids)
    print('Done.')

if __name__ == "__main__":
    write_train_npy()
    write_validation_npy()
