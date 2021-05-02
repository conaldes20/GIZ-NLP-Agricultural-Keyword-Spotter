import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import math
from datetime import datetime
import pandas as pd
from dateutil.parser import parse
import os.path
from pathlib import Path
from math import floor
from random import randint
from decimal import Decimal
#from random import seed
import random
from numpy.linalg import pinv
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
import os
from PIL import Image
from scipy.fftpack import fft
from scipy import signal

random.seed(0)

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate/1e3))
    noverlap = int(round(step_size * sample_rate/1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)

def wav2img(specpath, audiopath, file_name, figsize=(4,4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """
    fig = plt.figure(figsize=figsize)
    
    # use soundfile library to read in the wave files
    samplerate, sample_sound  = wavfile.read(audiopath + file_name)
    _, spectrogram = log_specgram(sample_sound, samplerate)
        
    #string = "freeCodeCamp"
    #print(string[2:6])      #  eeCo

    if (file_name.find('.') != -1):            
        ind = file_name.find('.')
        image_name = file_name[0:ind]
        #print ('image_name', image_name) 

        image_name = file_name[-11:-4]
        ## create output path
        specpath += image_name + '.png'
        plt.imsave(specpath, spectrogram, format='png')
        plt.cla()
        plt.clf()    
        plt.close(fig)
        #del plt    
        del fig
        #plt.close()
    else: 
        print ("Invalid file name: " + file_name)
    
def wav2img_waveform(wavepath, audiopath, file_name, figsize=(4,4)):
    samplerate, sample_sound  = wavfile.read(audiopath + file_name)
    
    if (file_name.find('.') != -1):
        ind = file_name.find('.')
        file_name = file_name[0:ind]
        #print ('file_name', file_name)
        
        fig = plt.figure(figsize=figsize)
        plt.plot(sample_sound)
        plt.axis('off')
        #file_name = file_name[-11:-4]
        ## create output path
        output_file = wavepath + file_name
        plt.savefig('%s.png' % output_file)
        plt.cla()
        plt.clf()    
        plt.close(fig)
        #del plt    
        del fig
        #plt.close()
    else: 
        print ("Invalid file name: " + file_name)

def generateWAVEIMG(audio_path, wave_path, spec_path):        
    filesList = []

    for x in os.listdir(audio_path):
        filesList.append(x)
        
    # get all the wave files
    #filesList = filesList[:11]
    
    spec_images = os.listdir(spec_path)
    if len(spec_images) == 0:
        print('spectogram_images not created yet! wait while spectogram_images are being created ....')
        for file in filesList:
            wav2img(spec_path, audio_path, file)
            
    print('wav2img: all images created')
    print("                            ")
    wave_images = os.listdir(wave_path)
    if len(wave_images) == 0:
        print('waveform_images not created yet! wait while waveform_images are being created ....')
        for file in filesList:
            wav2img_waveform(wave_path, audio_path, file)
    
    print('wav2img_waveform: all images created')    
    print("                            ")
    
def xyMatadata(spec_dir, audio_dir):
    now1 = datetime.now()
    starting_time = now1.strftime("%H:%M:%S")
    timestamp1 = datetime.timestamp(now1)
    
    # load the dataset from the CSV file       

    print("### Input Data ###")
    print("==================")
        
    img_uids_list = []     
    files = os.listdir(spec_dir)
    for f in files:
        file_name = spec_dir + f        
        if os.path.isfile(file_name):
            if file_name[-3:] == "bmp":
                img_uids_list.append([file_name[-11:-4], f])
            elif file_name[-3:] == "png":
                img_uids_list.append([file_name[-11:-4], f])
            elif file_name[-3:] == "jpg":
                img_uids_list.append([file_name[-11:-4], f])
            elif file_name[-4:] == "jpeg":
                img_uids_list.append([file_name[-13:-5], f])

    print('img_uids_list', img_uids_list)
    img_uidsln = len(img_uids_list)
    print('len(img_uids_list)', img_uidsln)
    print("                            ")
    
    wav_uids_list = []     
    files = os.listdir(audio_dir)
    # 'C:/Users/CONALDES/Documents/LanguageClassifier/audio_files/'
    audrln = len(audio_dir)
    audio_dir = audio_dir[-audrln:-1]
    subdirs = audio_dir.split('/')
    #print('subdirs', subdirs)
    #print('subdirs[5]', subdirs[5])
    audio_dir = audio_dir + '/'
    for f in files:
        file_name = audio_dir + f
        fdir_name = subdirs[5] + '/' + f
        #print('fdir_name', fdir_name)
        if os.path.isfile(file_name):
            if file_name[-3:] == "wav":
                #wav_uids_list.append([file_name[-11:-4], fdir_name])
                wav_uids_list.append(fdir_name)
                
    print('wav_uids_list', wav_uids_list)
    print('len(wav_uids_list)', len(wav_uids_list))
    print("                            ")
    
    # load the dataset from the CSV file 
    reader = csv.reader(open("C:/Users/CONALDES/Documents/LanguageClassifier/SampleSubmission.csv", "r"), delimiter=",")
    xx = list(reader)    
    columns_headers = []
    for row in range(0, 1):
        recln = len(xx[row])
        for i in range(1, recln):
            columns_headers.append(xx[row][i]) 

    #columns_headers = np.array(allrecs)
    print("columns_headers: " + str(columns_headers))
    print("len(columns_headers): " + str(len(columns_headers)))
    print("                            ")
    
    # load the dataset from the CSV file 
    reader = csv.reader(open("C:/Users/CONALDES/Documents/LanguageClassifier/Train.csv", "r"), delimiter=",")
    xx = list(reader)
    xxln = len(xx)
    allrecs = []
    for row in range(1, xxln):
        fields = []
        recln = len(xx[row])
        for i in range(0, recln):
            fields.append(xx[row][i])    
        allrecs.append(fields)

    features = np.array(allrecs)

    filenames = features[:,0]
    print("filenames: " + str(filenames))
    print("                            ")
    
    labels = features[:,1]
    labelsln = len(labels)
    #labels_set = list(set(labels))
    #labelsetln = len(labels_set)
    colheaderln = len(columns_headers)
    
    x = []
    y = []    
    test_x = []
    
    #random.shuffle(img_uids_list)
    #img_uids_list = img_uids_list[:2193]
    y = np.zeros((labelsln, colheaderln))
    fnamesln = len(filenames)
    imguidslstln = len(img_uids_list)
    for i in range(0, imguidslstln):
        seen = False
        for j in range(0, fnamesln):            
            #if img_uids_list[i][0] == filenames[j]:
            if (filenames[j].find(img_uids_list[i][0]) != -1):
                #print('i, j, img_uids_list: ' +  str(i) + ', ' + str(j) + ', ' + str(img_uids_list[i][1]))
                x.append(img_uids_list[i][1])
                for k in range(0, colheaderln):
                    #if labels[j] == labels_set[k]:
                    if labels[j] == columns_headers[k]:
                        y[j][k] = 1
                        
                seen = True
                break

        if seen == False:
            test_x.append(img_uids_list[i][1])

    print('img_uids_train_list', x)
    print('len(x)', len(x))
    print("                            ")
    print('img_uids_test_list', test_x)
    print('len(test_x)', len(test_x))
            
    y = np.vstack(y)
    
    test_x_fn = []    
    
    wavuidslstln = len(wav_uids_list)
    for i in range(0, wavuidslstln):
        seen = False
        for j in range(0, fnamesln):
            #print('i, j, wav_uids_list, filenames: ' +  str(i) + ', ' + str(j) + ', ' + str(wav_uids_list[i]) + ', ' + str(filenames[j]))
            if wav_uids_list[i] == filenames[j]:                                        
                seen = True
                break

        if seen == False:
            test_x_fn.append(wav_uids_list[i])

    #print("img_uids_list: " + str(img_uids_list))
    print("                            ")
    print("x: " + str(x))
    print("                            ")
    print("test_x: " + str(test_x))
    print("                            ")
    print("test_x_fn: " + str(test_x_fn))
    print("                            ")
    print("Image conversion to grayscale going on ......")
    print("                            ") 
       
    imgLstLen = len(x)
    x = getAgeRGBArrays(spec_dir, x)
    test_x = getAgeRGBArrays(spec_dir, test_x)
    xln = len(x)
    test_xln = len(test_x)
    print('len(x): ' + str(len(x)))
    print('len(test_x): ' + str(len(test_x)))
    print("                            ")
    
    x = np.vstack(x).astype("float")
    test_x = np.vstack(test_x).astype("float")
    xrows, xcols = x.shape
    testxrows, testxcols = test_x.shape
    print('raw -> train: ' + str(x))
    print("                            ")
    print('raw -> rows and cols in train: ' + str(x.shape))
    print("                            ")
    print('raw -> test: ' + str(test_x))
    print("                            ")
    print('raw -> rows and cols in test: ' + str(test_x.shape))

    minCol = [x[:,c].min() for c in range(xcols)] 
    maxCol = [x[:,c].max() for c in range(xcols)]
    print("minCol: " + str(minCol))
    print("maxCol: " + str(maxCol))    
        
    yrows, ycols = y.shape
    miny = [y[:,c].min() for c in range(ycols)] 
    maxy = [y[:,c].max() for c in range(ycols)] 
    print("miny: " + str(miny))
    print("maxy: " + str(maxy))    

    for j in range(0, ycols):
        for row in range(0, yrows):
            temp = (y[row][j] - miny[j])/(maxy[j] - miny[j])
            y[row][j] = temp
    
    for j in range(0, xcols):      
        for row in range(xrows):            
            x[row][j] = (x[row][j] - minCol[j])/(maxCol[j]- minCol[j])

    for j in range(0, testxcols):    
        #tempval = x[:,j]            
        #meanCol[j] = np.mean(tempval)
        #stdCol[j] = np.std(tempval)        
        for row in range(testxrows):            
            test_x[row][j] = (test_x[row][j] - minCol[j])/(maxCol[j]- minCol[j])
           
    print("                            ")
    print('x.shape: ' + str(x.shape))
    print("                            ")
    print('y.shape: ' + str(y.shape))    
    print('                           ')
    print('test_x.shape: ' + str(test_x.shape))    
    print('                           ')
        
    print('@@@@@@@@@@@ Normalised data @@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('x: ' + str(x))
    print('                           ')
    print('test_x: ' + str(test_x))
    print('                           ')
    print('y: ' + str(y))
    print('                           ')
        
    return x, y, test_x, test_x_fn, columns_headers, xrows, testxrows, colheaderln

def predict_targets(xtest, coefs):                   
    predy = 0.00
    cfln = len(coefs)
    predy = predy + coefs[0]
    for i in range(0, (cfln - 1)):
        predy = predy + xtest[i]*coefs[i + 1]    # xtest[i] i = 0 to 46 and coefs[i] i = 1 to 47
    
    return predy

def simulateModelA(x, y, x_data):
    # current date and time
    print('                           ')    
    print('@@@@@@@@@@@ Model Simulation @@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    now0 = datetime.now()
    timestamp0 = datetime.timestamp(now0)
    brow, bcol = x.shape
    m = np.ones((brow,1))
    x = np.concatenate((m,x),axis=1)

    print('x (1 column added): ' + str(x))

    #c, residuals, rank, s = lstsq(x, y)
    yrows, ycols = y.shape 

    c = {}    
    for l in range(0, ycols):
        c[l] = (None,)
    
    for k in range(0, ycols):
        c[k] = pinv(x).dot(y[:,k])
        print('Column ' + str(k) + ' Coefficients: ' + str(c[k]))
        
    xdtln = len(x_data)
    #print('len(x_data): ' + str(len(x_data)))
    #print('x_data: ' + str(x_data))
    
    pred_probs = {}    
    for l in range(0, ycols):
        pred_probs[l] = []
    
    for l in range(0, ycols):
        for i in range(0, xdtln):
            pred_prob = predict_targets(x_data[i], c[l])
            if pred_prob < 0:
                pred_prob = 0.00
            pred_probs[l].append(pred_prob)

    output_cols = []
    for i in range(0, xdtln):
        fields = []
        for l in range(0, ycols):
            fields.append(round(pred_probs[l][i], 8))
        output_cols.append(fields)
            
    now1 = datetime.now()
    timestamp1 = datetime.timestamp(now1)
    time_elapsed = timestamp1 - timestamp0
    print('Time elapsed for computations: ' + str(time_elapsed) + 'seconds')
    return output_cols

def simulateModelB(xrows, x_train, y, x_data):
    m = np.ones((xrows, 1))
    b = np.matrix(x_train)            # x
    print('b: ' + str(b))
    print('b.shape: ' + str(b.shape))
    b = np.concatenate((m, b),axis=1)
    print('b.shape (i column added): ' + str(b.shape))
    d = b.T                           # x.T
    print('b.T: ' + str(d))
    print('b.T.shape: ' + str(d.shape))
    e = np.linalg.inv(np.matmul(d, b)) # inverse(x.Tx)
    print('e: ' + str(e))
    print('e.shape: ' + str(e.shape))

    for k in range(0, colheaderln):
        y[:,k] = np.matrix(y[:,k])
        print('y[:,k].shape: ' + str(y[:,k].shape))
        y[:,k] = y[:,k].T
        print('y[:,k].T.shape: ' + str(y[:,k].shape))
        
    '''    
    y2 = np.matrix(y2)
    print('y2.shape: ' + str(y2.shape))
    y2 = y2.T
    print('y2.T.shape: ' + str(y2.shape))
    '''
    
    c = {}    
    for l in range(0, colheaderln):
        c[l] = (None,)
        
    f = np.matmul(e, d)                # inverse(x.Tx)x
    print('f: ' + str(f))
    print('f.shape: ' + str(f.shape))
    for k in range(0, colheaderln):
        c[k] = np.matmul(f, y[:,k])              # (inverse(x.Tx)x)y
        print('c[k].shape: ' + str(c[k].shape))
        print('c[k]: ' + str(c[k]))

    '''    
    c2 = np.matmul(f,y2)              # (inverse(x.Tx)x)y    
    print('c1.shape: ' + str(c1.shape))
    print('c2.shape: ' + str(c2.shape))    
    print('c1: ' + str(c1))
    print('c2: ' + str(c2))
    '''
    
    #Input the test data and thereby store it in a list, x_test. Predict the target variable using the test data
    #and the coefficient matrix and thereby stored the result in Y1, Y2 .

    #print(y1*(max[-2]-min[-2])+min[-2])
    #print((y2*(max[-1]-min[-1]))+min[-1])

    xdtln = len(x_data)
    print('len(x_data): ' + str(len(x_data)))
    print('x_data: ' + str(x_data))

    pred_probs = {}    
    for l in range(0, colheaderln):
        pred_probs[l] = []
    
    for l in range(0, colheaderln):
        for i in range(0, xdtln):
            pred_prob = predict_targets(x_data[i], c[l])
            if pred_prob < 0:
                pred_prob = 0.00
            pred_probs[l].append(round(pred_prob, 8))

    output_cols = []
    for i in range(0, xdtln):
        fields = []
        for l in range(0, ycols):
            fields.append(pred_probs[l][i])
        output_cols.append(fields)

    return output_cols

def getAgeRGBArrays(file_dir, img_list):
    pxel_array = []
    img_list_len = len(img_list)
    
    for i in range(0, img_list_len):
        try:
            file_name = file_dir + img_list[i]
            img = Image.open(file_name, "r")
            pix_val = list(img.getdata())
            pix_val_flat = [x for sets in pix_val for x in sets]
            sum_elem0 = 0
            sum_elem1 = 0
            sum_elem2 = 0
        
            listlen = len(pix_val)
            for l in range(0, listlen):
                sum_elem0 += pix_val[l][0]
                sum_elem1 += pix_val[l][1]
                sum_elem2 += pix_val[l][2]
            pxel_array.append([sum_elem0,sum_elem1,sum_elem2])            
        except TypeError as err:
                print('Handling run-time error:', err)
                
    #pxelArray = np.matrix(pxel_array)
        
    return pxel_array

def numOfRecords(inputX, inputY):
    Xrow, Xcol = inputX.shape
    Yrow, Ycol = inputY.shape
    return Xrow, Xcol, Ycol

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

audio_path = 'C:/Users/CONALDES/Documents/LanguageClassifier/audio_files/'
wave_path = 'C:/Users/CONALDES/Documents/LanguageClassifier/waveform_images/'
spec_path = 'C:/Users/CONALDES/Documents/LanguageClassifier/spectogram_images/'

generateWAVEIMG(audio_path, wave_path, spec_path)

x_train, Y, x_test, test_x_fn, columns_headers, xrows, testxrows, colheaderln = xyMatadata(spec_path, audio_path)

predicted_probs = simulateModelA(x_train, Y, x_test)
#predicted_probs = simulateModelB(xrows, x_train, Y, x_test)
    
headers = []
headers.append('fn')
for k in range(colheaderln):
    headers.append(columns_headers[k])
    
predicted_probs = np.vstack(predicted_probs)
print('predicted_probs', predicted_probs)
predprb_rows, predprb_cols = predicted_probs.shape
print('predicted_probs.shape', predicted_probs.shape)

test_x_fn = np.vstack(test_x_fn)
fn_probabilities = np.concatenate((test_x_fn, predicted_probs), axis=1)

with open("C:/Users/CONALDES/Documents/LanguageClassifier/ConaldesSubmission.csv", "w", newline='') as file:
    writer = csv.writer(file)
    #writer.writerow(['fn', 'Label'])
    writer.writerow(headers)
    for row in fn_probabilities:    
        l = list(row)    
        writer.writerow(l)
print("                                          ")
print("### C:/Users/CONALDES/Documents/LanguageClassifier/ConaldesSubmission.csv contains results ###")  
    
