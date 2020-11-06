#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import os
from scipy.io import wavfile
from tensorflow.python.keras.models import model_from_json
import pyaudio as pa
import cv2
import matplotlib.pyplot as plt
#import time

def normalize_input(clip):
    clip=clip/(clip.max()-clip.min())
    return clip

def pre_emphasis2(X,alpha=0.95,mode=0):
    """Input
    X = N x M array where N = number of records
    mode if mode =one N=0
    M = length of clip
    Output Y = N x M array: X after filtering
    """
    if mode==0:
        return np.append(X[0],X[1:]-X[:-1]*alpha)
    elif mode=="Batch":
        shape=X.shape
        y=np.concatenate((X[:,0:1],X[:,1:]-X[:,:-1]*alpha),axis=1)
        return y
def frame_cutting2(batch,rate,options={'size':40,'stride':20}):
    """devide clip into frames"""
    size=options['size']
    stride=options['stride']
    frame_length,frame_step=int(round(0.001*size*rate)),int(round(stride*0.001*rate))
    signal_length=batch.shape[1]
    num_frames=int((signal_length-frame_length)/frame_step+1)
    #print("num of frames:{0} frame length: {1}".format(num_frames,frame_length))
    pad_size=num_frames*frame_step+frame_length
    #pad_signal=np.concatenate(clip,np.zeros(pad_size-signal_length))
    pad_signal=np.concatenate((batch,np.zeros((batch.shape[0],pad_size-signal_length))),axis=1)
    frames=np.zeros((num_frames,frame_length,batch.shape[0]))
    pad_signal=pad_signal.T
    for i in range(num_frames):
        frames[i,:,:]=pad_signal[i*frame_step:i*frame_step+frame_length]
    return frames
def furTransform(frames,NFFT=1024):
    NFFT=1024
    #print(frames.shape[2])
    hamming=np.hamming(frames.shape[1])
    #переделать
    for i in range(frames.shape[2]):
        frames[:,:,i]=frames[:,:,i]*hamming
    fur_frames=np.absolute(np.fft.rfft(frames,NFFT,axis=1))
    #print(fur_frames.shape)
    pow_frames=((1.0/NFFT)*(fur_frames**2))
    return pow_frames

def log_mel_filters2(pow_frames,rate=16000,F_min=300,F_max=8000,Num_filt=100):
    
    
    """Input pow_frames for stack of clips shape: num_frames,frame_length,num_clips"""
    f_to_mel=lambda f: 2595 * np.log10(1 + f/700)
    mel_to_f=lambda m:700*(10**(m/2595)-1)
    NFFT=1024
    M_min,M_max=f_to_mel(F_min), f_to_mel(F_max)
    hz_scale=mel_to_f(np.linspace(M_min,M_max,Num_filt+2))
    F_scale= np.floor((NFFT+1)*hz_scale /rate)
    #print("HZ_scale with len={0}:\n".format(len(hz_scale)),hz_scale)
    #print("F_scale:\n",F_scale)
    filt_bank=np.zeros((Num_filt,int(np.floor(NFFT / 2 + 1))))
    filter_banks=np.zeros((pow_frames.shape[0],filt_bank.shape[0],pow_frames.shape[2]))

    for m in range(1, Num_filt + 1):
        f_m_minus = int(F_scale[m - 1]) 
        f_m = int(F_scale[m])
        f_m_plus = int(F_scale[m + 1])
        for k in range(f_m_minus, f_m):
            filt_bank[m - 1, k] = (k - F_scale[m - 1]) / (F_scale[m] - F_scale[m - 1])
        for k in range(f_m, f_m_plus):
            filt_bank[m - 1, k] = (F_scale[m + 1] - k) / (F_scale[m + 1] - F_scale[m])
    for i in range(pow_frames.shape[2]):
        filter_banks[:,:,i] = np.dot(pow_frames[:,:,i], filt_bank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20*np.log10(filter_banks)
    #num=0
    #print("filter {0}:\n".format(num),filt_bank[num,:])
    return filter_banks
def wav2img_tensor(batch,rate=16000,size=30,stride=15,NFFT=1024,F_min=300,F_max=8000,Num_filt=60):
    y=pre_emphasis2(batch,mode='Batch')
    frames=frame_cutting2(y,rate,{'size':size,'stride':stride})
    NFFT=1024
    pow_frames=furTransform(frames,NFFT=NFFT)
    log_mel_batch=log_mel_filters2(pow_frames,rate=rate)
    return log_mel_batch

class Cash:
    def __init__(self,RATE=16000,CHUNK=8000):
        self.CHUNK=int(CHUNK)
        self.RATE=int(RATE)
        self.__pool=np.zeros(RATE)
        self.index=0
        self.free=True
        self.loaded=False
        self.num=int(RATE//CHUNK)
    def load(self,clip):
        assert not self.loaded, 'already loaded'
        self.__pool[self.index:self.index+self.CHUNK]=clip
        self.index+=self.CHUNK
        if self.index==self.RATE:
            self.loaded =True
            self.free = False
    def append(self,clip):
        try:
            assert clip.shape[0]==self.CHUNK or len(clip.shape)>1
            if self.free:
                self.__pool[self.index:self.index+self.CHUNK]=clip
                self.index+=self.CHUNK
            else:
                self.pop()
                self.__pool[self.index:self.index+self.CHUNK]=clip
                self.index+=self.CHUNK
            if self.index==self.RATE:
                self.free=False
        except AssertionError:
            print("Uncorrect size of input, input with shape {0}, except {1}".format(clip.shape[0],self.CHUNK))
    def pop(self):
        try:
            assert not self.free
            self.index=(self.num-1)*self.CHUNK
            self.__pool[:self.CHUNK*(self.num-1)]=self.__pool[self.CHUNK:]
            self.free=True
        except AssertionError:
            print("Pool empty")
    def get(self):
        return self.__pool
    def __str__(self):
        return str(self.__pool)
    
class Recognizer():
    path_model='./my_modelVD2.json'
    weights_path='./my_model_weightsVD2.h5'
    path_model2= './my_model1RUS_97.json' #'./my_model111.json' #
    weights_path2= './my_model_weights1RUS_97.h5' # "./my_model_weights111.h5"
    #weight
    def __init__(self,model=None):
        if model:
            self.model=model
        else:
            #from tensorflow.python.keras.models import model_from_json
            jfile = open(Recognizer.path_model, "r")
            loaded_model = jfile.read()
            jfile.close()
            self.model = model_from_json(loaded_model)
            self.model.load_weights(Recognizer.weights_path)
        self.threshold=0.93
        jfile = open(Recognizer.path_model2, "r")
        loaded_model = jfile.read()
        jfile.close()
        self.model2 = model_from_json(loaded_model)
        self.model2.load_weights(Recognizer.weights_path2)
        self.__labels={0:'speech', 1:'silence'}
        self.__labels2={0: '0', 1: '1', 2: '10', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: 'no', 12: 'other', 13: 'yes'}
        #print("inicialization complite, models loaded")
    
    def get_labels(self):
        return self.__labels
  
    #return self.__images
    def __getRecord(self,recorder):
        #print(recorder.__class__.__name__)
        assert (recorder.__class__.__name__=='Recorder'), 'Invalid type of object in input'
        #print(recorder.stream_active())
        if recorder.stream_active():
            clip,sample_rate=recorder.get_from_micro()
        else:
            sample_rate=recorder.push_sample_rate()
            clip=recorder.push_record()
        return sample_rate,clip
  
    def prepare_clip(self,clip,rate,batch_size=1,batch_stride=0.5):
        """сut clip on smaller parts if it need"""
        batch_size=int(rate*batch_size)
        batch_stride=int(round(rate*batch_stride))
        clip_size=clip.shape[0]
        num_of_records=int((clip_size-batch_size)/(batch_stride)+1)
        pad_size=num_of_records*batch_stride+batch_size
        print(pad_size,'  ',clip_size)
        new_record=np.r_[clip,np.zeros(pad_size-clip_size)]
        records=np.zeros((num_of_records,batch_size))
        for i in range(num_of_records):
            records[i,:]=new_record[i*batch_stride:i*batch_stride+batch_size]
        print(records.shape)
        return rate,records
  
    def __make_images(self,records,rate=None):
        images=wav2img_tensor(records,rate=rate)
        images = images.astype('float32')
        return images
  
    def get_images(self,recorder,batch_size=1,batch_stride=0.5):
        """get image from record in recorder"""
        rate,clip=self.__getRecord(recorder)
        clip=normalize_input(clip)
        #rate,records=self.prepare_clip(clip,rate,batch_size=batch_size,batch_stride=batch_stride)
        records=np.expand_dims(clip,axis=0)
        images=self.__make_images(records,rate=rate)
        images=np.transpose(images,axes=(2,1,0))
        images=self.__normalise(images)
        images=np.expand_dims(images,axis=3)
        return images,clip
        print("images generated")
    
    def __normalise(self,images):
        """image normalization and transform in fl32"""
        min_im=images.min(axis=(1,2))
        max_im=images.max(axis=(1,2))
        for i in range(max_im.shape[0]):
            images[i,:,:]=(images[i,:,:]-min_im[i])/(max_im[i]-min_im[i])
            #images[i,:,:]=(images[i,:,:])/max_im[i]
        return images
    
    def save_images(self,images,f0lder_path,name='image'):
        num_of_images=images.shape[0]
        for i in range(num_of_images):
            plt.imsave(os.path.join(folder_path,name)+'{0}'.format(i)+'.png',images[i,:,:,0],cmap=plt.cm.gray)# сохранение как одноканальное
    def predict(self,images):
       
        #predictions=self.model.predict(np.expand.dims(self.images,axis=0))
        return self.model.predict(images)
  #для большого файла с проходом окном
    def predict_window(self,recorder):
        images=self.get_images(recorder)
        for i in range(images.shape[0]):
            
            print(self.__labels[np.argmax(self.model.predict(images[i:i+1,:,:,:]))])
            
    #для большого файла батчем
    def predict_record(self,recorder):
        images,clip=self.get_images(recorder)
        predictions=self.predict(images)
        images2 = np.uint8(images[0,:,:,0]*255)
        images2 = cv2.resize(images2,None,fx = 2.6,fy = 2.6, interpolation = cv2.INTER_AREA)
        images2_vis = cv2.applyColorMap(images2,cv2.COLORMAP_OCEAN)
        cv2.imshow("spectrum",images2_vis)
        #print(images.shape,clip.shape)
        label= self.__labels[np.argmax(predictions)]
        if label=='speech':
            predictions=self.model2.predict(images)
            predictions2=predictions[predictions>self.threshold]
            #print(predictions.shape)
            if predictions2.shape[0]!=0:
                indx = np.argmax(predictions)
                label=self.__labels2[indx]
                score = predictions[0,indx]
                #print(label,' score: {0}'.format(np.max(predictions,axis=1)))
            else:
                label = 'not sure'
                score = 1.
                #print(label)
            return label,score,images
        else: return "noise",1., images

class Vis_Spect:
    def __init__(self):
        self.cash = []
        self.cash_len = 8
        self.loaded = False
        
    def store(self,elem):
        self.push(elem)
        
    def load(self,elem):
        if self.loaded:
            print("already loaded")
        else:
            self.update(elem)
            if len(self.cash)==4:
                self.loaded = True
                
    def update(self,elem):
        elem = elem[0,:,:,0]
        if self.loaded:
            self.pop()
            self.push(elem)
        else:
            self.push(elem)
    def push(self,elem):
        self.cash.append(elem)
        
    def free(self):
        self.cash = []
        self.loaded = False
        
    def pop(self):
        _ = self.cash.pop(0)
        
    def show(self,recorder):
        clip,rate =recorder.get_from_micro()
        clip=normalize_input(clip)
        records = np.expand_dims(clip,axis = 0)
        images=wav2img_tensor(records,rate=rate)
        print(images.shape)
        images = images[:,:,0].astype('float32')
        #image = np.concatenate(self.cash, axis = 1)
        #image2 = np.zeros(image.shape)
        #image2 = cv2.normalize(src=image, dst=image2, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        images2 = np.uint8(images*255)
        images2_vis = cv2.applyColorMap(images2,cv2.COLORMAP_OCEAN)
        cv2.imshow("spectrum",images2_vis)
        
        
class Recorder():
    def __init__(self):
        self.__record=None
        self.__rate=None
        self.__stream_alive=False
        self.loaded=False
        self.cash=None
    def getFromArray(self,array,rate):
        pass
   
    def getFromFile(self,path):
        sample_rate, clip=wavfile.read(path)
        self.__record=clip
        self.__rate=sample_rate
        print("file {0} loaded".format(path))
    def Start_stream(self,CHANNELS=1,RATE=16000,CHUNK=4000):
        self.__rate=RATE
        self.__CHUNK=CHUNK
       # if CHUNK<RATE:
        #    self.__stack_chunks()
        self.__stream_alive=True
        self.__audio = pa.PyAudio()
        self.__stream = self.__audio.open(format=pa.paInt16,
                channels=CHANNELS,
                rate=RATE,
                frames_per_buffer=CHUNK,
                input=True)
        #print('Stream is open')
        print("Recognition in progress")
        
    def get_from_micro(self):
        if self.__CHUNK<self.__rate:
            data=self.cash.get()
            data1 = np.frombuffer(self.__stream.read(self.__CHUNK),dtype=np.int16)
            self.cash.append(data1)
        else:
            data = np.frombuffer(self.__stream.read(self.__CHUNK),dtype=np.int16)
        #if self.__CHUNK<RATE:
            #stack_records=np.zeros(int(self.__RATE/self.__CHUNK),self.)
        return data, self.__rate
    
    def stream_active(self):
        return self.__stream_alive
    def load(self):
        if not self.cash:
            self.cash=Cash(RATE=self.__rate,CHUNK=self.__CHUNK)
        if self.cash.free:
            data = np.frombuffer(self.__stream.read(self.__CHUNK),dtype=np.int16)
            #print('append, {}'.format(self.cash.free))
            self.cash.append(data)
        self.loaded= not self.cash.free
    def Stop_stream(self):
        self.__stream_alive=False
        self.__stream.stop_stream()
        self.__stream.close()
        self.__audio.terminate()
        print('Stream is closed')
   
    def push_record(self):
        if self.__record.any():
            return self.__record 
        else:
            return "Audio clip not exist"
    def push_sample_rate(self):
        if self.__rate:
            return self.__rate
        else:
            return "Audio clip rate not exist"
    
def main(path='',mode='Micro',Time=None):
    recorder=Recorder()
    recognizer=Recognizer()
    
    if path=='':
            path='/content/example.wav'
    if mode=='Micro':
        recorder.Start_stream()
        if Time!=None:
            while Time!=0:
                recognizer.predict_record(recorder)
                Time-=1
        if Time==None:
            while True:
                recognizer.predict_record(recorder)
        recorder.Stop_stream()
    else:
        recorder.getFromFile(path)
        recognizer.get_images(recorder)
        predictions=recognizer.predict()
        l=recognizer.get_labels()
        print([l[i] for i in np.argmax(predictions,axis=1)])

def main_microphone(Time=10):
    cv2.imshow("Word",255*np.ones((474,474,3),dtype = np.uint8))
    cv2.waitKey(500)
    print("Open for {0} seconds".format(Time))
    
    #recorder=Recorder()
    #recognizer=Recognizer()
    
    last_label ='none'
    recorder.Start_stream()
    if Time:
        real_time=16000//4000*Time
    
    while not recorder.loaded:
        recorder.load()
        
    while real_time!=0:
        #for i in range(4):
        label,score,image = recognizer.predict_record(recorder)
        if label!="noise":
            if label =="not sure" or label == "other":
                label = "not_sure"
            #print(label,last_label, "score {0}".format(str(score)))
            if last_label!=label:
                if label !="not_sure":
                    if label =='7' and last_label == "8":
                        label = '8'
                    cv2.imshow("Word",images[:,:,:,label_name[label]])
            if label == "not_sure" and label == last_label:
                cv2.imshow("Word",images[:,:,:,13])
        last_label = label
        cv2.waitKey(1)
                
            
        real_time-=1
    recorder.Stop_stream()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cv2.destroyAllWindows()
    PATH = r"C:\Users\Roman2\Desktop"
    Time = 100
    file_names = [str(i) for i in range(11)]
    label_name = {'0':0, '1':1,'2':2, '3':3,'4':4,'5':5,'6':6,'7':7, '8':8,'9':9,"10":10, 'no':11,'not_sure':13,'yes':12}
    file_names.extend(["no","yes","not_sure"])
    images = np.zeros((474,474,3,len(file_names)),dtype = np.uint8)
    for index,name in enumerate(file_names):
        file_path  = os.path.join(PATH,name+".png")
        #print(file_path,index)
        images[:,:,:,index]  = cv2.imread(file_path)

    print("Testing here")
    #cv2.imshow("Word",255*np.ones((474,474,3),dtype = np.uint8))
    #cv2.waitKey(5000)
        #cv2.waitKey(100)
    #time.sleep(10)
        #Time = int(input("write time of recording(in seconds): "))
        
        #images,records=test_main()#(mode='Micro')
        #start = input()
        #if start =="start":
        #break
    #main(Time = 500)
    #cash=test_new_loader()
    recorder=Recorder()
    recognizer=Recognizer()
    vis = Vis_Spect()
    main_microphone(Time)
    cv2.destroyAllWindows()