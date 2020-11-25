import matplotlib.pyplot as plt
import numpy as np
import argparse
import pywt
import scipy.io.wavfile as wavfile
sr=44100 #sample rate

def getChannels(image):
    """
    input is image path to png image, output is 3 matrices, for each rgb chnnel
    """
    X = np.array(plt.imread(image, format='png'))
    red = X[:,:,0]
    green =  X[:,:,1]
    blue = X[:,:,2]
    return red,green,blue

def convertImage(greyimage,length,wavelet,threshold):
    """
    input is a 2d matrix to be converted to sound, the length of the sound, and the type of wavelet used in the conversion
    """
    Y = pywt.wavedec(np.zeros(int(sr*opt.length)), opt.wavelet) #make empty coefficient "bands" for filling up with image data
    dims = greyimage.shape
    greyimage = np.flip(greyimage,axis=0) #flip the image so the high frequnecies are on top and low freq on bottom
    stepsize = int(np.floor(dims[0]/len(Y)))
    for i,j in enumerate(Y):
        level2=pywt.wavedec(j, opt.wavelet) #do another wavelet transform for each coeff band
        stepsize2 = int(stepsize/len(level2))
        for p,q in enumerate(level2):
            currentblock = greyimage[(i*stepsize)+(p*stepsize2):(i*stepsize)+((p+1)*stepsize2),:]
            currentblock = currentblock.flatten(order='F') #try np.average(currentblock,axis=0) 
            currentblock = (np.clip(currentblock,threshold,1.0)-threshold) #threshold to maintain sparseness (sounds better)
            level2[p]=np.interp(np.linspace(0,1,len(level2[p])),np.linspace(0,1,len(currentblock)),currentblock)
        Y[i] = np.sin(i/len(Y)*np.pi)*pywt.waverec(level2,wavelet)[:len(Y[i])] #sin factor is to damp sub freqs and high freqs, remove if you dont mind
    return pywt.waverec(Y,wavelet)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to PNG file to convert to sound")
    parser.add_argument('--length', type=float, required=False, default=10, help="Length in seconds")
    parser.add_argument('--threshold', type=float, required=False, default=0.5, help="convert bottom x to black")
    parser.add_argument('--wavelet', type=str, required=False, default="db38", help="wavelet type, db, sym etc")
    parser.add_argument('--output', type=str, required=False, default="output.wav", help="Path to wav file where converted image will be saved")
    opt = parser.parse_args()
    left,right,center = getChannels(opt.input)
    X = convertImage(center,opt.length,opt.wavelet,opt.threshold)
    Xl = convertImage(left,opt.length,opt.wavelet,opt.threshold)
    Xr = convertImage(right,opt.length,opt.wavelet,opt.threshold)
    X0 = X+Xl
    X1 = X+Xr
    X0 = X0/np.max(np.abs(X0))
    X1 = X1/np.max(np.abs(X1))
    wavfile.write(opt.output, sr, np.array([X0,X1],dtype='float32').T)
    print("done. and saved at {}".format(opt.output))
