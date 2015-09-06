__author__ = 'carsonchen'

# # -*- coding: utf-8 -*-
# import wave
# import pylab as pl
# import numpy as np
#
#
# f = wave.open(r"/Users/carsonchen/Desktop/test_music/ding.wav", "rb")
#
#
# # (nchannels, sampwidth, framerate, nframes, comptype, compname)
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
#
#
# str_data = f.readframes(nframes)
# f.close()
#
#
# wave_data = np.fromstring(str_data, dtype=np.short)
# wave_data.shape = -1, 2
# wave_data = wave_data.T
# time = np.arange(0, nframes) * (1.0 / framerate)
#
#
# pl.subplot(211)
# pl.plot(time, wave_data[0])
# pl.subplot(212)
# pl.plot(time, wave_data[1], c="g")
# pl.xlabel("time (seconds)")
# pl.show()


# import wave
# import numpy as np
# import pylab as pl
#
# # ============ test the algorithm =============
# # read wave file and get parameters.
# fw = wave.open('/Users/carsonchen/Desktop/test_music/ding.wav','rb')
# params = fw.getparams()
# print(params)
# nchannels, sampwidth, framerate, nframes = params[:4]
# strData = fw.readframes(nframes)
# waveData = np.fromstring(strData, dtype=np.int16)
# waveData = waveData*1.0/max(abs(waveData))  # normalization
# fw.close()
#
# # plot the wave
# time = np.arange(0, len(waveData)) * (1.0 / framerate)
#
# index1 = 10000.0 / framerate
# index2 = 10512.0 / framerate
# index3 = 15000.0 / framerate
# index4 = 15512.0 / framerate
#
# pl.subplot(311)
# pl.plot(time, waveData)
# pl.plot([index1,index1],[-1,1],'r')
# pl.plot([index2,index2],[-1,1],'r')
# pl.plot([index3,index3],[-1,1],'g')
# pl.plot([index4,index4],[-1,1],'g')
# pl.xlabel("time (seconds)")
# pl.ylabel("Amplitude")
#
# pl.subplot(312)
# pl.plot(np.arange(512),waveData[10000:10512],'r')
# pl.plot([59,59],[-1,1],'b')
# pl.plot([169,169],[-1,1],'b')
# print(1/( (169-59)*1.0/framerate ))
# pl.xlabel("index in 1 frame")
# pl.ylabel("Amplitude")
#
# pl.subplot(313)
# pl.plot(np.arange(512),waveData[15000:15512],'g')
# pl.xlabel("index in 1 frame")
# pl.ylabel("Amplitude")
# pl.show()

# Plot the sine wave with the frequency f1= 156.25, f2=234.375
import numpy as np
import pylab as pl

# Assignment section
# sampling_rate   = 8000
# fft_size        = 512
# t               = np.arange(0, 1.0, 1.0/sampling_rate)
# f1              = 156.25
# f2              = 234.375
# x               = np.sin(2*np.pi*f1*t) + 2*np.sin(2*np.pi*f2*t)        # Wave equation
# xs              = x[:fft_size]
# xf              = np.fft.rfft(xs)/fft_size
# freqs           = np.linspace(0, sampling_rate/2, fft_size/2+1)
# xfp             = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
#
# # Graph section
# pl.figure(figsize=(8,4))
# pl.subplot(211)
# pl.plot(t[:fft_size], xs)
# pl.xlabel(u"Time(sec)")
# pl.title(u"156.25Hz and 234.375Hz")
# pl.subplot(212)
# pl.plot(freqs, xfp)
# pl.xlabel(u"Frequency(Hz)")
# pl.subplots_adjust(hspace=0.4)
# pl.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import wave
# import sys
#
#
# spf = wave.open('/Users/carsonchen/Desktop/test_music/ding.wav','r')
#
# #Extract Raw Audio from Wav File
# signal = spf.readframes(-1)
# signal = np.fromstring(signal, 'Int16')
# fs = spf.getframerate()
#
# #If Stereo
# if spf.getnchannels() == 2:
#     print 'Just mono files'
#     sys.exit(0)
#
#
# Time=np.linspace(0, len(signal)/fs, num=len(signal))
#
# plt.figure(1)
# plt.title('Signal Wave...')
# plt.plot(Time,signal)
# plt.show()

from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

# Load the data and calculate the time of each sample
samplerate, data    = wavfile.read('/Users/carsonchen/Desktop/test_music/ding.wav')
times               = np.arange(len(data))/float(samplerate)

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
plt.fill_between(times, data[:,0], data[:,1], color='k')
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.savefig('plot.png', dpi=100)
plt.show()