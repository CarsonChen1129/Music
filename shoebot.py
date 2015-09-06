__author__ = 'carsonchen'
# Major library imports
import atexit
import pyaudio
from numpy import zeros, short, fromstring, array
from numpy.fft import fft

NUM_SAMPLES = 512
SAMPLING_RATE = 11025

def setup():
    size(350, 260)
    speed(SAMPLING_RATE / NUM_SAMPLES)

_stream = None

def read_fft():
    global _stream
    pa = None

    def cleanup_audio():
        if _stream:
            _stream.stop_stream()
            _stream.close()
        pa.terminate()

    if _stream is None:
        pa = pyaudio.PyAudio()
        _stream = pa.open(format=pyaudio.paInt16, channels=1,
                           rate=SAMPLING_RATE,
                           input=True, frames_per_buffer=NUM_SAMPLES)
        atexit.register(cleanup_audio)

    audio_data  = fromstring(_stream.read(NUM_SAMPLES), dtype=short)
    normalized_data = audio_data / 32768.0

    return fft(normalized_data)[1:1+NUM_SAMPLES/2]

def flatten_fft(scale = 1.0):
    """
    Produces a nicer graph, I'm not sure if this is correct
    """
    for i, v in enumerate(read_fft()):
        yield scale * (i * v) / NUM_SAMPLES

def triple(audio):
    '''return bass/mid/treble'''
    c = audio.copy()
    c.resize(3, 255 / 3)
    return c

def draw():
    '''Draw 3 different colour graphs'''
    global NUM_SAMPLES
    audio = array(list(flatten_fft(scale = 80)))
    freqs = len(audio)
    bass, mid, treble = triple(audio)

    colours = (0.5, 1.0, 0.5), (1, 1, 0), (1, 0.2, 0.5)

    fill(0, 0, 1)
    rect(0, 0, WIDTH, 400)
    translate(50, 200)

    for spectrum, col in zip((bass, mid, treble), colours):
        fill(col)
        for i, s in enumerate(spectrum):
            rect(i, 0, 1, -abs(s))
        else:
            translate(i, 0)

    audio = array(list(flatten_fft(scale = 80)))