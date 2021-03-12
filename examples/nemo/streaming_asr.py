#!/usr/bin/env python3
"""
"""
import sys
sys.path.insert(0, '../..')

import time
import numpy as np
import pyaudio as pa

from TheSoundOfAIOSR.stt.nemo.streaming_model import create_quartznet_streaming_asr


# sample rate, Hz
SAMPLE_RATE = 16000
# number of audio channels (expect mono signal)
CHANNELS = 1
# duration of signal frame, seconds
FRAME_LEN = 1.0
# frame overlap
FRAME_OVERLAP = 2
# decoder offset
OFFSET = 4

asr = create_quartznet_streaming_asr(sample_rate=SAMPLE_RATE,
                                     frame_len=FRAME_LEN,
                                     frame_overlap=FRAME_OVERLAP,
                                     offset=OFFSET)

p = pa.PyAudio()
print('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        print(i, dev.get('name'))

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        print('Please type input device ID:')
        dev_idx = int(input())

    empty_counter = 0

    def callback(in_data, frame_count, time_info, status):
        global empty_counter
        signal = np.frombuffer(in_data, dtype=np.int16)
        text = asr.transcribe(signal)
        if len(text):
            print(text,end='')
            empty_counter = OFFSET
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ',end='')
        return (in_data, pa.paContinue)

    stream = p.open(format=pa.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    stream_callback=callback,
                    frames_per_buffer=int(FRAME_LEN * SAMPLE_RATE))

    print('Listening...')

    stream.start_stream()
    
    # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
    try:
        while stream.is_active():
            time.sleep(0.1)
            sys.stdout.flush()
    finally:        
        stream.stop_stream()
        stream.close()
        p.terminate()

        print()
        print("PyAudio stopped")
    
else:
    print('ERROR: No audio input device found.')
    
    