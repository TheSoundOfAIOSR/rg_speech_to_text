'''
Example Snippets
'''
import sys
sys.path.insert(0, '../..')

import asyncio
from TheSoundOfAIOSR.audiointerface.capture import MicrophoneStreaming, AudioStreaming, AudioReader

## audio path to load
audio_path = "input/Achievements_of_the_Democratic_Party_(Homer_S._Cummings).ogg"


## load full audio in memory (not recommended)
audio_reader = AudioReader(audio_path=audio_path, sr=16000, dtype="float32")
data, sr = audio_reader.read()
# do whatever with data
print(data)



## load audio data as streaming
stream = AudioStreaming(audio_path=audio_path, 
                            blocksize=16000*2, 
                            overlap=0, 
                            padding=0, 
                            sr=16000, 
                            dtype="float32")
for block in stream.generator():
    # process here
    print(len(block))



## saving recording to audio file
filename = "hello.wav"
duration = 10
## saving recording to audio file
async def save_audio():
    stream = MicrophoneStreaming()
    await stream.record_to_file(filename=filename, duration=duration)

print("start recording")
try:
    asyncio.run(save_audio())
except KeyboardInterrupt:
    print("Exited")


#  microphone streaming
async def capture():
    stream = MicrophoneStreaming()
    async for block, status in stream.generator():
        # process data here
        print(len(block))
try:
    asyncio.run(capture())
except KeyboardInterrupt:
    print("Exited")