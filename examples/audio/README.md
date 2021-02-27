This folder contains example applications to check if audio (microphone) capture and playback is working.
At the same time the examples are good reference how to integrate in applications.

`capture_and_playback.py` is useful to test simultaneous capture and play back. A short delay between the input signal and the playback is observed, which is proportional with the buffer size.

`capture_to_file.py` is a command line recorder.

`play_audio_file.wav` is useful to listen back the recorded file.

All the three example applications are memory efficient in the sense that does not depend on the duration of the audio stream, doesn't load in memory more than the buffer size of audio data.

The two capture examples are completely asynchronous and easy to integrate in GUI.