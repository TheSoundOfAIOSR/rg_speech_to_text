import numpy as np
import torch

# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameASR:
    
    def __init__(self, streaming_decoder, sample_rate,
                 frame_len=2, frame_overlap=2.5):
        '''
        Args:
          streaming_decoder: streaming_decoder based on the model
          sample_rate: #TODO model_definition['sample_rate']
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
        '''
        self.streaming_decoder = streaming_decoder
        
        self.n_frame_len = int(frame_len * sample_rate)
        self.buffer = np.zeros(
            shape=2*int(frame_overlap * sample_rate) + self.n_frame_len,
            dtype=np.float32)
        self.reset()
        
    def _decode(self, frame, merge):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        return self.streaming_decoder.infer_signal(self.buffer, merge)
    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        return self._decode(frame, merge)
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.streaming_decoder.reset()

