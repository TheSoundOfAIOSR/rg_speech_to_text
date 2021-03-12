import numpy as np

class StreaminDecoderASR:
    def __init__(self, model_definition, asr_model, data_layer, data_loader,
                 frame_overlap=2.5, offset=10):
        '''
        Args:
          model_definition: 
          asr_model:
          data_layer:
          data_loader:
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.asr_model = asr_model
        self.data_layer = data_layer
        self.data_loader = data_loader
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')
        timestep_duration = model_definition['AudioToMelSpectrogramPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.offset = offset

    # inference method for audio signal (single instance)
    def infer_signal(self, signal, merge):
        self.data_layer.set_signal(signal)
        batch = next(iter(self.data_loader))
        audio_signal, audio_signal_len = batch
        audio_signal = audio_signal.to(self.asr_model.device)
        audio_signal_len = audio_signal_len.to(self.asr_model.device)
        log_probs, encoded_len, predictions = self.asr_model.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        logits = log_probs.cpu().numpy()[0]
        # print(logits.shape, encoded_len, predictions)
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap], 
            self.vocab
        )
        unmerged = decoded[:len(decoded) - self.offset]
        if not merge:
            return unmerged
        return self.greedy_merge(unmerged)

    def reset(self):
        '''
        Reset decoder's state
        '''
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s

    def greedy_merge(self, s):
        s_merged = ''
        
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged
