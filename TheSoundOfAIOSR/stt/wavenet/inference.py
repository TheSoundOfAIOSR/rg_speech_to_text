import torch
import asyncio
import functools
import transformers


import numpy as np

class WaveNet:
    def __init__(self, device="cpu", tokenizer_path=None, model_path=None):
        self.device = torch.device(device)
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path

    def load_model(self):
        tokenizer = (transformers.Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
                if self.tokenizer_path is None else torch.load(self.tokenizer_path))
        model = (transformers.Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") 
                    if self.model_path is None else torch.load(self.model_path))
        model.eval()
        model.to(self.device)
        self.model = model
        self.tokenizer = tokenizer

        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad')

        self.vad_utils = {
            'get_speech_ts': utils[0],
            'save_audio': utils[1],
            'read_audio': utils[2],
            # 'get_speech_ts': utils[0],
            # 'get_speech_ts': utils[0],
            # 'get_speech_ts': utils[0],
        }

        self.audio_buffer = np.array([])
        self.counter = 0

    def _transcribe(self, inputs):
        # print(inputs[:10])
        # CHECK AGAIN IF CUT is OK
        inputs = np.concatenate((self.audio_buffer, inputs))
        # time stamps
        voice_ts = self.vad_utils['get_speech_ts'](torch.Tensor(inputs), self.vad_model)
        #print(len(inputs), voice_ts)
        if len(voice_ts) == 0:
            self.audio_buffer = inputs
            return ''

        cur_st = voice_ts[0]['start']
        cur_ed = len(inputs)
        for ts in voice_ts:
            if ts['end'] == len(inputs):
                break
            cur_ed = ts['end']
        
        if cur_ed == len(inputs):
            cur_st = 0
            self.audio_buffer = inputs[cur_st:cur_ed]
            return ''

        # print(cur_st, cur_ed)
        to_transcribe = inputs[cur_st:cur_ed]
        self.audio_buffer = inputs[cur_ed+1:]

        self.vad_utils['save_audio'](f'chunk{self.counter}.wav', torch.Tensor(to_transcribe), 16000)
        self.counter += 1

        inputs = self.tokenizer(to_transcribe, return_tensors='pt').input_values.to(self.device)
        logits = self.model(inputs).logits
        predicted_ids = torch.argmax(logits, dim =-1)
        return self.tokenizer.decode(predicted_ids[0])

    async def capture_and_transcribe(self,
                                     stream_obj,
                                     started_future: asyncio.Future = None,
                                     loop = None):
        if loop is None:
            loop = asyncio.get_running_loop()
        async for block, status in stream_obj.generator(started_future):
            process_func = functools.partial(self._transcribe, inputs=block)
            transcriptions = await loop.run_in_executor(None, process_func)
            yield transcriptions

    async def transcribe(self, input, loop = None):
        process_func = functools.partial(self._transcribe, inputs=input)
        return await loop.run_in_executor(None, process_func)
    