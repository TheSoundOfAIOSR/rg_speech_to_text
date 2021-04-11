import torch
import asyncio
import functools
import transformers


import numpy as np

class WaveNet:
    def __init__(self, device="cpu", tokenizer_path=None, model_path=None, use_vad=False):
        self.device = torch.device(device)
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.use_vad = use_vad

    def load_model(self):
        tokenizer = (transformers.Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
                if self.tokenizer_path is None else torch.load(self.tokenizer_path))
        model = (transformers.Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") 
                    if self.model_path is None else torch.load(self.model_path))
        model.eval()
        model.to(self.device)
        self.model = model
        self.tokenizer = tokenizer

        if self.use_vad:
            self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                   model='silero_vad')

            self.vad_utils = {
                'get_speech_ts': utils[0],
                'save_audio': utils[1],
                'read_audio': utils[2],
            }

            self.audio_buffer = np.array([])
            #self.counter = 0

    def _transcribe(self, inputs):
        if self.use_vad:
            inputs = np.concatenate((self.audio_buffer, inputs))
            # time stamps
            voice_ts = self.vad_utils['get_speech_ts'](torch.Tensor(inputs), self.vad_model)
            if len(voice_ts) == 0: # no voice detected
                self.audio_buffer = np.array([])
                return self.transformer_transcribe(inputs) # process it anyway (since VAD may be wrong)

            cur_st = voice_ts[0]['start']
            cur_ed = len(inputs)
            for ts in voice_ts:
                if ts['end'] == len(inputs):
                    break
                cur_ed = ts['end']

            if cur_ed == len(inputs):
                self.audio_buffer = inputs[cur_st:cur_ed]
                return ''

            self.audio_buffer = inputs[cur_ed+1:]
            inputs = inputs[cur_st:cur_ed]

            # for saving to file the chunks separated by VAD
            #self.vad_utils['save_audio'](f'chunk{self.counter}.wav', torch.Tensor(inputs), 16000)
            #self.counter += 1
            return '<' + self.transformer_transcribe(inputs) + '>'
        else:
            return self.transformer_transcribe(inputs)


    def transformer_transcribe(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors='pt').input_values.to(self.device)
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
    