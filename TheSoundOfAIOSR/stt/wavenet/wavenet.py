import torch
import asyncio
import functools
import transformers

class WaveNet:
    def __init__(self, device="cpu", tokenzer_path=None, model_path=None):
        self.device = torch.device(device)
        self.tokenizer_path = tokenzer_path
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

    def transcribe(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors='pt').input_values.to(self.device)
        logits = self.model(inputs).logits
        predicted_ids = torch.argmax(logits, dim =-1)
        return self.tokenizer.decode(predicted_ids[0])

    async def capture_and_transcribe(self, stream_obj,
                                     started_future: asyncio.Future = None, loop = None):
        if loop is None:
            loop = asyncio.get_running_loop()
        async for block, status in stream_obj.generator(started_future):
            process_func = functools.partial(self.transcribe, inputs=block)
            transcriptions = await loop.run_in_executor(None, process_func)
            yield transcriptions