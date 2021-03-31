import torch
import asyncio
import functools
import transformers

class WaveNet:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def load_model(self, tokenizer_path=None, model_path=None):
        tokenizer = (transformers.Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
                if tokenizer_path is None else torch.load(tokenizer_path))
        model = (transformers.Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") 
                    if model_path is None else torch.load(model_path))
        model.eval()
        model.to(self.device)
        self.model = model
        self.tokenizer = tokenizer

    def transcribe(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors='pt').input_values.to(self.device)
        logits = self.model(inputs).logits
        predicted_ids = torch.argmax(logits, dim =-1)
        return self.tokenizer.decode(predicted_ids[0])

    async def capture_and_transcribe(self, stream_obj, loop=None):
        if loop is None:
            loop = asyncio.get_running_loop()
        async for block, status in stream_obj.generator():
            process_func = functools.partial(self.transcribe, inputs=block)
            transcriptions = await loop.run_in_executor(None, process_func)
            yield transcriptions