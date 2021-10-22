import torch
from TheSoundOfAIOSR.stt.language_model import LanguageModel
from TheSoundOfAIOSR.stt.wavenet.decoder.beam_search_decoder import BeamSearchDecoder
from TheSoundOfAIOSR.stt.wavenet.decoder.vocab import vocab_list


class CTCDecoder:
    def __init__(self, blank_idx=0, beam_width=100, lm_path=None):
        lm = None
        if beam_width <= 1:
            self.mode = "greedy"
        else:
            self.mode = "beam"
            if lm_path is not None:
                self.mode = "beam_lm"
                lm = LanguageModel()
                lm.load(lm_path)
        self._beam_search = BeamSearchDecoder(vocab_list[1:],
                                              blank_idx,
                                              beam_width,
                                              lm=lm)

    def __call__(self, logits):
        out_proba = torch.nn.functional.softmax(logits, dim=-1)[0]
        if self.mode == "greedy":
            out = self._greedy_path(out_proba)
        elif self.mode == "beam":
            out = self._beam_search(out_proba.numpy())[0]
        elif self.mode == "beam_lm":
            out = self._beam_search(out_proba.numpy())[0]
        else:
            out = None
            raise ValueError("Mode not defined mode choices [greedy, beam and beam_lm]")
        return out

    def _greedy_path(self, logits: torch.tensor):
        out_proba = torch.nn.functional.softmax(logits, dim=-1)
        return torch.argmax(out_proba, axis=1)
