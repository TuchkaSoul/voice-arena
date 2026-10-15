from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analyzer.models import Wav2Vec2AntiSpoof, SR


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_audio_matches_model_dtype_and_mask_remains_integer(dtype):
    class Model:
        def __call__(self, input_values, attention_mask):
            assert input_values.dtype == self.dtype
            assert attention_mask.dtype == torch.int64
            return SimpleNamespace(logits=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=self.dtype))

    def processor(chunks, **kwargs):
        assert len(chunks) == 2
        assert len(chunks[0]) == SR * 5
        assert len(chunks[1]) == SR  # последний фрагмент дополняется до секунды
        return {"input_values": torch.zeros(2, SR * 5, dtype=torch.float32),
                "attention_mask": torch.ones(2, SR * 5, dtype=torch.int64)}

    detector = Wav2Vec2AntiSpoof.__new__(Wav2Vec2AntiSpoof)
    detector.device = torch.device("cpu")
    detector.model = Model()
    detector.model.dtype = dtype
    detector.processor = processor
    detector.spoof_idx = 1
    assert detector.predict(np.zeros(SR * 5 + SR // 2, dtype=np.float32)) == pytest.approx(0.5, abs=0.001)
