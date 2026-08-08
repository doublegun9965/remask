from types import SimpleNamespace
import unittest

import torch

from common.generation.generation import generate_unified


class ConfidenceShiftModel:
    """Predict token 1 first, then withdraw support and settle on token 2."""

    dtype = torch.float32

    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, **kwargs):
        del kwargs
        self.calls += 1
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 4), -6.0)
        generation = input_ids[:, 1:]

        if self.calls == 1:
            preferred = torch.ones_like(generation)
        else:
            preferred = torch.full_like(generation, 2)

        logits[:, 1:, :].scatter_(
            -1, preferred.unsqueeze(-1), torch.full((*preferred.shape, 1), 6.0)
        )
        return SimpleNamespace(logits=logits)


class ThresholdRemaskTest(unittest.TestCase):
    def test_low_current_token_confidence_rolls_back_and_recovers(self):
        model = ConfidenceShiftModel()
        result = generate_unified(
            model=model,
            prompt=torch.tensor([[0]]),
            remasking="fastdllm",
            thres=0.7,
            gen_length=2,
            block_length=2,
            mask_id=3,
            model_type="LLaDA",
            remask_threshold=0.5,
            remask_min_age=1,
            max_remasks_per_token=2,
            remask_max_extra_steps=4,
        )

        self.assertEqual(result.sequences[0, 1:].tolist(), [2, 2])
        self.assertEqual(result.remask_counts.tolist(), [2])
        self.assertEqual(result.steps_taken.tolist(), [4])
        self.assertEqual(model.calls, 4)
        self.assertEqual(result.remask_history.sum().item(), 2)

    def test_disabled_remask_preserves_original_fastdllm_behavior(self):
        model = ConfidenceShiftModel()
        result = generate_unified(
            model=model,
            prompt=torch.tensor([[0]]),
            remasking="fastdllm",
            thres=0.7,
            gen_length=2,
            block_length=2,
            mask_id=3,
            model_type="LLaDA",
        )

        self.assertEqual(result.sequences[0, 1:].tolist(), [1, 1])
        self.assertEqual(result.remask_counts.tolist(), [0])
        self.assertEqual(result.steps_taken.tolist(), [1])
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
