#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""A single offline example for verifying the local generation stack."""

import torch

from data.loaders.gsm8k import GSM_SYSTEM_PROMPT


class SmokeDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_examples=0, subsample=-1, **kwargs):
        del num_examples, subsample, kwargs
        self.tokenizer = tokenizer
        self.question = "What is 17 plus 25?"
        self.answer = "42"

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        messages = [
            {
                "role": "user",
                "content": f"{GSM_SYSTEM_PROMPT}\n\n{self.question}",
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return prompt + "<reasoning>", self.question, self.answer

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        encoded = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )
        return {
            "input_ids": encoded.input_ids,
            "questions": [item[1] for item in batch],
            "answers": [item[2] for item in batch],
            "prompts": prompts,
            "attention_mask": encoded.attention_mask,
        }
