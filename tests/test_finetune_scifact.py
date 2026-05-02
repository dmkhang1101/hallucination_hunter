"""Unit tests for `audit.tasks.finetune_scifact` runtime helpers.

Covers only the small device/precision shims — the actual fine-tune is an
integration job that runs on Lightning, not in CI.
"""

import unittest
from unittest.mock import patch

from audit.tasks import finetune_scifact


class FineTuneSciFactRuntimeConfigTests(unittest.TestCase):
    def test_pipeline_uses_first_gpu_when_accelerator_is_available(self) -> None:
        with patch.object(finetune_scifact.torch.cuda, "is_available", return_value=True):
            self.assertEqual(finetune_scifact._get_pipeline_device(), 0)

    def test_pipeline_uses_cpu_when_no_accelerator_is_available(self) -> None:
        with patch.object(finetune_scifact.torch.cuda, "is_available", return_value=False):
            self.assertEqual(finetune_scifact._get_pipeline_device(), -1)

    def test_precision_flags_force_fp32(self) -> None:
        # Mixed precision destabilized DeBERTa-v3-large's classifier head on
        # SciFact (collapsed to a single class). fp32 is the contract now.
        self.assertEqual(
            finetune_scifact._get_precision_flags(),
            {"bf16": False, "fp16": False},
        )


if __name__ == "__main__":
    unittest.main()
