"""Round-trip UT for the SWA FP8 pack/unpack hack.

TEMP -- pending real fp8 attention rewrite, see plan section 1.1.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4._swa_fp8_pack import pack_swa_fp8, unpack_swa_fp8

HEAD_DIM = 512
SLOT_BYTES = 584


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class SwaFp8PackTest(unittest.TestCase):
    def _round_trip(self, src: torch.Tensor) -> torch.Tensor:
        packed = pack_swa_fp8(src)
        self.assertEqual(packed.shape, (src.shape[0], SLOT_BYTES))
        self.assertEqual(packed.dtype, torch.uint8)
        out = unpack_swa_fp8(packed)
        self.assertEqual(out.shape, src.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        return out

    def test_round_trip_unit_magnitude(self):
        torch.manual_seed(0)
        src = torch.randn(64, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        out = self._round_trip(src)
        s = src.float()
        o = out.float()
        rel = (o - s).abs() / (s.abs() + 1e-3)
        self.assertLess(rel.max().item(), 0.5)
        self.assertLess(rel.mean().item(), 0.05)

    def test_round_trip_large_magnitude(self):
        torch.manual_seed(1)
        src = torch.randn(16, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 100.0
        out = self._round_trip(src)
        s = src.float()
        o = out.float()
        rel = (o - s).abs() / (s.abs() + 1e-3)
        self.assertLess(rel.max().item(), 0.5)
        self.assertLess(rel.mean().item(), 0.05)

    def test_round_trip_zero_rows(self):
        src = torch.zeros(8, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        out = self._round_trip(src)
        self.assertTrue(torch.equal(out, src))

    def test_empty(self):
        src = torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        packed = pack_swa_fp8(src)
        self.assertEqual(packed.shape, (0, SLOT_BYTES))
        out = unpack_swa_fp8(packed)
        self.assertEqual(out.shape, (0, HEAD_DIM))
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
