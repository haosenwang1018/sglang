import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.layers import communicator as comm
from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLongcatShortcut(unittest.TestCase):
    def test_shortcut_contributes_only_moe_output(self):
        for tp in (1, 2):
            for scattered in (False, True):
                for terminal in (False, True):
                    for rows in (0, 4):
                        with self.subTest(
                            tp=tp, scattered=scattered, terminal=terminal, rows=rows
                        ):
                            self._check(tp, scattered, terminal, rows)

    def _check(self, tp, scattered, terminal, rows):
        modes = comm.ScatterMode
        context = comm.CommunicateContext(
            process_group_sizes={
                modes.SCATTERED: 1,
                modes.TP_ATTN_FULL: tp,
                modes.FULL: tp,
            },
            attn_tp_rank=0,
            attn_tp_size=tp,
            attn_dp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            tp_size=tp,
            tp_rank=0,
        )
        source = modes.SCATTERED if scattered else modes.FULL
        target = modes.TP_ATTN_FULL if terminal or not scattered else modes.SCATTERED
        postprocess = comm.CommunicateSummableTensorPairFn.get_fn(
            source,
            modes.SCATTERED if scattered else modes.TP_ATTN_FULL,
            target,
            context,
        )
        local_rows = rows // tp if scattered else rows
        output_rows = rows if terminal or not scattered else local_rows
        fork_hidden = torch.full((local_rows, 3), 2.0)
        fork_residual = torch.full_like(fork_hidden, 5.0)
        preparation = SimpleNamespace(
            prepare_attn=lambda h, r, batch: (h, r),
            prepare_mlp=lambda h, r, batch: (fork_hidden, fork_residual),
            postprocess_layer=lambda h, r, batch: postprocess(h, r, batch, context),
        )
        layer = LongcatFlashDecoderLayer.__new__(LongcatFlashDecoderLayer)
        nn.Module.__init__(layer)
        layer.moe_layer_communicator = preparation
        layer.self_attn = [lambda **kw: kw["hidden_states"]]
        layer.mlp = nn.Identity()
        layer.forward_mlp = Mock(
            return_value=(
                torch.full((output_rows, 3), 3.0),
                torch.full((output_rows, 3), 11.0),
                None,
            )
        )
        with (
            patch.object(
                comm,
                "get_local_dp_buffer",
                side_effect=lambda group: torch.empty(rows, 3),
            ),
            patch.object(
                comm,
                "attn_tp_all_gather_into_tensor",
                side_effect=lambda out, x: out.copy_(x.repeat(tp, 1)),
            ),
            patch.object(
                comm,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_group=object()),
            ),
        ):
            hidden, residual, _ = layer(
                torch.arange(rows), torch.zeros(rows, 3), None, None, None, None
            )
        torch.testing.assert_close(hidden, torch.full((output_rows, 3), 5.0))
        torch.testing.assert_close(
            hidden + residual, torch.full((output_rows, 3), 16.0)
        )
        torch.testing.assert_close(fork_residual, torch.full_like(fork_residual, 5.0))


if __name__ == "__main__":
    unittest.main()
