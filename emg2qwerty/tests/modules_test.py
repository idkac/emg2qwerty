# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from emg2qwerty.modules import RecurrentEncoder


def test_recurrent_encoder_respects_input_lengths():
    torch.manual_seed(0)

    encoder = RecurrentEncoder(
        input_size=4,
        hidden_size=3,
        num_layers=1,
        rnn_type="gru",
        bidirectional=True,
        dropout=0.0,
    )
    encoder.eval()

    short = torch.randn(5, 1, 4)
    long = torch.randn(8, 1, 4)
    batched = nn.utils.rnn.pad_sequence([short[:, 0], long[:, 0]])
    input_lengths = torch.tensor([5, 8], dtype=torch.int64)

    standalone = encoder(short)
    batched_outputs = encoder(batched, input_lengths=input_lengths)

    assert batched_outputs.shape == (8, 2, 6)
    assert torch.allclose(
        batched_outputs[: short.size(0), 0],
        standalone[:, 0],
        atol=1e-6,
    )
