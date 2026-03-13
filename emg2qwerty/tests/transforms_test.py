# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from emg2qwerty.transforms import ChannelMask, Resample


def test_channel_mask_noop():
    tensor = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    masked = ChannelMask()(tensor)
    assert torch.equal(masked, tensor)


def test_channel_mask_num_channels():
    tensor = torch.ones(3, 2, 4)
    masked = ChannelMask(num_channels=2)(tensor)

    assert torch.equal(masked[..., :2], tensor[..., :2])
    assert torch.equal(masked[..., 2:], torch.zeros_like(tensor[..., 2:]))


def test_channel_mask_keep_indices():
    tensor = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    masked = ChannelMask(keep_indices=[1, 3])(tensor)

    assert torch.equal(masked[..., 1], tensor[..., 1])
    assert torch.equal(masked[..., 3], tensor[..., 3])
    assert torch.equal(masked[..., 0], torch.zeros_like(tensor[..., 0]))
    assert torch.equal(masked[..., 2], torch.zeros_like(tensor[..., 2]))


def test_resample_noop_when_rates_match():
    tensor = torch.randn(8, 2, 4)
    resampled = Resample(orig_freq=2000, new_freq=2000)(tensor)
    assert torch.equal(resampled, tensor)


def test_resample_downsamples_time_dimension():
    tensor = torch.randn(8, 2, 4)
    resampled = Resample(orig_freq=2000, new_freq=1000)(tensor)

    assert resampled.shape == (4, 2, 4)
