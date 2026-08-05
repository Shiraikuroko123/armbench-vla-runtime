from __future__ import annotations

import numpy as np
import pytest

from integrations.openpi.realtime_chunking import prefix_attention_weights
from integrations.openpi.rtc_guidance_g0 import _prefix_weights
from integrations.openpi.rtc_guidance_g0 import _shift_reference


@pytest.mark.parametrize("schedule", ["linear", "exp", "ones", "zeros"])
def test_g0_weights_match_independent_rtc_reference(schedule: str) -> None:
    expected = prefix_attention_weights(4, 5, 10, schedule)
    actual = _prefix_weights(4, 5, 10, schedule)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    assert actual.dtype == np.dtype("<f4")


def test_g0_reference_uses_rtc_suffix_shift_and_zero_padding() -> None:
    actions = np.arange(70, dtype=np.float32).reshape(10, 7)

    reference = _shift_reference(actions, 5)

    np.testing.assert_array_equal(reference[:5], actions[5:])
    np.testing.assert_array_equal(reference[5:], 0.0)
    assert reference.flags.c_contiguous


@pytest.mark.parametrize(
    ("delay", "execute", "message"),
    [(-1, 5, "nonnegative"), (6, 5, "inconsistent"), (4, 11, "inconsistent")],
)
def test_g0_rejects_invalid_overlap_horizons(
    delay: int, execute: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _prefix_weights(delay, execute)
