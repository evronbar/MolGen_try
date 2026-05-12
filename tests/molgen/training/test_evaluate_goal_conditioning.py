import pytest

from molgen.training.evaluate import _normalize_goal_conditioning


def test_normalize_goal_conditioning_rejects_scalar_for_multi_goal():
    with pytest.raises(ValueError, match="Expected 3 RTG targets, got scalar input"):
        _normalize_goal_conditioning(ret=1.0, goal_idx=None, n_goals=3)


def test_normalize_goal_conditioning_rejects_wrong_rtg_len():
    with pytest.raises(ValueError, match="Expected 3 RTG targets, got 2"):
        _normalize_goal_conditioning(ret=[0.8, 0.6], goal_idx=[[0], [1], [2]], n_goals=3)


def test_normalize_goal_conditioning_rejects_scalar_goal_for_multi_goal():
    with pytest.raises(ValueError, match="Expected 2 goal index streams, got scalar input"):
        _normalize_goal_conditioning(ret=[0.8, 0.6], goal_idx=0, n_goals=2)


def test_normalize_goal_conditioning_allows_single_goal_scalar():
    rtgs, goals = _normalize_goal_conditioning(ret=1.0, goal_idx=0, n_goals=1)
    assert rtgs == [[1.0]]
    assert goals == [[0]]


def test_normalize_goal_conditioning_explicit_multi_goal_passes():
    rtgs, goals = _normalize_goal_conditioning(ret=[0.8, 0.6], goal_idx=[[0], [1]], n_goals=2)
    assert rtgs == [[0.8], [0.6]]
    assert goals == [[0], [1]]
