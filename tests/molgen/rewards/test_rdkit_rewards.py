import pytest
from rdkit import Chem
from rdkit.Chem.QED import qed

from molgen.rewards.functions.rdkit_rewards import QEDReward, SASReward


@pytest.fixture
def reward_fn_scale():
    return QEDReward(scale="mult")


@pytest.fixture
def reward_fn():
    return QEDReward()


@pytest.fixture
def thiamine():
    return "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N"


@pytest.fixture
def score(thiamine):
    return qed(Chem.MolFromSmiles(thiamine))


def test_qed_reward(reward_fn, thiamine, score):
    reward = reward_fn(thiamine)
    assert abs(reward - score) == 0.0


def test_qed_reward_scale(reward_fn_scale, thiamine, score):
    reward = reward_fn_scale(thiamine)
    assert abs(reward - 10 * score) == 0.0


def test_qed_reward_eval(reward_fn_scale, thiamine, score):
    reward_fn_scale.eval = True
    reward = reward_fn_scale(thiamine)
    assert abs(reward - score) == 0.0


def test_sas_reward_default_is_normalized():
    reward_fn = SASReward()
    reward = reward_fn("CCO")
    assert 0.0 <= reward <= 1.0


def test_sas_reward_can_disable_normalization():
    reward_fn = SASReward(normalize=False)
    reward = reward_fn("CCO")
    assert reward > 1.0


def test_sas_reward_invalid_smiles_returns_zero():
    reward_fn = SASReward()
    reward = reward_fn("not_a_smiles")
    assert reward == 0.0
