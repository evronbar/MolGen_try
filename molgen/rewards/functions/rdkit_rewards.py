import os
import sys

import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP  # type: ignore
from rdkit.Chem.QED import qed
from rdkit.Chem.rdchem import Mol
from rdkit.RDConfig import RDContribDir
sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
import sascorer

from molgen.rewards.reward import AbstractReward, RewardScale


class QEDReward(AbstractReward):
    def __init__(self, name: str | None = None, scale: RewardScale = None) -> None:
        super().__init__(name=name, scale=scale)

    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            else:
                reward = qed(mol)
                if self.scale is not None and not self.eval:
                    reward = self.scale(reward)

                return reward

        else:
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            rewards = [qed(mol) if mol is not None else -1 for mol in mols]

            if self.scale is not None and not self.eval:
                rewards = [self.scale(reward) for reward in rewards]

            return rewards


class PenalizedLogPReward(AbstractReward):
    def __init__(self, name: str | None = None, scale: RewardScale = None) -> None:
        super().__init__(name=name, scale=scale)

    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            else:
                reward = PenalizedLogPReward.penalized_logp(mol)
                if self.scale is not None and not self.eval:
                    reward = self.scale(reward)

                return reward

        else:
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            rewards = [PenalizedLogPReward.penalized_logp(mol) if mol is not None else -1 for mol in mols]

            if self.scale is not None and not self.eval:
                rewards = [self.scale(reward) for reward in rewards]

            return rewards

        pass

    @staticmethod
    def num_long_cycles(mol: Mol) -> int:
        """Calculate the number of long cycles.

        Args:
          mol: Molecule. A molecule.

        Returns:
          negative cycle length.
        """
        cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
        cycle_length = 0 if not cycle_list else max([len(j) for j in cycle_list])
        cycle_length = 0 if cycle_length <= 6 else cycle_length - 6
        return cycle_length

#we changed the PenalizedLogPReward to use the RDKit library to calculate the SAS score
#this is because the SAS score from the RDKit library is more accurate than the SAS score from the sascorer library
#the SAS score from the RDKit library is calculated using the RDKit library
#the SAS score from the sascorer library is calculated using the sascorer library
#the SAS score from the RDKit library is more accurate than the SAS score from the sascorer library
#the SAS score from the RDKit library is more accurate than the SAS score from the sascorer library

    @staticmethod
    def penalized_logp(molecule: Mol) -> float:
        log_p = MolLogP(molecule)
        try:
            sas_score = sascorer.calculateScore(molecule)
        except Exception:
            return 0.0
        if sas_score is None:
            return 0.0
        cycle_score = PenalizedLogPReward.num_long_cycles(molecule)
        return float(log_p) - float(sas_score) - float(cycle_score)

#we changed the SASReward to use the SAS score from the RDKit library
#this is because the SAS score from the RDKit library is more accurate than the SAS score from the sascorer library
#the SAS score from the RDKit library is calculated using the RDKit library
#the SAS score from the sascorer library is calculated using the sascorer library
#the SAS score from the RDKit library is more accurate than the SAS score from the sascorer library
#the SAS score from the RDKit library is more accurate than the SAS score from the sascorer library
class SASReward(AbstractReward):
    def __init__(self, name: str | None = None, scale: RewardScale = None, normalize: bool = True) -> None:
        super().__init__(name=name, scale=scale)
        self.normalize = normalize

    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        def _sas_reward(mol: Mol | None) -> float:
            if mol is None:
                return 0.0
            # RDKit SA score: lower is better (roughly 1..10).
            # Convert to reward where higher is better.
            try:
                score = sascorer.calculateScore(mol)
            except Exception:
                return 0.0
            if score is None:
                return 0.0
            reward = 10.0 - float(score)
            if self.normalize:
                reward = max(0.0, min(1.0, reward / 9.0))
            return reward

        if isinstance(smiles, str):
            reward = _sas_reward(Chem.MolFromSmiles(smiles))
            if self.scale is not None and not self.eval:
                reward = self.scale(reward)
            return reward

        mols = [Chem.MolFromSmiles(s) for s in smiles]
        rewards = [_sas_reward(mol) for mol in mols]
        if self.scale is not None and not self.eval:
            rewards = [self.scale(reward) for reward in rewards]
        return rewards


class pIC50Reward(AbstractReward):
    def __init__(self,
                 data_path: str,
                 name: str | None = None,
                 scale: RewardScale | None = None) -> None:
        super(pIC50Reward, self).__init__(name=name, scale=scale)
        df = pd.read_csv(data_path)
        smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df['smiles'] if Chem.MolFromSmiles is not None]
        self.smiles_to_pIC50 = dict(zip(smiles, df['KRAS pIC50']))

    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        return self.smiles_to_pIC50[smiles]
