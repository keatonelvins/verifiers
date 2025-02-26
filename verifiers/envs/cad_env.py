from typing import Any, List, Tuple

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.prompts import CAD_PROMPT
from verifiers.utils import preprocess_dataset
from verifiers.parsers import XMLParser
from verifiers.rubrics import CadRubric

from verifiers.envs.simple_env import SimpleEnv

class CadEnv(SimpleEnv):
    def __init__(self,
                 dataset: str = "mm",
                 fields: List[str | Tuple[str, ...]] = ["thought", "action"],
                 **kwargs):
        super().__init__(**kwargs)
        self.parser = XMLParser(fields=fields)
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset
        )
        self.eval_dataset = None
        self.rubric = CadRubric()
    
    def get_dataset(self, **kwargs: Any):
        return self.dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs() 
