import re
import json
from typing import List, Dict, Any
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser
from verifiers.utils import compare_dict
from verifiers.bloblang.commands import CamferCommand
import logging

class CadRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["thought", "action"])
        self.logger = logging.getLogger(f"verifiers.rubrics.{self.__class__.__name__}")

        def correctness_reward_func(completions, answer, **kwargs) -> List[float]:
            responses = [self.parser.parse(c[0]['content']).action for c in completions]
            
            def compute_reward(response: str, ground_truth: str) -> float:
                try:
                    response_dict = json.loads(response)
                    truth_dict = json.loads(ground_truth)
                    response_cmd = CamferCommand.model_validate(response_dict)
                    truth_cmd = CamferCommand.model_validate(truth_dict)

                    if response_cmd.root.Name != truth_cmd.root.Name:
                        return 0.0

                    reward = 0.5
                    similarity = compare_dict(response_cmd.root.Args.model_dump(), truth_cmd.root.Args.model_dump(), return_similarity=True)
                    
                    self.logger.info(f"\nResponse: \033[33m{json.dumps(response_dict, indent=4)}\033[0m" + 
                                     f"\nTruth: \033[32m{json.dumps(truth_dict, indent=4)}\033[0m" + 
                                     f"\nSimilarity: \033[34m{similarity}\033[0m")

                    reward += 2.0 * similarity
                    return reward

                except Exception:
                    return -0.1

            return [compute_reward(r, a) for r, a in zip(responses, answer)]

        def xml_reward_func(completions, **kwargs) -> List[float]:
            def count_xml(text: str) -> float:
                count = 0
                for tag in ["thought", "action"]:
                    count += 1 - abs(text.count(f"<{tag}>") - 1)
                    count += 1 - abs(text.count(f"</{tag}>") - 1)
                return 0.1 * count 
            return [count_xml(c[-1]['content']) for c in completions]

        def format_reward_func(completions, **kwargs) -> list[float]:
            """Reward function that checks if the completion has a specific format."""
            pattern = r"^<thought>\n.*?\n</thought>\n<action>\n.*?\n</action>\n$"
            responses = [c[0]["content"] for c in completions]
            matches = [re.match(pattern, r, re.DOTALL) for r in responses] 
            return [0.5 if match else 0.0 for match in matches]
        
        def schema_reward_func(completions, **kwargs) -> list[float]:
            responses = [self.parser.parse(c[0]['content']).action for c in completions]
            
            def validate_schema(response: str) -> float:
                try:
                    response_dict = json.loads(response)
                    try:
                        CamferCommand.model_validate(response_dict)
                        return 1.0
                    except Exception:
                        return 0.1
                except Exception:
                    return -0.5
                
            return [validate_schema(r) for r in responses]

        self.reward_funcs = [
            correctness_reward_func, 
            xml_reward_func, 
            format_reward_func,
            schema_reward_func
        ]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs # type: ignore
    
