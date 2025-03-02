import random
import json
from typing import List, Dict, Any, Union

from datasets import Dataset, load_dataset # type: ignore

def extract_boxed_answer(text: str) -> str | None:
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find \boxed{
    boxed_start = text.find('\\boxed{')
    if boxed_start == -1:
        return text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)
    
    if closing_brace == -1:
        return text
    
    return text[content_start:closing_brace]

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_prompt(prompt: str,
                  system_prompt: str | None = None,
                  few_shot: List[Dict[str, str]] | None = None,
                  fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": prompt})
    return messages

def preprocess_dataset(dataset_name: str = "gsm8k", 
                       split: str = "train",
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, str]] | None = None,
                       fewshot_prob: float = 1.0) -> Dataset:
    if dataset_name == "gsm8k":
        dataset: Dataset = load_dataset("openai/gsm8k", "main")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["question"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_hash_answer(x["answer"])
        })
        return dataset
    elif dataset_name == "math":
        dataset: Dataset = load_dataset("chiayewken/competition_math")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["problem"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_boxed_answer(x["solution"])
        })
        return dataset
    elif dataset_name == "openbookqa":
        dataset: Dataset = load_dataset("allenai/openbookqa", "main")[split] # type: ignore
        
        def format_question(example):
            choices_texts = example['choices']['text']
            choices_labels = example['choices']['label']
            
            formatted_choices = []
            for i in range(len(choices_labels)):
                formatted_choices.append(f"{choices_labels[i]}. {choices_texts[i]}")
            
            question = f"Question: {example['question_stem']}\n\nChoices:\n" + "\n".join(formatted_choices)
            return question
        
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(format_question(x), str(system_prompt) + "\n\nReturn only the letter of the correct answer.", few_shot, fewshot_prob),
            "answer": x["answerKey"]
        })
        return dataset
    elif dataset_name == "mm":
        dataset: Dataset = load_dataset("keatone/cad_rl")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": x["prompt"],
            "answer": x["answer"]
        })
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for preprocess_dataset.")
    
def make_hashable(item):
    """Convert nested lists/dicts into hashable tuples with consistent ordering."""
    if isinstance(item, dict):
        # Sort dictionary items and make values hashable
        return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
    elif isinstance(item, list):
        # Sort the hashable versions of list items for consistent ordering
        hashable_items = [make_hashable(i) for i in item]
        if all(isinstance(x, dict) for x in item):
            # For lists of dictionaries, sort by their hashable representation
            return tuple(sorted(hashable_items))
        return tuple(hashable_items)
    return item

def compare_command(response: Dict[str, Any], truth: Dict[str, Any], return_similarity: bool = False) -> Union[bool, float]:
    """
    Compares two command dicts and returns either a boolean value or a similarity score.
    
    Args:
        response: The command to compare.
        truth: The command to compare against.
        return_similarity: If True, returns a float between 0 and 1 indicating similarity.
                          If False, returns a boolean indicating exact match.
    
    Returns:
        bool: True if dictionaries match exactly (when return_similarity=False)
        float: Similarity score between 0 and 1 (when return_similarity=True)
    """
    if not return_similarity:
        # Binary comparison
        if len(response) != len(truth):
            return False
        
        if set(response.keys()) != set(truth.keys()):
            return False
        
        for key in response:
            response_val = response[key]
            truth_val = truth[key]
            
            if isinstance(response_val, dict) and isinstance(truth_val, dict):
                if not compare_command(response_val, truth_val):
                    return False
            elif isinstance(response_val, list) and isinstance(truth_val, list):
                if len(response_val) != len(truth_val):
                    return False
                
                response_set = set(make_hashable(response_val))
                truth_set = set(make_hashable(truth_val))
                
                if response_set != truth_set:
                    return False
            elif response_val != truth_val:
                return False
        
        return True

    if response["Name"] != truth["Name"]:
        return 0.0

    # Weighted similarity score calculation
    reward = 0.2
    args_t,  args_r = truth["Args"], response["Args"]
    if truth["Name"] == "AddFeature":
        # AddFeature must have the same feature type to get reward
        if args_t["Feature"]["Type"] != args_r["Feature"]["Type"]:
            return reward
        reward += 0.1
        if compare_command(args_t["Feature"], args_r["Feature"]):
            reward += 0.6
        return reward
    elif truth["Name"] in [
        "AddSketchEntities", 
        "AddSketchConstraints", 
        "RemoveSketchEntities", 
        "RemoveSketchConstraints", 
        "ProjectEntitiesToPlane", 
        "SetBodiesToKeep", 
        "GetSplitBodies"
    ]:
        # These commands are an id + List of ids
        if "Id" in args_t and args_t["Id"] == args_r["Id"]:
            reward += 0.2

        if compare_command(args_t, args_r):
            reward += 0.8

        return reward
    else:
        # These commands are just strings
        reward += 0.8 if compare_command(args_t, args_r) else 0
    
    return reward