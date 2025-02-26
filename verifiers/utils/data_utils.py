import random
import json
from typing import List, Dict

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
            "prompt": format_prompt(format_question(x), system_prompt + "\n\nReturn only the letter of the correct answer.", few_shot, fewshot_prob),
            "answer": x["answerKey"]
        })
        return dataset
    elif dataset_name == "mm":
        dataset: Dataset = load_dataset("keatone/mm_rl") # type: ignore
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

def compare_dict(response_dict: Dict[str, Any], truth_dict: Dict[str, Any]) -> bool:
    """
    Compares two dictionaries and returns a boolean value.
    Handles nested dictionaries and lists, sorting them before comparison.
    """
    if len(response_dict) != len(truth_dict):
        print(f"Length mismatch: {len(response_dict)} != {len(truth_dict)}")
        return False
    
    # Convert keys to sets for order-independent comparison
    if set(response_dict.keys()) != set(truth_dict.keys()):
        print(f"Key mismatch: {set(response_dict.keys())} != {set(truth_dict.keys())}")
        return False
    
    for key in response_dict:
        response_val = response_dict[key]
        truth_val = truth_dict[key]
        
        # Handle nested dictionaries
        if isinstance(response_val, dict) and isinstance(truth_val, dict):
            if not compare_dict(response_val, truth_val):
                print(f"Nested dict mismatch: {response_val} != {truth_val}")
                return False
        # Handle lists by comparing elements individually if not sortable
        elif isinstance(response_val, list) and isinstance(truth_val, list):
            if len(response_val) != len(truth_val):
                print(f"List length mismatch: {len(response_val)} != {len(truth_val)}")
                return False
            
            # Convert lists to sets for order-independent comparison
            response_set = set(make_hashable(response_val))
            truth_set = set(make_hashable(truth_val))
            
            if response_set != truth_set:
                print(f"List mismatch: {response_val} != {truth_val}")
                return False
        # Direct comparison for other types
        elif response_val != truth_val:
            print(f"Mismatch: {response_val} != {truth_val}")
            return False
    
    return True