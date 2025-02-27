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
            "prompt": format_prompt(format_question(x), system_prompt + "\n\nReturn only the letter of the correct answer.", few_shot, fewshot_prob),
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

def compare_dict(response_dict: Dict[str, Any], truth_dict: Dict[str, Any], return_similarity: bool = False) -> Union[bool, float]:
    """
    Compares two dictionaries and returns either a boolean value or a similarity score.
    
    Design choices for similarity calculation:
    1. Key overlap: Measures how many keys are shared between dictionaries (30% of final score)
    2. Value similarity: Measures how similar the values are for shared keys (70% of final score)
    3. Structural penalties: 
       - Empty vs non-empty lists receive severe penalties
       - Different "Type" fields result in an 80% reduction in similarity
       - Critical keys like "Type" and "Name" have double/triple penalties when different
    4. Nested structures:
       - Recursive similarity calculation for nested dictionaries
       - Extra penalties for very different nested structures (similarity < 0.3)
    5. List comparison:
       - Combines Jaccard similarity (set overlap) with length ratio
       - Empty lists compared to non-empty lists receive severe penalties
    
    The similarity score is designed to be close to zero for structurally different objects,
    even if they share some keys, and close to one only for nearly identical objects.
    
    Args:
        response_dict: The dictionary to compare.
        truth_dict: The dictionary to compare against.
        return_similarity: If True, returns a float between 0 and 1 indicating similarity.
                          If False, returns a boolean indicating exact match.
    
    Returns:
        bool: True if dictionaries match exactly (when return_similarity=False)
        float: Similarity score between 0 and 1 (when return_similarity=True)
    """
    if not return_similarity:
        # Binary comparison
        if len(response_dict) != len(truth_dict):
            return False
        
        if set(response_dict.keys()) != set(truth_dict.keys()):
            return False
        
        for key in response_dict:
            response_val = response_dict[key]
            truth_val = truth_dict[key]
            
            if isinstance(response_val, dict) and isinstance(truth_val, dict):
                if not compare_dict(response_val, truth_val):
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
    
    # Improved similarity score calculation
    total_keys = len(set(response_dict.keys()).union(set(truth_dict.keys())))
    if total_keys == 0:
        return 1.0  # Empty dictionaries are identical
    
    # Calculate key overlap score
    common_keys = set(response_dict.keys()).intersection(set(truth_dict.keys()))
    key_score = len(common_keys) / total_keys
    
    if not common_keys:
        return 0.0  # No common keys
    
    # Calculate value similarity for common keys
    value_scores = []
    for key in common_keys:
        response_val = response_dict[key]
        truth_val = truth_dict[key]
        
        if isinstance(response_val, dict) and isinstance(truth_val, dict):
            # Recursive similarity for nested dictionaries
            dict_similarity = compare_dict(response_val, truth_val, return_similarity=True)
            value_scores.append(dict_similarity)
            
            # If nested dictionaries are very different, penalize more heavily
            if dict_similarity < 0.3:
                value_scores.append(0.0)  # Add extra penalty
                
        elif isinstance(response_val, list) and isinstance(truth_val, list):
            # List similarity
            if len(response_val) == 0 and len(truth_val) == 0:
                value_scores.append(1.0)  # Empty lists are identical
            elif len(response_val) == 0 or len(truth_val) == 0:
                # If one list is empty and the other isn't, they're completely different
                value_scores.append(0.0)
                # Add extra penalty for structural difference
                value_scores.append(0.0)
            else:
                # Calculate length difference penalty (more severe)
                length_ratio = min(len(response_val), len(truth_val)) / max(len(response_val), len(truth_val))
                
                # Calculate Jaccard similarity for lists
                response_set = set(make_hashable(response_val))
                truth_set = set(make_hashable(truth_val))
                
                if not response_set and not truth_set:
                    value_scores.append(1.0)
                else:
                    intersection = len(response_set.intersection(truth_set))
                    union = len(response_set.union(truth_set))
                    jaccard = intersection / union if union > 0 else 0.0
                    
                    # Combine length ratio and Jaccard similarity with higher penalty
                    list_similarity = jaccard * length_ratio
                    value_scores.append(list_similarity)
                    
                    # Add extra penalty for very different lists
                    if list_similarity < 0.3:
                        value_scores.append(0.0)
        else:
            # Direct comparison for other types
            if response_val == truth_val:
                value_scores.append(1.0)
            else:
                value_scores.append(0.0)
                # Add extra penalty for different values of the same key
                if key in ["Type", "Name"]:  # These keys are especially important
                    value_scores.append(0.0)
                    value_scores.append(0.0)  # Double penalty for critical keys
    
    # Combine key similarity and value similarity with more weight on values
    value_score = sum(value_scores) / len(value_scores) if value_scores else 0.0
    
    # Final similarity with more weight on value similarity
    final_score = (key_score * 0.3) + (value_score * 0.7)
    
    # Additional structural penalty for fundamentally different objects
    if "Type" in response_dict and "Type" in truth_dict:
        if response_dict["Type"] != truth_dict["Type"]:
            final_score *= 0.2  # 80% reduction for different types
    
    return final_score