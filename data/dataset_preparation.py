# data/dataset_preparation.py
import json
import os
# from torch.utils.data import Dataset, DataLoader # Not strictly needed for Phase 1 inference list
from typing import List, Dict, Any, Optional


class MathProblem:
    """Simple class to represent a math problem, can be expanded."""

    def __init__(self, problem_id: str, question: str, answer: str, full_answer_text: Optional[str] = None):
        self.problem_id = problem_id
        self.question = question
        self.answer = answer  # Expected to be the extracted numerical answer
        self.full_answer_text = full_answer_text  # Original full answer text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.problem_id,
            "question": self.question,
            "answer": self.answer,
            "full_answer_text": self.full_answer_text
        }


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extracts the final numerical answer from GSM8K's format."""
    parts = answer_text.split("####")
    final_numeric_answer = parts[-1].strip()
    # Further clean common artifacts like leading/trailing newlines or "The final answer is "
    final_numeric_answer = final_numeric_answer.replace(",", "")  # Remove commas from numbers
    # Try to convert to number and back to string to standardize format (e.g., remove .0 from integers)
    try:
        num = float(final_numeric_answer)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        # If not a simple number, return the stripped version
        return final_numeric_answer.strip()


def load_gsm8k_dataset(file_path: str) -> List[Dict[str, Any]]:
    dataset = []
    print(f"Attempting to load GSM8K dataset from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    question = data.get("question")
                    full_answer_text = data.get("answer", "")

                    if question and full_answer_text:
                        extracted_answer = extract_gsm8k_answer(full_answer_text)
                        dataset.append({
                            "id": f"gsm8k_{os.path.basename(file_path).split('.')[0]}_{i}",
                            "problem": question,  # Key expected by inference_pipeline
                            "answer": extracted_answer,  # Extracted numerical answer
                            "question_original": question,  # Keep original key as well
                            "answer_full_original": full_answer_text  # Keep full original answer
                        })
                    else:
                        print(f"Warning: Skipping line {i + 1} in {file_path} (missing 'question' or 'answer').")
                except json.JSONDecodeError:
                    print(f"Warning: JSON decode error on line {i + 1} in {file_path}.")
        print(f"Successfully loaded {len(dataset)} problems from {file_path}.")
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {file_path}")
    except Exception as e:
        print(f"ERROR: Unexpected error loading {file_path}: {e}")
    return dataset


def load_dataset(file_path: str, dataset_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    General dataset loading function.
    If dataset_type is not provided, it tries to infer from file_path.
    """
    if dataset_type is None:  # Try to infer
        if "gsm8k" in file_path.lower():
            dataset_type = "gsm8k"
        else:
            print(f"Warning: Could not infer dataset type for {file_path}. Defaulting to GSM8K loader.")
            dataset_type = "gsm8k"

    if dataset_type.lower() == "gsm8k":
        return load_gsm8k_dataset(file_path)
    # Add loaders for other datasets (e.g., MATH, SVAMP) here
    # elif dataset_type.lower() == "math":
    #     return load_math_dataset(file_path) # You'd need to implement this
    else:
        print(f"ERROR: Unknown dataset type '{dataset_type}' specified.")
        return []

# PyTorch Dataset class (more for training phases)
# from torch.utils.data import Dataset
# class MathProblemDataset(Dataset):
#     def __init__(self, problems_data: List[Dict[str, Any]]):
#         self.problems_data = problems_data
#     def __len__(self):
#         return len(self.problems_data)
#     def __getitem__(self, idx) -> Dict[str, Any]:
#         return self.problems_data[idx]