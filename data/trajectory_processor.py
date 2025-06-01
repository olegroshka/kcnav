# data/trajectory_processor.py
import json
from typing import List, Dict, Any, Tuple, Optional
import torch  # For tensor creation
import math  # For log scaling

# Define encoding for step types. This should be consistent.
STEP_TYPE_ENCODING = {
    "ThoughtAction": 0,
    "Observation": 1,
    "ErrorLog": 2,
}
STEP_TYPE_DIM = len(STEP_TYPE_ENCODING)


def encode_step_type(step_type_str: str) -> List[float]:
    encoding = [0.0] * STEP_TYPE_DIM
    if step_type_str in STEP_TYPE_ENCODING:
        encoding[STEP_TYPE_ENCODING[step_type_str]] = 1.0
    else:
        print(f"Warning: Unknown step_type '{step_type_str}' encountered during encoding.")
    return encoding


def process_single_trajectory_for_training(
        trajectory_data: Dict[str, Any],
        feature_dim: int,
        include_text_embeddings: bool = False
) -> List[Dict[str, Any]]:
    processed_steps = []
    steps = trajectory_data.get("steps", [])
    if not steps:
        return []

    is_successful_trajectory = 1.0 if trajectory_data.get("is_correct_assessment", False) else 0.0
    current_feature_sequence = []

    for i in range(len(steps)):
        current_step = steps[i]
        step_type_str = current_step.get("step_type", "Unknown")
        metadata = current_step.get("metadata", {})
        step_features = []

        # a) Step type (one-hot encoded)
        step_features.extend(encode_step_type(step_type_str))

        # b) Local NCD of current_step
        current_local_ncd = metadata.get("ncd_local", 0.0)
        step_features.append(float(current_local_ncd))

        # c) Global Complexity of current_step (SCALED)
        raw_current_global_complexity = metadata.get("ncd_global", 0.0)
        # Apply log1p scaling: log(x+1) to handle x=0 and compress large values
        scaled_current_global_complexity = math.log1p(float(raw_current_global_complexity))
        step_features.append(scaled_current_global_complexity)

        # d) Time step embedding (normalized index)
        normalized_time_step = (current_step.get("step_index", i + 1) - 1) / max(1, len(steps) - 1) if len(
            steps) > 1 else 0.0
        step_features.append(normalized_time_step)

        if include_text_embeddings:
            pass  # Placeholder

        if len(step_features) != feature_dim:
            # This indicates a mismatch between expected feature_dim and actual features generated.
            # The training script now calculates feature_dim based on this processor's logic,
            # so this should ideally not be triggered if STEP_TYPE_DIM + 3 matches.
            print(
                f"Critical Warning: Mismatch in feature dimensions for problem {trajectory_data.get('problem_id')}, step {i}. "
                f"Generated {len(step_features)} features, but model expects {feature_dim}. "
                f"Features: {step_features}. STEP_TYPE_DIM is {STEP_TYPE_DIM}.")
            # Pad or truncate, though this might hide issues.
            if len(step_features) < feature_dim:
                step_features.extend([0.0] * (feature_dim - len(step_features)))
            else:
                step_features = step_features[:feature_dim]

        current_feature_sequence.append(step_features)

        if i < len(steps) - 1:
            next_step = steps[i + 1]
            next_step_metadata = next_step.get("metadata", {})

            next_step_ncd_local_target = float(next_step_metadata.get("ncd_local", 0.0))

            raw_next_global_ncd_target = float(next_step_metadata.get("ncd_global", 0.0))
            # Apply log1p scaling to the global NCD target as well
            scaled_next_step_ncd_global_target = math.log1p(raw_next_global_ncd_target)

            processed_steps.append({
                "feature_sequence": [fs[:] for fs in current_feature_sequence],
                "next_step_ncd_local_target": next_step_ncd_local_target,
                "next_step_ncd_global_target": scaled_next_step_ncd_global_target,  # Use scaled value
                "trajectory_success_target": is_successful_trajectory,
                "problem_id": trajectory_data.get("problem_id"),
                "target_step_index": next_step.get("step_index", i + 2)
            })

    return processed_steps


if __name__ == '__main__':
    # Example Usage (for testing this processor)
    dummy_traj_data = {
        "problem_id": "dummy_problem_1",
        "is_correct_assessment": True,
        "steps": [
            {"step_index": 1, "step_type": "ThoughtAction", "content": "TA1",
             "metadata": {"ncd_local": 0.5, "ncd_global": 10}},
            {"step_index": 2, "step_type": "Observation", "content": "O1",
             "metadata": {"ncd_local": 0.2, "ncd_global": 1500}},  # Made one global NCD large
            {"step_index": 3, "step_type": "ThoughtAction", "content": "TA2",
             "metadata": {"ncd_local": 0.6, "ncd_global": 2200}},
        ]
    }
    EXPECTED_FEATURE_DIM = STEP_TYPE_DIM + 3

    processed_data_points = process_single_trajectory_for_training(dummy_traj_data, feature_dim=EXPECTED_FEATURE_DIM)

    print(f"Generated {len(processed_data_points)} training data points from dummy trajectory:")
    for point in processed_data_points:
        print(f"  Problem ID: {point['problem_id']}")
        print(f"  Targeting NCDs for original step index: {point['target_step_index']}")
        print(
            f"  Feature Sequence (Example last step's global NCD feature): {point['feature_sequence'][-1][STEP_TYPE_DIM + 1]:.3f}")  # Print scaled global NCD feature
        print(f"  Next Local NCD Target: {point['next_step_ncd_local_target']:.3f}")
        print(
            f"  Next Global NCD Target (Scaled): {point['next_step_ncd_global_target']:.3f}")  # Print scaled global NCD target
        print(f"  Trajectory Success Target: {point['trajectory_success_target']}")
        print("-" * 10)
