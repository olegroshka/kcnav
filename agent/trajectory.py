# agent/trajectory.py
import json
from typing import List, Dict, Any, Optional
import time  # For timestamping steps


class Trajectory:
    def __init__(self, problem_id: str, problem_statement: str):
        self.problem_id = problem_id
        self.problem_statement = problem_statement
        self.steps: List[Dict[str, Any]] = []
        self.final_answer: Optional[str] = None  # Initial final answer from ReAct loop
        self.status: str = "incomplete"
        self.critique: Optional[str] = None
        self.refined_answer: Optional[str] = None  # Answer after reflection (if any)
        self.metadata: Dict[str, Any] = {"created_at": time.time()}  # General metadata for the trajectory

    def add_step(self, step_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        step_data = {
            "step_index": len(self.steps) + 1,  # 1-indexed steps
            "step_type": step_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self.steps.append(step_data)
        if step_type.lower() == "finalanswer":  # Handle if FinalAnswer is part of content
            # The ReActAgent's _run_react_loop sets self.final_answer more directly
            pass

    def get_full_serialized_history(self, include_problem=True, step_separator="\n") -> str:
        history_parts = []
        if include_problem:
            history_parts.append(f"Problem: {self.problem_statement}")

        for step in self.steps:
            prefix = f"Step {step['step_index']} ({step['step_type']}):"

            # Ensure content is string and handle potential multi-line content
            step_content_str = str(step['content']) if step['content'] is not None else ""
            content_lines = step_content_str.splitlines()

            if len(content_lines) > 1:
                # Indent subsequent lines for readability
                formatted_content = content_lines[0] + "\n" + "\n".join([f"  {line}" for line in content_lines[1:]])
            elif len(content_lines) == 1:
                formatted_content = content_lines[0]
            else:  # Empty content
                formatted_content = ""

            history_parts.append(f"{prefix}\n{formatted_content}")

            # Optionally include some metadata if it's simple and useful for LLM context
            # For NCD calculation, raw text is usually better.
            # if step.get('metadata') and 'ncd_local' in step['metadata']:
            #     history_parts.append(f"  (Local NCD: {step['metadata']['ncd_local']:.3f})")
        return step_separator.join(history_parts)

    def set_status(self, status: str):
        self.status = status
        self.metadata["updated_at"] = time.time()

    def set_critique(self, critique: str):
        self.critique = critique
        self.metadata["critiqued_at"] = time.time()

    def set_refined_answer(self, answer: str):
        self.refined_answer = answer
        self.metadata["refined_at"] = time.time()

    def get_final_answer_for_evaluation(self) -> Optional[str]:
        """Returns the refined answer if available, otherwise the original final answer."""
        return self.refined_answer if self.refined_answer is not None else self.final_answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem_statement": self.problem_statement,
            "steps": self.steps,
            "final_answer_initial": self.final_answer,  # From ReAct loop
            "final_answer_refined": self.refined_answer,  # After reflection
            "status": self.status,
            "critique": self.critique,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        traj = cls(data['problem_id'], data['problem_statement'])
        traj.steps = data.get('steps', [])
        traj.final_answer = data.get('final_answer_initial')
        traj.refined_answer = data.get('final_answer_refined')
        traj.status = data.get('status', 'unknown')
        traj.critique = data.get('critique')
        traj.metadata = data.get('metadata', {"created_at": time.time()})  # Add default if missing
        return traj

    def __str__(self) -> str:
        final_ans_eval = self.get_final_answer_for_evaluation()
        return (f"Trajectory(id='{self.problem_id}', steps={len(self.steps)}, "
                f"status='{self.status}', final_answer_eval='{str(final_ans_eval)[:50]}...')")
