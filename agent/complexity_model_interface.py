# agent/complexity_model_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Forward reference for Trajectory if needed, or import directly
# from .trajectory import Trajectory

class ComplexityModelInterface(ABC):
    """
    Abstract Base Class for complexity models (Transformer, Diffusion, etc.).
    Defines the interface expected by the ReActAgent for guidance.
    """

    @abstractmethod
    def load_model(self, path: str, device: Optional[str] = None):
        """
        Load the trained complexity model from a specified path.
        Args:
            path: Path to the saved model file.
            device: Optional device ('cpu', 'cuda') to load the model onto.
        """
        pass

    @abstractmethod
    def score_candidates(self, trajectory: Any, candidates_text: List[str]) -> List[float]:
        """
        Scores candidate next steps based on complexity alignment.
        Args:
            trajectory: The current Trajectory object (or relevant history representation).
            candidates_text: A list of strings, each representing a potential next step's
                             raw text output from LLM (e.g., "Thought: ... Action: ...").
        Returns:
            A list of scores, one for each candidate, where higher scores indicate better
            alignment with desirable complexity patterns. Length must match candidates_text.
        """
        pass

    @abstractmethod
    def evaluate_trajectory_appropriateness(self, trajectory: Any) -> Dict[str, Any]:
        """
        Evaluates the overall complexity appropriateness of a completed trajectory.
        Provides guidance/scores for the self-reflection phase.
        Args:
            trajectory: The completed Trajectory object (or its representation).
        Returns:
            A dictionary containing analysis results, e.g.,
            {'overall_score': 0.85, 'problematic_steps': [3, 7],
             'guidance': 'High NCD at step 3 suggests potential inefficiency...'}
        """
        pass