# agent/react_agent.py
import math
import time
import re
from typing import Optional, Tuple, List, Dict, Any

from .llm_wrapper import LlmWrapper
from .tools import ToolManager
from .complexity_calculator import NcdCalculator
from .trajectory import Trajectory
from .complexity_model_interface import ComplexityModelInterface


class ReActAgent:
    def __init__(self,
                 llm: LlmWrapper,
                 tool_manager: ToolManager,
                 ncd_calculator: NcdCalculator,
                 complexity_model: ComplexityModelInterface,  # Will be Placeholder for Phase 1
                 max_steps: int = 10,
                 num_candidates_rerank: int = 1,
                 cfg: Dict = {}):  # Default to 1 for Phase 1 baseline

        self.llm = llm
        self.tool_manager = tool_manager
        self.ncd_calculator = ncd_calculator
        self.complexity_model = complexity_model
        self.max_steps = max_steps
        self.num_candidates_rerank = max(1, num_candidates_rerank)  # Ensure at least 1
        self.use_reranking = self.num_candidates_rerank > 1
        self.cm_risk_low = cfg.get("complexity_risk_low", 0.0)
        self.cm_risk_high = cfg.get("complexity_risk_high", float("inf"))
        self.cm_gamma = cfg.get("complexity_gamma", 0.01)

        print(f"ReActAgent Initialized:")
        print(f"  LLM Provider: {llm.llm_provider}")
        print(f"  Max Steps: {self.max_steps}")
        print(
            f"  Re-ranking: {'Enabled' if self.use_reranking else 'Disabled'} (Candidates: {self.num_candidates_rerank})")
        print(f"  NCD Compressor: {ncd_calculator.compressor_name}")
        print(f"  Complexity Model Type: {type(complexity_model).__name__}")

    def _format_react_prompt(self, trajectory: Trajectory) -> str:
        # This prompt structure is crucial.
        # For local models, especially smaller ones, a very clear, structured prompt is key.
        # Some models might benefit from explicit examples of Thought/Action/Action Input format.
        prompt = (
            "You are a precise and methodical assistant solving a math problem step-by-step. "
            "Think carefully before each step. Your goal is to reach the correct final answer. "
            "When you need to calculate, use the 'Calculator' tool. "
            "When you have the final numerical answer, use the 'FinalAnswer' action. "
            "Follow this format strictly for each step:\n"
            "Thought: [Your reasoning for the next step, analysis of the current state, and plan. Be concise but clear.]\n"
            "Action: [Tool name (e.g., Calculator) or FinalAnswer]\n"
            "Action Input: [Input for the tool, or the final numerical answer for FinalAnswer]\n\n"
        )
        # Example of a good step (optional to include in prompt, but good for model to learn):
        # "Example of a step:\n"
        # "Thought: I need to calculate 2 plus 2.\n"
        # "Action: Calculator\n"
        # "Action Input: 2 + 2\n\n"
        # "Example of a final step:\n"
        # "Thought: I have calculated the final answer to be 4.\n"
        # "Action: FinalAnswer\n"
        # "Action Input: 4\n\n"

        prompt += f"Problem: {trajectory.problem_statement}\n\n"

        # Include tool descriptions
        prompt += f"{self.tool_manager.get_tool_descriptions()}\n\n"

        prompt += "Current Reasoning Trajectory (Previous Steps):\n"
        prompt += "----------------------------\n"
        if not trajectory.steps:
            prompt += "(No steps taken yet. This is your first step.)\n"
        else:
            # Serialize history, excluding the problem statement itself from here
            history_str = trajectory.get_full_serialized_history(include_problem=False, step_separator="\n\n")
            prompt += history_str + "\n"  # Add a newline after the history
        prompt += "----------------------------\n\n"
        prompt += "Based on the problem and the trajectory so far, provide your next Thought, Action, and Action Input.\nThought:"  # Prompt LLM to start with Thought
        return prompt

    def _format_reflection_prompt(self, trajectory: Trajectory, complexity_analysis: Dict[str, Any]) -> str:
        # This won't be used in Phase 1 baseline (enable_reflection=false)
        # but is here for completeness.
        prompt = ("You are an expert reviewer analyzing a reasoning process for a math problem.\n"
                  # ... (full reflection prompt as designed previously) ...
                  "Critique:")
        return prompt

    def _select_best_candidate(
            self,
            trajectory: "Trajectory",
            candidates_text: List[str]
    ) -> str:
        """
        1. ask the complexity model for a *score* (higher == better)
           we interpret score = exp( -γ · risk )  →  risk = -ln(score)/γ
        2. discard candidates whose risk is outside [risk_low, risk_high]
        3. among survivors pick the highest-score one;
           if none survive pick the overall lowest-risk candidate
        """

        if not candidates_text:  # should never happen
            print("[ReAct] WARNING: 0 candidates; returning fallback string")
            return "Error: LLM produced no candidate."

        if len(candidates_text) == 1:  # nothing to rank
            return candidates_text[0]

        # score
        try:
            scores = self.complexity_model.score_candidates(trajectory, candidates_text)
        except Exception as e:
            print(f"[ReAct] complexity model failed ({e}) – using first cand.")
            return candidates_text[0]

        if len(scores) != len(candidates_text):
            print("[ReAct] Score/candidate length mismatch – using first cand.")
            return candidates_text[0]

        #  risk ↔ score
        gamma = self.cm_gamma # Default gamma value from config
        risks = [-math.log(max(s, 1e-12)) / gamma for s in scores]

        survivors = [
            (c, r, s) for c, r, s in zip(candidates_text, risks, scores)
            if self.cm_risk_low <= r <= self.cm_risk_high
        ]

        pick_from = survivors if survivors else list(
            zip(candidates_text, risks, scores)
        )

        # highest score among the allowed set
        best = max(pick_from, key=lambda t: t[2])[0]

        return best

    def _select_best_candidate_prev(self, trajectory: Trajectory, candidates_text: List[str]) -> str:
        # This is called if self.use_reranking is True.
        # For Phase 1 baseline, num_candidates_rerank is 1, so this won't select from multiple.
        if not candidates_text:
            # This should ideally not happen if LLM generates something.
            print("ERROR: _select_best_candidate called with empty candidates list.")
            return "Error: No candidates generated."  # Fallback
        if len(candidates_text) == 1:
            return candidates_text[0]

        print(
            f"Scoring {len(candidates_text)} candidates using complexity model '{type(self.complexity_model).__name__}'...")
        try:
            scores = self.complexity_model.score_candidates(trajectory, candidates_text)
        except Exception as e:
            print(f"ERROR during complexity model scoring: {e}. Selecting first candidate as fallback.")
            return candidates_text[0]

        if len(scores) != len(candidates_text):
            print(
                f"ERROR: Scores count mismatch. Scores: {len(scores)}, Candidates: {len(candidates_text)}. Selecting first candidate.")
            return candidates_text[0]

        best_candidate_index = scores.index(max(scores))
        best_candidate = candidates_text[best_candidate_index]
        # print(f"Selected candidate {best_candidate_index + 1}/{len(candidates_text)} with score {scores[best_candidate_index]:.4f}")
        return best_candidate

    def _run_react_loop(self, problem_id: str, problem_statement: str) -> Trajectory:
        trajectory = Trajectory(problem_id, problem_statement)
        last_step_content_for_ncd = problem_statement  # Initial content for first NCD calc

        for step_num in range(self.max_steps):
            current_step_index = step_num + 1
            # print(f"\n--- Agent Step {current_step_index}/{self.max_steps} (Problem ID: {problem_id}) ---")

            prompt = self._format_react_prompt(trajectory)

            thought, action_type, action_input = "", "", ""
            llm_full_output_text = ""  # The raw text of the chosen thought/action step

            if self.use_reranking and self.num_candidates_rerank > 1:
                try:
                    candidates = self.llm.generate_candidates(prompt, self.num_candidates_rerank)
                    if not candidates:
                        print("WARNING: LLM generated no candidates. Falling back to single generation.")
                        llm_full_output_text, thought, action_type, action_input = self._generate_single_thought_action(
                            prompt)
                    else:
                        selected_output_text = self._select_best_candidate(trajectory, candidates)
                        llm_full_output_text = selected_output_text
                        thought, action_type, action_input = self.llm._parse_thought_action(selected_output_text)
                except Exception as e:
                    print(f"ERROR during candidate generation/selection: {e}. Attempting single generation.")
                    llm_full_output_text, thought, action_type, action_input = self._generate_single_thought_action(
                        prompt)
            else:  # Single generation (Phase 1 baseline path)
                llm_full_output_text, thought, action_type, action_input = self._generate_single_thought_action(prompt)

            if action_type == "Error":  # Indicates LLM call or parsing failed critically
                print(f"CRITICAL ERROR from LLM/Parsing: {thought}. Stopping.")
                trajectory.add_step(step_type="ErrorLog",
                                    content=f"LLM/Parsing Error: {thought}. Raw output: {llm_full_output_text}",
                                    metadata={})
                trajectory.set_status("error_llm_critical")
                break

            # Calculate NCDs using the raw chosen LLM output for the ThoughtAction step
            ncd_local = self.ncd_calculator.calculate_local_ncd(last_step_content_for_ncd, llm_full_output_text)
            history_for_global_ncd = trajectory.get_full_serialized_history(
                include_problem=True) + f"\n\nStep {len(trajectory.steps) + 1} (ThoughtAction):\n{llm_full_output_text}"
            ncd_global = self.ncd_calculator.calculate_global_complexity(history_for_global_ncd)
            step_metadata = {"ncd_local": ncd_local, "ncd_global": ncd_global, "raw_llm_output": llm_full_output_text}

            # Add the combined Thought/Action as one step to the trajectory
            # The content stored is the raw LLM output for this turn.
            trajectory.add_step(step_type="ThoughtAction", content=llm_full_output_text, metadata=step_metadata)
            last_step_content_for_ncd = llm_full_output_text  # Update for next NCD

            # Process the parsed action
            if action_type.lower() == "finalanswer":
                # print(f"  Action: FinalAnswer, Input: '{action_input}'")
                trajectory.final_answer = action_input  # Store the answer
                trajectory.set_status("completed")
                # print("--- Loop Finished (FinalAnswer) ---")
                break
            elif action_type == "ContinueThought" or not action_type:  # No valid action, just thought
                # print(f"  Action: ContinueThought (Thought: '{thought[:50]}...')")
                # The thought is already part of llm_full_output_text and added. Loop will continue.
                if not thought:  # If thought is also empty, it's problematic
                    print("Warning: ContinueThought action but thought is empty. Potentially stuck.")
            elif action_type == "ParsingError":
                print(f"  Action: ParsingError (Thought: '{thought[:50]}...'). Will try to continue.")
                # The problematic output is already logged in ThoughtAction. Agent will try to generate next step.
            elif action_type:  # A tool action
                # print(f"  Action: {action_type}, Input: '{action_input}'")
                observation = self.tool_manager.execute_tool(action_type, action_input)
                # print(f"  Observation: {observation}")

                obs_ncd_local = self.ncd_calculator.calculate_local_ncd(last_step_content_for_ncd, observation)
                obs_metadata = {"tool_name": action_type, "tool_input": action_input, "ncd_local": obs_ncd_local}
                trajectory.add_step(step_type="Observation", content=observation, metadata=obs_metadata)
                last_step_content_for_ncd = observation  # Observation becomes context for next NCD
            else:  # Should be caught by ParsingError or ContinueThought
                print(f"WARNING: No action type identified, thought was: '{thought[:50]}...'. Stopping.")
                trajectory.set_status("error_no_action")
                break

            if current_step_index >= self.max_steps:
                print(f"--- Loop Finished (Max Steps '{self.max_steps}' Reached) ---")
                trajectory.set_status("error_max_steps")
                break

            # time.sleep(0.1) # Small delay, might not be needed for local models

        return trajectory

    def _generate_single_thought_action(self, prompt: str) -> Tuple[str, str, str, str]:
        """Helper to call LLM for a single thought/action and return its raw output too."""
        raw_llm_output, parsed_thought, parsed_action_type, parsed_action_input = "", "", "", ""
        try:
            # The generate_thought_action method in LlmWrapper already handles parsing.
            # We need the raw output before parsing for NCD.
            # So, we'll call the more basic generation method from LlmWrapper if possible,
            # or reconstruct. For now, LlmWrapper.generate_thought_action returns parsed.
            # Let's assume LlmWrapper.generate_thought_action is the primary way.
            # If we need raw output, LlmWrapper needs to be adapted or we call its internal _call_..._api.

            # Simplified: let LlmWrapper handle the call and parsing.
            # We'll reconstruct an approximate raw_output if needed for NCD.
            parsed_thought, parsed_action_type, parsed_action_input = self.llm.generate_thought_action(prompt)

            # Reconstruct an approximate raw output string for NCD calculation and logging
            # This is not perfect but often good enough.
            raw_llm_output = f"Thought: {parsed_thought}\nAction: {parsed_action_type}\nAction Input: {parsed_action_input}"
            if parsed_action_type == "Error":  # If LLM call itself failed
                raw_llm_output = f"Error in LLM generation: {parsed_thought}"

        except Exception as e:
            print(f"Exception in _generate_single_thought_action: {e}")
            parsed_thought = f"Error during LLM call: {e}"
            parsed_action_type = "Error"
            raw_llm_output = f"Error during LLM call: {e}"

        return raw_llm_output, parsed_thought, parsed_action_type, parsed_action_input

    def _run_reflection(self, initial_trajectory: Trajectory) -> Tuple[Trajectory, Optional[str]]:
        # This is not used in Phase 1 baseline.
        print("\n--- Starting Self-Reflection Phase (Placeholder for Phase 1) ---")
        if initial_trajectory.status.startswith("error"):
            print("Skipping reflection due to error in initial ReAct loop.")
            return initial_trajectory, "Reflection skipped due to prior error."

        # 1. Evaluate Complexity Appropriateness
        try:
            complexity_analysis = self.complexity_model.evaluate_trajectory_appropriateness(initial_trajectory)
        except Exception as e:
            print(f"ERROR during complexity model evaluation for reflection: {e}")
            critique = f"Error during complexity evaluation: {e}"
            initial_trajectory.set_critique(critique)
            return initial_trajectory, critique

        # 2. Format Reflection Prompt
        reflection_prompt = self._format_reflection_prompt(initial_trajectory, complexity_analysis)

        # 3. Generate Critique
        try:
            critique = self.llm.generate_critique(reflection_prompt)
            initial_trajectory.set_critique(critique)
        except Exception as e:
            print(f"ERROR during critique generation: {e}")
            critique = f"Error during critique generation: {e}"
            initial_trajectory.set_critique(critique)
            return initial_trajectory, critique

        # print(f"Reflection Critique Generated (first 150 chars): {critique[:150]}...")
        # No actual refinement step in Phase 1
        print("--- Reflection Phase Finished (Critique Stored) ---")
        return initial_trajectory, critique

    def run(self, problem_id: str, problem_statement: str, enable_reflection: bool = False) -> Trajectory:
        # print(f"\n{'='*10} Running Agent for Problem ID: {problem_id} {'='*10}")
        # print(f"Problem: {problem_statement[:100]}...")
        # print(f"Agent Settings: Re-ranking={self.use_reranking}, Reflection={enable_reflection}")

        # Run the main ReAct loop
        final_trajectory = self._run_react_loop(problem_id, problem_statement)

        if enable_reflection:  # This will be false for Phase 1 baseline
            final_trajectory, _ = self._run_reflection(final_trajectory)

        # print(f"{'='*10} Finished Agent for Problem ID: {problem_id}. Final Status: {final_trajectory.status} {'='*10}")
        return final_trajectory