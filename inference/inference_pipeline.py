# inference/inference_pipeline.py
import argparse, yaml, json, os, time, sys, traceback, re
from typing import Optional

try:
    from agent.react_agent import ReActAgent
    from agent.llm_wrapper import LlmWrapper, TRANSFORMERS_AVAILABLE, OPENAI_AVAILABLE
    from agent.tools import ToolManager, CalculatorTool
    from agent.complexity_calculator import NcdCalculator
    from agent.complexity_models import get_complexity_model
    from agent.trajectory import Trajectory
    from data.dataset_preparation import load_dataset
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.react_agent import ReActAgent
    from agent.llm_wrapper import LlmWrapper, TRANSFORMERS_AVAILABLE, OPENAI_AVAILABLE
    from agent.tools import ToolManager, CalculatorTool
    from agent.complexity_calculator import NcdCalculator
    from agent.complexity_models import get_complexity_model
    from agent.trajectory import Trajectory
    from data.dataset_preparation import load_dataset


def extract_numerical_answer_from_string(text: Optional[str]) -> Optional[str]:
    if text is None: return None
    s = str(text).strip().split("####")[-1].strip().replace(",", "").replace("$", "")
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if not match: return s
    try:
        f_val = float(match.group(0)); return str(int(f_val)) if f_val.is_integer() else str(f_val)
    except ValueError:
        return match.group(0)


def run_inference(config: dict):
    print(f"--- Starting Inference/Data Collection ({config.get('experiment_name', 'Default')}) ---")
    exp_out_dir = os.path.join(config.get('trajectory_output_dir', 'data/collected_trajectories'),
                               config.get('experiment_name', 'default_run'))
    os.makedirs(exp_out_dir, exist_ok=True)
    print(f"Output trajectories will be saved to: {exp_out_dir}")

    try:
        llm_p = {"llm_provider": config['llm_provider'], "temperature": config['llm_temperature'],
                 "max_new_tokens": config['llm_max_new_tokens'],
                 "max_tokens_critique": config.get('max_tokens_critique', config['llm_max_new_tokens'])}
        if config['llm_provider'] == 'openai':
            if not OPENAI_AVAILABLE: raise RuntimeError("OpenAI lib missing.")
            llm_p.update({"openai_model_name": config['openai_model_name'], "api_key": os.getenv("OPENAI_API_KEY")})
            if not llm_p["api_key"]: raise ValueError("OPENAI_API_KEY not set.")
        elif config['llm_provider'] == 'local':
            if not TRANSFORMERS_AVAILABLE: raise RuntimeError("Transformers lib missing.")
            llm_p.update({k: config.get(k) for k in ["local_model_id", "local_model_device_map",
                                                     "local_model_trust_remote_code", "local_model_torch_dtype",
                                                     "local_model_quantization"] if config.get(k) is not None})
        llm = LlmWrapper(**llm_p)

        cm_config_keys = [
            'transformer_feature_dim', 'transformer_hidden_size',
            'transformer_num_layers', 'transformer_nhead',
            'diffusion_seq_len', 'diffusion_feature_dim', 'diffusion_hidden_size'
        ]
        cm_params = {k: config[k] for k in cm_config_keys if k in config}
        cm_params['use_gpu'] = config.get('use_gpu', True)

        agent = ReActAgent(llm, ToolManager([CalculatorTool()]), NcdCalculator(config['ncd_compressor']),
                           get_complexity_model(config['complexity_model_type'], config.get('complexity_model_path'),
                                                **cm_params),
                           config['max_steps'], config['num_candidates_rerank'])
    except Exception as e:
        print(f"FATAL Init Error: {e}"); traceback.print_exc(); sys.exit(1)
    print("Components initialized successfully.")

    # Use the new 'inference_source_file' key from config
    source_data_file = config.get('inference_source_file')
    if not source_data_file:
        print("FATAL ERROR: 'inference_source_file' not specified in config.")
        sys.exit(1)

    data_file_path = os.path.join(config['dataset_path'], source_data_file)
    dataset_type_infer = os.path.basename(config['dataset_path'])
    print(f"Loading data for inference/collection from: {data_file_path}")
    source_dataset = load_dataset(data_file_path, dataset_type=dataset_type_infer)

    if not source_dataset: print(f"ERROR: No data loaded from {data_file_path}. Exiting."); sys.exit(1)

    start, limit = config.get('inference_start_index', 0), config.get('inference_limit')
    to_process = source_dataset[start: (start + limit if limit is not None else len(source_dataset))]
    if not to_process: print(
        f"No problems to process (Source: {source_data_file}, Indices: {start} to {start + (limit if limit is not None else len(source_dataset)) - 1}). Check config and dataset."); return
    print(
        f"Processing {len(to_process)} problems from {source_data_file} (Original indices {start} to {start + len(to_process) - 1})...")

    summaries = []
    total_time_start = time.time()
    for i, prob_data in enumerate(to_process):
        prob_id, statement, gt_ans_str = str(prob_data.get('id')), prob_data.get('problem'), str(
            prob_data.get('answer', ''))
        print(f"\n--- Processing Problem {i + 1}/{len(to_process)} (ID: {prob_id}) ---")
        prob_time_start = time.time()
        traj_dict = {}
        try:
            traj_obj = agent.run(prob_id, statement, config['enable_reflection'])
            traj_dict = traj_obj.to_dict()
            agent_ans_raw = traj_obj.get_final_answer_for_evaluation()
            agent_ans_ext = extract_numerical_answer_from_string(agent_ans_raw)
            gt_ans_ext = extract_numerical_answer_from_string(gt_ans_str)
            correct = False
            if gt_ans_ext is not None and agent_ans_ext is not None:
                try:
                    correct = abs(float(gt_ans_ext) - float(agent_ans_ext)) < 1e-5
                except (ValueError, TypeError):
                    correct = agent_ans_ext.strip() == gt_ans_ext.strip()

            # Ensure status reflects correctness if not an error status
            current_status = traj_dict.get("status", "unknown")
            if not current_status.startswith("error"):
                traj_dict["status"] = 'success' if correct else 'failure'
            traj_dict["is_correct_assessment"] = correct
            print(
                f"  GT (Ext): '{gt_ans_ext}' (Data: '{gt_ans_str}') | Agent (Ext): '{agent_ans_ext}' (Raw: '{agent_ans_raw}') | Correct: {correct} | Status: {traj_dict['status']}")
        except Exception as e:
            print(f"FATAL agent.run Error (Prob {prob_id}): {e}");
            traceback.print_exc()
            traj_dict = {"problem_id": prob_id, "problem_statement": statement,
                         "status": f"error_runtime:{type(e).__name__}", "is_correct_assessment": False,
                         "metadata": {"error": str(e), "timestamp": time.time()}}

        # Ensure these keys exist even if extraction failed or error occurred
        traj_dict.setdefault("ground_truth_from_data", gt_ans_str)
        traj_dict.setdefault("ground_truth_extracted_eval", gt_ans_ext if 'gt_ans_ext' in locals() else None)
        traj_dict.setdefault("agent_answer_raw_eval", agent_ans_raw if 'agent_ans_raw' in locals() else None)
        traj_dict.setdefault("agent_answer_extracted_eval", agent_ans_ext if 'agent_ans_ext' in locals() else None)

        summaries.append(traj_dict)
        fname = f"trajectory_{prob_id.replace('/', '_').replace(':', '_').replace(' ', '_')}.json"  # Sanitize filename
        try:
            with open(os.path.join(exp_out_dir, fname), 'w', encoding='utf-8') as f:
                json.dump(traj_dict, f, indent=2, ensure_ascii=False)
        except Exception as e_save:
            print(f"  Error saving traj {prob_id}: {e_save}")
        print(f"  Time: {time.time() - prob_time_start:.2f}s")

    total_time = time.time() - total_time_start
    print(f"\n--- Data Collection Run Summary ---")
    print(f"Processed: {len(summaries)}, Time: {total_time:.2f}s")
    correct_n = sum(1 for r in summaries if r.get('is_correct_assessment'))
    acc = (correct_n / len(summaries) * 100) if summaries else 0.0
    print(f"Accuracy on this run: {correct_n}/{len(summaries)} = {acc:.2f}%")  # Accuracy on the processed segment
    summary_fpath = os.path.join(exp_out_dir, "_full_trajectories_summary.jsonl")
    try:
        with open(summary_fpath, 'w', encoding='utf-8') as f:
            for td in summaries: f.write(json.dumps(td, ensure_ascii=False) + "\n")
        print(f"Summary JSONL: {summary_fpath}")
    except Exception as e_sum:
        print(f"Error saving summary: {e_sum}")
    print("Data collection run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCD Math Inference / Data Collection")
    parser.add_argument("--config", default="config/default_config.yaml", help="Config YAML path.")
    args = parser.parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded config: {args.config}")
    except Exception as e:
        print(f"FATAL config error {args.config}: {e}"); sys.exit(1)

    # Pre-run config checks
    if cfg.get('llm_provider') == 'openai' and not os.getenv('OPENAI_API_KEY'): print(
        "FATAL: OpenAI provider but OPENAI_API_KEY not set."); sys.exit(1)
    if cfg.get('llm_provider') == 'local' and not cfg.get('local_model_id'): print(
        "FATAL: Local provider but local_model_id not set."); sys.exit(1)
    if not cfg.get('dataset_path') or not cfg.get('inference_source_file'): print(
        "FATAL: 'dataset_path' or 'inference_source_file' missing in config."); sys.exit(1)

    run_inference(cfg)
