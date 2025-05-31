# main.py
# Main entry point to dispatch to training or inference scripts.

import argparse
import yaml
import os
import sys
import subprocess

# Determine project root directory for consistent imports/paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_script(script_path: str, config_path: str, extra_args: list = None):
    """Helper function to run a Python script using subprocess."""
    python_executable = sys.executable  # Use the same python interpreter
    command = [python_executable, script_path, "--config", config_path]
    if extra_args:
        command.extend(extra_args)

    print(f"\nExecuting: {' '.join(command)}\n")
    # Set PYTHONPATH to include the project root for consistent imports within sub-scripts
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    try:
        process = subprocess.run(command, check=True, text=True, cwd=PROJECT_ROOT, env=env)
        print(f"\nScript {os.path.basename(script_path)} finished successfully.")
    except FileNotFoundError:
        print(f"ERROR: Script not found at {script_path}. Make sure you are in the project root.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Script {os.path.basename(script_path)} failed with return code {e.returncode}.")
        # print(f"Stderr:\n{e.stderr}") # Uncomment to see error output from script
        # print(f"Stdout:\n{e.stdout}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while running the script: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="NCD Math Project - Main Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train_complexity", "train_agent", "inference"],
        help="Which pipeline step to run."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to the main configuration YAML file."
    )
    args = parser.parse_args()

    config_path_abs = os.path.join(PROJECT_ROOT, args.config)

    if not os.path.exists(config_path_abs):
        print(f"ERROR: Configuration file not found at {config_path_abs}")
        sys.exit(1)

    extra_script_args = []  # Currently no CLI overrides passed to scripts via main.py

    if args.mode == "train_complexity":
        script_rel_path = "training/train_complexity_model.py"
    elif args.mode == "train_agent":
        script_rel_path = "training/train_agent.py"
    elif args.mode == "inference":
        script_rel_path = "inference/inference_pipeline.py"
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

    script_abs_path = os.path.join(PROJECT_ROOT, script_rel_path)
    run_script(script_abs_path, config_path_abs, extra_script_args)


if __name__ == "__main__":
    main()