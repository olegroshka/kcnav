# KCNav: Complexity-Guided Math Reasoning with ReAct Agents
Cmplexity-Guided ReAct Agents – steer an LLM’s chain-of-thought with Normalised Compression Distance and a learned Transformer to solve maths problems more accurately and efficiently.


![tests](https://img.shields.io/badge/build-passing-brightgreen)
![license](https://img.shields.io/github/license/your-org/kcnav)

A research code-base that teaches a *local* large-language model (e.g. Gemma-9B, Phi-3) to reason about maths problems while **measuring and controlling the complexity of each reasoning step**.  
The agent follows the **ReAct** loop (Reason → Act → Observe → … → Answer).  
A lightweight **Transformer** is trained on successful trajectories to predict whether a *candidate next step* keeps complexity in a “productive” range, enabling:

* **NCD-based candidate re-ranking** – avoid meandering thought chains.  
* **Self-reflection** – flag and critique suspicious complexity spikes.  

> *Diffusion complexity modelling is in the roadmap but disabled by default.*

---

## ✨ Key features
| Module | File | What it does |
|--------|------|--------------|
| ReAct Agent | `kcnav/react_agent.py` | Runs the reasoning loop, logs every thought, action and observation. |
| NCD Calculator | `kcnav/complexity_calculator.py` | Computes local & global Normalised Compression Distance. |
| Complexity Transformer | `kcnav/complexity_model_transformer.py` | Learns “good” complexity dynamics and scores candidate steps. |
| Dataset loaders | `kcnav/dataset_preparation.py` | GSM8K, MATH, SVAMP helpers. |
| Training scripts | `scripts/train_complexity.sh`, `scripts/train_agent.sh` | One-command training driven by YAML configs. |
| Inference pipeline | `kcnav/inference_pipeline.py` | Runs the agent, saves trajectories, computes metrics. |

---

## 🔧 Installation

```bash
git clone https://github.com/your-org/kcnav.git
cd kcnav

# Python ≥ 3.10 and PyTorch ≥ 2.1 are assumed
pip install -r requirements.txt

# (Optional) install compress-algos with C extensions for faster NCD
pip install python-lzma
