# KCNav – Complexity-Guided Math Reasoning for Local LLMs
> *Kolmogorov-Complexity Navigator*  
> Teach your ReAct agent to recognise when its chain-of-thought is getting **too messy** and self-correct.

[![tests](https://img.shields.io/badge/build-passing-brightgreen)]()  
[![license](https://img.shields.io/github/license/your-org/kcnav)]()

KCNav adds a tiny “metacognition” layer on top of any *local* instruction-tuned LLM (Gemma-2-9B, Phi-3, Llama-3 8B…).  
It measures the **Normalized Compression Distance (NCD)** of every reasoning step, learns what healthy trajectories look like, and discards candidate thoughts that wander into risky complexity.

|  | Baseline&nbsp;Gemma-2-9B | **KCNav&nbsp;(Diffusion-guided)** |
|---|:---:|:---:|
| **Accuracy**<br>(GSM8K × 300) | 80.0 % | **81.7 %** |
| **Steps / successful** | 6.5 | **5.7** (-12 %) |
| **Variance of max NCD** | 1.00 × | **0.77 ×** |
| **Wall-clock latency** | 1 × | 4.2 × |

> *Accuracy gain is not yet statistically significant (p ≈ 0.60, n = 300) but the agent becomes markedly more predictable.*

<p align="center">
  <img src="docs/figs/latency_vs_ncd.png" width="460" alt="Latency vs NCD">
  <br><sup><b>KCNav prunes high-risk branches.</b><br>
  Fewer runaway traces → tighter complexity envelope.</sup>
</p>

---

## ✨ Why KCNav?

* **NCD as a universal signal** – no maths-domain features, just compression sizes.  
* **Learned complexity corridor** – VAE-Diffusion (or tiny Transformer) scores how typical a partial trace is for *successful* proofs.  
* **Step gating** – the agent retries until its next thought fits the safe corridor.

---


## 🔧 Installation

```bash
git clone https://github.com/your-org/kcnav.git
cd kcnav

# Requires Python ≥3.10 and PyTorch ≥2.1
pip install -r requirements.txt
# (Optional) faster compressors
pip install python-lzma
```

Set your GPU / model once:

```bash
export KCNAV_DEVICE="cuda:0"              # or 'cpu'
export KCNAV_MODEL="google/gemma-2-9b-it"
```

---

## 🚀 Quick Start – single-GPU demo

1. **Collect baseline traces**

    ```bash
    python main.py \
      --mode   inference \
      --config config/base_gemma2_9b.yaml
    # → data/collected_trajectories/Base_Gemma2_9B/
    ```

2. **Train the complexity scorer (VAE-Diffusion)**

    ```bash
    python main.py \
      --mode   train_complexity \
      --config config/train_diffusion.yaml
    # → checkpoints/complexity_diffusion.pt
    ```

3. **Run guided inference**

    ```bash
    python main.py \
      --mode   inference \
      --config config/guided_diffusion_gemma2_9b.yaml
    # → data/collected_trajectories/Diffusion_Guided_Gemma2_9B/
    ```

---

## 📊 Rebuild the paper figures

```bash
python plot/plot_complexity_metrics.py \
  --exp_dirs  data/collected_trajectories/Base_Gemma2_9B/ \
              data/collected_trajectories/Diffusion_Guided_Gemma2_9B/ \
  --out_dir   figures3/
```

All plots for the paper/table go to `figures3/`.

---

## 🛠️ Roadmap

* **Distil the diffusion scorer** into a 2-layer MLP to cut latency ×4  
* **Add “too-simple” guard-band** to avoid over-pruning trivial steps  
* **Evaluate on MATH / SVAMP** and code-generation tasks  
* **Self-reflection phase** – LLM critiques entire trace using NCD profile  

---

## 📄 Citation

```bibtex
@misc{roshka2025kcnav,
  author  = {Oleg Roshka},
  title   = {{KCNav}: Complexity-Guided Math Reasoning with ReAct Agents},
  year    = {2025},
  url     = {https://github.com/your-org/kcnav}
}
```

---

## 🤝 Acknowledgements

Inspired by **ReAct** (Yao *et al.*) and the **Normalized Compression Distance** (Li *et al.*).  
Thanks to the open-weights crews behind Gemma and Llama!

---

### License

KCNav is released under the MIT License – see `LICENSE`.
