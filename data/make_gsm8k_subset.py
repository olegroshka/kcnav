"""
Make a gsm8k_300.jsonl containing *complete* problem records
chosen uniformly at random from the official test split.
"""

import json, random, argparse, pathlib

def main(src, dst, n, seed):
    random.seed(seed)
    with open(src) as f:
        problems = [json.loads(l) for l in f]

    subset = random.sample(problems, n)
    with open(dst, "w") as f:
        for ex in subset:
            f.write(json.dumps(ex)+"\n")

    print(f"✓ wrote {n} examples → {dst}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src",  default="datasets/gsm8k/test.jsonl")
    p.add_argument("--dst",  default="datasets/gsm8k/gsm8k_300.jsonl")
    p.add_argument("--n",    type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    main(**vars(p.parse_args()))
