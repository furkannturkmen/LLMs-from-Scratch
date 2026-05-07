# LLMs from Scratch

Notebooks I wrote while trying to understand how modern LLMs actually work under the hood. The point isn't to build anything production-ready — it's to implement the core ideas myself, in plain PyTorch, until they stop feeling like magic.

Each notebook is one concept, top to bottom, with the math and the code side by side.

## What's here

- **GRPO.ipynb** — Group Relative Policy Optimization, the RL algorithm from DeepSeek-Math (also used to train DeepSeek-R1). From-scratch implementation of the full algorithm: group sampling with `num_return_sequences`, group-relative advantages, dual-model setup (policy / behavior / reference), token-level KL penalty against the reference, PPO-style clipped surrogate objective with sign-aware min/max, and periodic reference-model refresh. Trained Qwen3-0.6B on Math-500 prompts with a verifiable reward. Faithful to the original paper, including details that most modern implementations (e.g. TRL's default config) drop — full β-weighted KL term and the iterative reference refresh.

- **MoE.ipynb** — Mixture of Experts. Top-k gating, noisy top-k routing, and a sparse dispatch layer plugged into a small GPT trained on Tiny Shakespeare. The attention block and training scaffold are adapted from Karpathy's nanoGPT and AviSoori1x's makeMoE; the router, expert, and sparse MoE layer are implemented from scratch. No load-balancing loss yet, and expert specialization analysis is on the to-do list — currently more of a "make sure I can build the parts" exercise than a finished study.

- **QuantizationFromScratch.ipynb** — symmetric and asymmetric INT8 quantization implemented in pure PyTorch. The interesting part was finding exactly where accuracy starts to break and why. Next: extending to 4-bit (NF4 / GPTQ-style).

## How I got here

I started the usual way — building a small GPT from scratch, attention, tokenizers, the whole thing. After that I kept asking "ok but how do they actually scale this / ship this / make it reason?" and each question turned into a notebook.

Roughly:

1. transformers from scratch (not in this repo, lives in my notes)
2. how do you grow params without growing compute → MoE
3. how do you ship a big model → quantization
4. how do reasoning models learn to reason → GRPO

Plenty of things I implemented are wrong or suboptimal. I left them as-is when the bug was instructive.

## Up next

A short list, in roughly the order I want to tackle them.

**Knowledge distillation.** Specifically soft distillation — KL on the teacher's full logits with a temperature, not just hard labels. I want to take a competent ~27B teacher and see how small a student (Qwen3 1.7B) can get before the wheels come off. Curious what temperature actually does to the gradient signal in practice vs. what the original Hinton paper claims. Plan to use vLLM for teacher inference — collecting logits with HF Transformers' `.generate()` is the bottleneck, and vLLM with paged attention should turn this from "let it run overnight" into "run it during a coffee."

**Unsloth.** I've been doing everything in vanilla PyTorch / HF, which is great for understanding and terrible for actually training anything bigger than toys. Unsloth's Triton kernels are supposed to make fine-tuning a 7B–8B model on a single consumer GPU realistic. Plan: replicate one of my GRPO experiments at a serious scale, benchmark it honestly against the slow version.

**Agents.** The thing I'm most curious about and least sure how to approach. Starting simple — a tool-using loop with no framework, just a model, a parser, and a few tools. Then ReAct. Then a planner-executor split. I want to actually understand where agents fail before reaching for a library. Probably ends with a small coding agent of some kind.

## What I read along the way

- Sebastian Raschka — *Build a Large Language Model from Scratch*
- Andrej Karpathy — nanoGPT, "Neural Networks: Zero to Hero" series
- Shao et al. — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning* (GRPO)
- Shazeer et al. — *Outrageously Large Neural Networks* (MoE)
- Hinton, Vinyals, Dean — *Distilling the Knowledge in a Neural Network*
- AviSoori1x — *makeMoE* (referenced for the MoE notebook scaffolding)

## Running

Python 3.10+, PyTorch 2.x, a single GPU is enough for everything currently in the repo (GRPO uses bf16 autocast on a T4/L4-class card; MoE and Quantization are small enough to run on CPU). No `requirements.txt` yet — `pip install` whatever a notebook complains about. Most notebooks were developed in Colab.
