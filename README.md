# LLMs from Scratch

Notebooks I wrote while trying to understand how modern LLMs actually work under the hood. The point isn't to build anything production-ready — it's to implement the core ideas myself, in plain PyTorch, until they stop feeling like magic.

Each notebook is one concept, top to bottom, with the math and the code side by side.

## What's here

- **MoE.ipynb** — Mixture of Experts. Top-k gating, sparse routing, load balancing loss. Built a tiny MoE layer and watched the routing actually specialize, which was satisfying.
- **QuantizationFromScratch.ipynb** — symmetric and asymmetric quantization, INT8 and INT4, calibration. The interesting part was seeing exactly where accuracy starts to break and why.
- **GRPO.ipynb** — Group Relative Policy Optimization, the RL algorithm DeepSeek used. No value model, just group-relative advantages. Implemented the loss from the paper and trained on a toy reasoning task.

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

**Knowledge distillation.** Specifically soft distillation — KL on the teacher's full logits with a temperature, not just hard labels. I want to take a competent ~27B teacher and see how small a student (qwen 3.5 2B) can get before the wheels come off. Curious what temperature actually does to the gradient signal in practice vs. what the original Hinton paper claims.

**Unsloth.** I've been doing everything in vanilla PyTorch / HF, which is great for understanding and terrible for actually training anything bigger than toys. Unsloth's Triton kernels are supposed to make fine-tuning a 7B–8B model on a single consumer GPU realistic. Plan: replicate one of my GRPO experiments at a serious scale, benchmark it honestly against the slow version.

**Agents.** The thing I'm most curious about and least sure how to approach. Starting simple — a tool-using loop with no framework, just a model, a parser, and a few tools. Then ReAct. Then a planner-executor split. I want to actually understand where agents fail before reaching for a library. Probably ends with a small coding agent of some kind.

## Running

Nothing fancy. Each notebook is self sufficent.
