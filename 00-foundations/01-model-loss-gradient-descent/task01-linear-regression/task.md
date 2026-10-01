# S1-T1 — Linear Regression (By Hand)

## Why this is needed
This is the smallest possible instance of the model/loss/gradient-descent loop that every later stage reuses — including TRM's real training loop (Module D). Get this loop solid before anything else builds on it.

## Concepts to understand
- Vectors/matrices
- Derivatives and the chain rule
- What a loss function measures
- What a gradient update actually does to a parameter

## What to read/watch/use
- 3Blue1Brown — Essence of Calculus (derivatives), if the chain rule isn't solid yet
- Otherwise no external resource needed — this should be derivable from first principles

## Mode
**A (Build)** — AI Tier capped at 1-2 (Part 8).
- Asking AI to explain a numpy error is fine.
- Asking AI to derive the gradient or write the update loop is **not** — that's the entire exercise.

## Implementation / Experiment
1. Generate a small toy dataset: `y = 3x + 2 + noise`
2. By hand, derive the gradient of mean-squared-error loss with respect to the weight and bias
3. Implement gradient descent in plain numpy — **no sklearn, no autograd**
4. Confirm the learned weight/bias converge close to the true values (w≈3, b≈2)

## What they must be able to explain
- What MSE loss measures and why
- The derivation of dL/dw and dL/db, step by step
- What the learning rate does and what happens if it's too large/small
- Why gradient descent moves in the negative gradient direction

## TRM Connection
Module D (loss aggregation, AdamW) — this is the same loop TRM's training runs on, just without the recursion wrapped around it yet.

## Expected Output
- `implementation/linear_regression.py` (hand-coded, no autograd)
- A plot or printed log showing loss decreasing
- `README.md` explaining the derivation in their own words

## Pass Criteria
- Code converges to approximately the true weight/bias
- They can reproduce the gradient derivation on a whiteboard, cold, without notes

## Cross-question Examples
- **L1:** "What does MSE loss measure?"
- **L2:** "Walk me through one gradient descent step, by hand."
- **L5:** "Why do we subtract the gradient instead of adding it?"
- **L6:** "What happens if the learning rate is 100x too large?"
- **L10:** "Your loss is oscillating instead of decreasing — what would you check?"

## Dependencies
None — this is the first real task.