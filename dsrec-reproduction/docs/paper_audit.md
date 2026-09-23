# Paper audit: DSRec (Dual-Interest Sequential Product Recommendation with Multi-Granular SSM)

Source: Liao & Mok, arXiv:2609.21548v1, uploaded PDF.

Each item below gives the paper's statement, where it comes from, how it will be implemented, and a confidence level. Classification tags follow the project convention: PAPER-SPECIFIED, IMPLEMENTATION-DERIVED, ASSUMPTION, VERIFIED.

## 1. Problem definition
Predict the next item v_{T+1} a user will interact with, given a historical sequence S_u = [v_1, ..., v_T]. Sec. III-A.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 2. Input representation
Items are one-hot vectors projected through a shared learnable embedding table E ∈ R^{|V|×D}, D = 64. Sec. III-D.1.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 3. User sequence definition
S_u is the chronologically ordered list of items a user interacted with; the subscript u is dropped for convenience elsewhere in the paper. Sec. III-A.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 4. Timestamp representation
Raw timestamps T = {t_1,...,t_T} are converted to inter-click gaps D = {d_1,...,d_T}, with d_1 = 0 and d_i = t_i - t_{i-1} for i > 1 (Eq. 9). Sec. III-E.1.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 5. Training data construction
The paper defines the training set as D = {(S_u, v_{T+1})} (Sec. III-A) and states the leave-one-out split (item 8 below). It never states whether training uses one (prefix, next-item) pair per user or a sliding window generating one pair per position in the sequence, as SASRec/BERT4Rec-style setups typically do.
Not specified in the paper. [ASSUMPTION]: use the sliding-window scheme (every prefix up to T-2 predicts the next item), since this is the de facto standard for leave-one-out sequential rec and the RecBole framework the authors cite defaults to it. Confidence: LOW on the paper matching this; MEDIUM that it's the right implementation choice. This will be documented in implementation_decisions.md, and the config will expose both modes.

## 6. Validation construction
Second-to-last item in the sequence is the validation target; items before it form the input context. Sec. IV-A.4.
[PAPER-SPECIFIED] for the target. [ASSUMPTION] for whether the validation input context is S_u[:-2] only, or is allowed to reuse the training-time sliding windows — the paper doesn't distinguish. Confidence: MEDIUM.

## 7. Test construction
Last item in the sequence is the test target, with everything before it (including the validation item) as context. Sec. IV-A.4.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 8. Leave-one-out protocol
"For each user, the last interacted item in temporal order is considered as the test data and the previous one as validation data." Sec. IV-A.4.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 9. Negative sampling / candidate evaluation
The paper reports HR@10, NDCG@10, MRR@10 (Sec. IV-A.4) but never states the candidate set: full ranking over all items, or ranking against a fixed number of sampled negatives (a well-known source of irreproducibility across SASRec-family papers).
Not specified in the paper. [ASSUMPTION]: full ranking over the entire item catalog (all |V| items, excluding only items already in the user's training history), since this is RecBole's default full-sort evaluation and the authors cite RecBole as their evaluation harness. Confidence: LOW. This is flagged as the single biggest reproducibility risk in the paper and will get its own paragraph in evaluation_protocol.md.

## 10. Long-term branch
A Mamba block (Eqs. 6-8): 1D convolution + SiLU gate, a selective SSM, then a residual projection back to the input. Sec. III-D.2.
[PAPER-SPECIFIED] at the block level. [IMPLEMENTATION-DERIVED] for the exact selective-SSM parameterization (Δ, A, B, C discretization), since the paper only gives the generic continuous/discretized SSM equations (Eqs. 1-4) and defers to Gu & Dao's Mamba. Confidence: HIGH for structure, MEDIUM for exact numerics — plan is to use the official `mamba-ssm` package rather than reimplement the selective scan, and pin its version.

## 11. Short-term branch
A time-aware SSM with a sigmoid time gate that blends new input against the previous hidden state (Eqs. 10-12). Sec. III-E.
[PAPER-SPECIFIED]. Confidence: HIGH for the gating equation; MEDIUM for the internal SSM call in Eq. 12 (`SSM(x_i^s)`), since it isn't spelled out whether this reuses the same Conv1D+SiLU preprocessing from Eqs. 6-7 or is a bare SSM call. Implementation will mirror Eqs. 6-7's preprocessing before the gate, since Fig. 2's short-term stack shows the same Con1d/σ/SSM shape as the long-term stack — documented as IMPLEMENTATION-DERIVED.

## 12. Time encoding
Log-scaled gaps, quantile-bucketized into N buckets, mapped to a learnable embedding, concatenated with the item embedding and passed through MLP_S (Eq. 10). Sec. III-E.1.
[PAPER-SPECIFIED] for the pipeline. Not specified: N (bucket count) and the time-embedding dimension before concatenation. [ASSUMPTION]: N = 10 buckets (common default in time-aware rec literature) and a time-embedding dimension equal to D = 64 before the MLP_S projection back to D. Confidence: LOW.

## 13-14. Cross-branch interaction / residual cross-fusion
"Each encoder is augmented with a residual cross-connection, meaning each branch receives the detached output of the other as auxiliary input." Sec. III-F. No equation is given for the fusion operator itself.
[PAPER-SPECIFIED] that gradients are stopped across branches. [ASSUMPTION] for the fusion operator: Fig. 2 draws a "cross" arrow feeding into each branch's Add & Norm block after its FFN, so the implementation will add the other branch's detached FFN output as an extra residual term at that point (`H_l' = AddNorm(FFN(H_l) + H_s.detach())` and symmetrically for H_s). Confidence: MEDIUM, based on the figure rather than an equation. This is the second-biggest reproducibility risk after item 9 and will be documented prominently.

## 15. FFN
Two-layer GELU FFN per branch, hidden dim 4D, separate weights for long-term and short-term (Eqs. 13-14). Sec. III-G.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 16. Add & Norm
Dropout + layer norm applied after each Mamba/time-aware-SSM block and after each FFN (Sec. III-G, prose only). Whether norm is applied before or after the sublayer (pre-norm vs. post-norm) is not stated in equations.
[IMPLEMENTATION-DERIVED] from Fig. 2, which draws "Layer Norm" before the Mamba/SSM block and a separate "Add & Norm" after the FFN — i.e. pre-norm going into the sequence block, post-norm after the FFN. Confidence: MEDIUM (figure-derived, not equation-derived). Documented explicitly in architecture.md rather than assumed silently.

## 17. Prediction layer
Concatenate the final-position long-term and short-term hidden states, pass through an MLP, then take the dot product with the (shared) embedding table E and softmax (Eqs. 15-16).
[PAPER-SPECIFIED]. Confidence: HIGH. Note: this ties the output projection to the input embedding table, which the prediction head will implement directly rather than allocating a second |V|×D matrix.

## 18. Loss
Cross-entropy over the full softmax distribution across all items (Eq. 17).
[PAPER-SPECIFIED]. Confidence: HIGH. No mention of label smoothing, sampled softmax, or auxiliary losses — none will be added.

## 19-21. Optimizer, learning rate, batch size
Adam, lr = 0.001, training batch size 2048, validation batch size 4096. Sec. IV-A.3.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 22. Sequence length
Maximum sequence length 200 for MovieLens-1M (50 for the Amazon datasets, not our target here). Sec. IV-A.3.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 23. Embedding dimension
D = 64. Sec. IV-A.3.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 24. SSM/Mamba parameters
State dimension 32, convolution/kernel width 4, expansion factor 2. Sec. IV-A.3.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 25. Dropout
0.2 after each layer. Sec. IV-A.3.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 26. Number of blocks (B)
Table IV sweeps B ∈ {1..5} for Amazon-Beauty (best at B=3) and Amazon-Video-Games (best at B=2). No B is ever reported for the MovieLens-1M main results in Table II.
Not specified in the paper for MovieLens-1M. [ASSUMPTION]: default to B = 2, matching the better of the two reported optima and Fig. 2's generic "×B" notation. Confidence: LOW. The config will expose B as a swept parameter (per the depth-experiment phase) rather than hard-coding this guess as fact.

## 27. Evaluation metrics
HR@10 = 1 if the held-out target is in the top-10 ranked list, else 0. NDCG@10 = 1/log2(rank+1) if the target is in the top 10, else 0. MRR@10 = 1/rank if the target is in the top 10, else 0. Sec. IV-A.4.
[PAPER-SPECIFIED]. Confidence: HIGH.

## 28. Dataset statistics
MovieLens-1M: 6,041 users, 3,417 items, 999,611 interactions, avg. 165.5 actions/user, avg. 292.6 actions/item, 95.157% sparsity. Table I.
[PAPER-SPECIFIED]. Confidence: HIGH. Used as a check: preprocessing must reproduce these counts (within the tolerance of any filtering rule, which the paper doesn't state — see implementation_decisions.md for the k-core filtering assumption).

## 29. Ablations
Table III lists five variants: full DSRec, w/o Residual Cross-Fusion, w/o Dual-interest embedding (single), w/o SSM_S, w/o Fusion Encoder, and a Dual-Mamba baseline.
[PAPER-SPECIFIED] that these five exist and their reported numbers. [ASSUMPTION]/flag: "w/o Fusion Encoder" is described in the ablation prose (Sec. IV-C.d) as retaining the dual-interest structure but lacking "effective integration before output" — this reads as a different module than the residual cross-fusion of Sec. III-F, but the methodology section never separately defines a "Fusion Encoder." This is an internal inconsistency in the paper. Implementation will treat "w/o Fusion Encoder" as removing the final concatenation+MLP prediction step (Eq. 15) in favor of, e.g., summing the two branch outputs — documented as a best-effort reconstruction, not a verified match. Confidence: LOW for this one ablation; HIGH for the other four.

## 30. Reported results
Table II, MovieLens-1M, DSRec: HR@10 = 0.3217, NDCG@10 = 0.1869, MRR@10 = 0.1454. Best baseline (SIGMA) HR@10 = 0.3136.
[PAPER-SPECIFIED]. Confidence: HIGH. These are the numbers reproduction_report.md will compare against.

## Other numerical details worth flagging now

- Eq. 5's user-history aggregation, x_i^l = x_i + (1/(i-1)) * sum_{j<i} x_j, divides by zero at i = 1. Not addressed in the paper. [ASSUMPTION]: define x_1^l = x_1 (no aggregation term for the first position).
- Sec. III-D.1's prose describes the long-term embedding as X_l = MLP(X‖E_u), which doesn't match Eq. 5's actual formula (no MLP, no explicit E_u term). These are two different descriptions of the same thing in the same section. [ASSUMPTION]: implement Eq. 5 as the operative definition, since it's the only fully specified one, and note the mismatch rather than silently picking one.
- Duplicate items within a user's sequence: retained or deduplicated is never stated. [ASSUMPTION]: retain duplicates (no dedup), since MovieLens-1M's average of 165.5 actions/user against 3,417 items implies repeat viewing is not filtered out elsewhere in the pipeline.
- Padding convention (left- vs right-padding) for sequences shorter than 200: not stated. [ASSUMPTION]: right-pad with PAD_ID = 0, and use a padding mask everywhere; item IDs remapped so no real item collides with 0.
