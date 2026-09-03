"""model.py - Router, experts, recursive block, halting, MoRE model.

Extracted from the former monolithic train.py (plan.md 9: ONE training
system, split into readable modules; not parallel implementations).
"""

import os
import sys
import json
import time
import math
import collections
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from .families import NUM_OP_TYPES, NUM_EXPERTS_CANONICAL

# ---------------------------------------------------------------------------
# Routing modes (plan.md 4.1, T2.1 / T2.4)
# ---------------------------------------------------------------------------
#
# "top1_sparse" is THE canonical MoE/MoRE path:
#     router logits -> softmax -> top-1 -> selected expert ONLY
#                   -> selected gate probability multiplies its output
#
# "dense_blend" evaluates every expert on every token and blends by router
# probability. It is NOT MoE -- no token is routed anywhere, every expert sees
# every token, and expert independence is destroyed. updated_rules.md 1.1
# permits it solely as the labelled "MoRE - Dense Routing Ablation", so
# more/config.py refuses it unless the run label says dense_routing_ablation.
CANONICAL_ROUTING_MODE = "top1_sparse"
DENSE_ABLATION_ROUTING_MODE = "dense_blend"
ROUTING_MODES = (CANONICAL_ROUTING_MODE, DENSE_ABLATION_ROUTING_MODE)

# ---------------------------------------------------------------------------
# Router exploration noise (T5.4, plan.md 7.4 / updated_rules.md 1.1 ablation D)
# ---------------------------------------------------------------------------
#
# CANONICAL IS "none". Noise was previously always on, at a TRAINABLE scale
# initialised to 0.1, with an L2 penalty on that scale added to the loss. Three
# reasons that cannot be the canonical path:
#
#   1. It is an unreported architectural difference. Nothing in the config, the
#      resolved config or W&B said noise existed, yet every routing decision
#      during training was made on perturbed logits. A reader comparing MoE and
#      MoRE would not know it was there.
#   2. The scale is TRAINABLE, so the model can grow it. Growing it lowers the
#      routing-supervision loss in a way that has nothing to do with learning
#      the operation-to-expert map -- noise flattens the softmax, which changes
#      the balance objective's entropy term directly.
#   3. The L2-on-noise-scale term added to the objective is not a proven fix for
#      that (CLAUDE.md 2 says so explicitly). It is a third pressure on the
#      router whose weight was never measured against anything.
#
# The two noisy variants are retained, unchanged in behaviour, as ABLATION D:
#   "fixed_annealed" -- a non-learnable scale decaying linearly to zero over
#                       `router_noise_anneal_steps` training steps. Exploration
#                       early, deterministic routing later; nothing to learn and
#                       nothing added to the loss.
#   "trainable"      -- the historical behaviour, including the L2 penalty,
#                       preserved so old runs can be reproduced exactly.
#
# In "none" and "fixed_annealed" no noise PARAMETER exists, so it cannot appear
# in the canonical state_dict (the Gate 4 check).
CANONICAL_ROUTER_NOISE = "none"
ROUTER_NOISE_MODES = ("none", "fixed_annealed", "trainable")
ROUTER_NOISE_INIT_SCALE_DEFAULT = 0.1
ROUTER_NOISE_ANNEAL_STEPS_DEFAULT = 1000

# ---------------------------------------------------------------------------
# Capacity policy (plan.md 4.3, T2.3)
# ---------------------------------------------------------------------------
#
# Chosen policy: NO CAPACITY LIMIT. Every routed token is evaluated by its
# selected expert; no token is dropped and none receives a residual fallback.
#
# Justification, per plan.md 4.3 ("if the controlled dataset distribution does
# not require capacity limiting, prefer a simple implementation that is easy to
# verify"): this is a single-machine, fixed-batch, synthetic-arithmetic setup.
# Capacity factors exist to bound per-device buffer sizes in distributed
# expert-parallel training, which this is not. Dropping tokens would also
# silently corrupt the depth-allocation measurement, since a dropped token has
# no honest exit depth.
#
# The consequence for reporting: `overflow` is always a genuine measured 0, not
# a placeholder standing in for "not implemented". The real capacity pressure
# is reported as `max_load_fraction` -- the largest share of tokens any single
# expert received -- so a collapsing router is still visible in the logs.
CAPACITY_POLICY = "no_capacity_limit"


# ---------------------------------------------------------------------------
# 1a. Task names  (T-L5.0)
# ---------------------------------------------------------------------------
#
# Defined HERE, in the lowest layer, rather than in `config.py`. `MoREModel` has to
# branch on the task to decide which heads exist at all (§7.1), and `config.py`
# imports `model.py` and never the reverse (see run_context.py:52), so the choice
# was between moving the constant down or duplicating a string literal across two
# modules. `config.py` re-exports both names unchanged, so no call site moved.
TASK_ARITHMETIC = "arithmetic"
TASK_LANGUAGE   = "language"


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

class MoEBlock(nn.Module):
    """
    Single Mixture-of-Experts block with Top-1 routing.

    Contains `num_experts` independent 2-layer feedforward networks
    (one per operation family).  A learned linear router assigns each
    token to exactly one expert.  An entropy-based auxiliary loss
    encourages uniform expert utilisation.

    Args:
        num_experts : number of expert networks (default 7)
        d_model     : hidden / embedding dimension
        dropout     : dropout probability applied inside each expert
    """

    # T5.1 (plan.md 7.1): the default expert count comes from the one family
    # manifest (6), not from a literal 7. A constructor default is not
    # cosmetic -- every caller that does not pass num_experts explicitly, tests
    # included, silently got a 7th expert with no family behind it.
    def __init__(self, num_experts: int = NUM_EXPERTS_CANONICAL,
                 d_model: int = 256, dropout: float = 0.1,
                 routing_mode: str = "top1_sparse",
                 router_noise: str = CANONICAL_ROUTER_NOISE,
                 router_noise_init: float = ROUTER_NOISE_INIT_SCALE_DEFAULT,
                 router_noise_anneal_steps: int = ROUTER_NOISE_ANNEAL_STEPS_DEFAULT,
                 ffn_mult: int = 4):
        super().__init__()
        if routing_mode not in ROUTING_MODES:
            raise ValueError(
                f"routing_mode must be one of {ROUTING_MODES}, got {routing_mode!r}"
            )
        self.num_experts  = num_experts
        self.d_model      = d_model
        self.routing_mode = routing_mode
        # T6.7: the FFN width multiplier is a reported parameter, not a literal
        # buried in a constructor. updated_rules.md 9 requires it in provenance,
        # and Phase 9's parameter-matching work needs somewhere to change it that
        # is not a source edit. Default 4 = the width every run so far used.
        self.ffn_mult     = ffn_mult

        # Independent FFN experts -- one per family, no cross-expert paths.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ffn_mult, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        # Top-1 router: projects hidden state → num_experts logits
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # T5.4: exploration noise is OFF unless a run explicitly asks for
        # ablation D. `router_noise_scale` is created ONLY in the "trainable"
        # variant, so the canonical state_dict contains no noise parameter at
        # all -- absence is verifiable, whereas a parameter pinned to 0.0 would
        # still be there, still be in the optimizer, and still be reported.
        if router_noise not in ROUTER_NOISE_MODES:
            raise ValueError(
                f"router_noise must be one of {ROUTER_NOISE_MODES}, got "
                f"{router_noise!r}. Canonical is {CANONICAL_ROUTER_NOISE!r}; the "
                "other two are ablation D (updated_rules.md 1.1) and must be run "
                "under an explicitly labelled variant."
            )
        self.router_noise              = router_noise
        self.router_noise_init         = float(router_noise_init)
        self.router_noise_anneal_steps = int(router_noise_anneal_steps)
        # Plain int, not a buffer: it is ablation-only bookkeeping and must not
        # enter the state_dict, where it would make canonical and ablation
        # checkpoints structurally different for a reason unrelated to the model.
        self._router_noise_step        = 0
        if router_noise == "trainable":
            # Historical behaviour, preserved verbatim for reproducibility: the
            # optimiser adapts the scale, and engine.py adds an L2 penalty on it.
            # CLAUDE.md 2: that penalty is NOT a proven fix and must not be
            # described as one.
            self.router_noise_scale = nn.Parameter(torch.tensor(self.router_noise_init))

    def current_router_noise_scale(self) -> float:
        """The scale that WILL be applied on the next training forward pass.

        Reported rather than inferred: with three modes and an annealing
        schedule, "how much noise was on this epoch" is otherwise unanswerable
        from the logs. Returns 0.0 in canonical.
        """
        if self.router_noise == "none":
            return 0.0
        if self.router_noise == "trainable":
            return float(self.router_noise_scale.detach().abs().item())
        remaining = 1.0 - (self._router_noise_step /
                           max(1, self.router_noise_anneal_steps))
        return float(self.router_noise_init * max(0.0, remaining))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [N, d_model]  (flattened active tokens)
        Returns:
            out         : [N, d_model]  expert-processed tokens
            balance_loss: scalar — entropy balancing auxiliary loss for THIS ONE
                          call. The caller is responsible for averaging over
                          (block, depth) calls; do not sum it (plan.md 6.1).
            expert_idx  : [N] long tensor of chosen expert per token
            router_logits: [N, E] pre-softmax logits (after noise)
            route_stats : dict of measured dispatch diagnostics (T2.3), plus the
                          separately-reported `entropy_term` and
                          `switch_aux_term` (T4.2).
        """
        router_logits = self.router(x)                          # [N, E]
        if self.training and self.router_noise != "none":
            # ABLATION D ONLY (T5.4). Canonical routing is deterministic given
            # the weights: the argmax below is taken on the raw logits, so a
            # canonical run's routing decisions are reproducible from the
            # checkpoint. Under either noisy variant they are not, which is why
            # the variant has to be named in the run label and provenance.
            if self.router_noise == "trainable":
                scale = self.router_noise_scale.abs()           # gradient flows
            else:                                               # fixed_annealed
                self._router_noise_step += 1
                scale = self.current_router_noise_scale()
            noise         = torch.randn_like(router_logits) * scale
            router_logits = router_logits + noise
        router_probs = F.softmax(router_logits, dim=-1)         # [N, E]
        expert_idx   = torch.argmax(router_probs, dim=-1)       # [N]

        # ---- Dispatch -------------------------------------------------------
        # `expert_idx` is THE dispatch decision and is also the tensor returned
        # for the routing metrics, so the scored index always controls the
        # forward computation (T2.2 / plan.md 4.2: "do not score an argmax that
        # does not control dispatch").
        N = x.shape[0]
        if self.routing_mode == "top1_sparse":
            # Canonical path (plan.md 4.1):
            #   logits -> softmax -> top-1 -> SELECTED EXPERT ONLY
            #          -> selected gate probability multiplies its output
            #
            # Each token is evaluated by exactly one expert. Non-selected
            # experts never see the token, so they receive no gradient from it —
            # which is what makes the experts independently specialisable.
            #
            # The gradient path into the router is the gate multiply: d out/d
            # gate is the expert output, and gate = softmax(logits)[argmax], so
            # the router is trained by the task loss and not only by the
            # auxiliary/oracle terms.
            gate = router_probs.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)  # [N]
            out  = torch.zeros_like(x)
            for e in range(self.num_experts):
                sel = (expert_idx == e).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue                      # expert not called this batch
                out.index_add_(0, sel, self.experts[e](x[sel]))
            out = out * gate.unsqueeze(-1)
        else:
            # "MoRE - Dense Routing Ablation" ONLY (updated_rules.md 1.1,
            # ablation F). Evaluates every expert on every token and blends by
            # router probability. This is NOT MoE: no token is ever routed
            # anywhere, every expert sees every token, and expert independence
            # is destroyed. Reachable solely through
            # model.routing_mode = "dense_blend", which more/config.py refuses
            # unless the run is explicitly labelled dense_routing_ablation.
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
            out = (router_probs.unsqueeze(-1) * expert_outputs).sum(dim=1)

        # --- Two-term balance loss -------------------------------------------
        # Term 1 – Soft entropy: gradient always present via softmax, but blind
        #   to hard routing collapse because softmax is never exactly one-hot.
        avg_probs    = router_probs.mean(dim=0)                          # [E]
        entropy_term = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        # Term 2 – Switch-Transformer auxiliary loss:
        #   f_e  = fraction of tokens dispatched to expert e (hard argmax,
        #          detached so no gradient through the discrete choice).
        #   P_e  = mean soft router probability for expert e (differentiable).
        #   Loss = E * Σ_e(f_e * P_e)  — minimised at uniform routing (value=1),
        #          maximised at full collapse (value=E).  Must be ADDED to total
        #          loss (minimised), not subtracted.  We subtract balance_loss in
        #          total_loss, so we store: balance_loss = entropy_term - switch_aux
        #          → -entropy_term (maximise entropy) + switch_aux (minimise collapse).
        load = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            load[e] = (expert_idx == e).float().sum()
        load       = load / load.sum().clamp(min=1.0)                    # [E]
        switch_aux = self.num_experts * (load.detach() * avg_probs).sum()

        balance_loss = -entropy_term + switch_aux

        # --- Dispatch diagnostics (T2.3) -------------------------------------
        # Capacity policy is NO CAPACITY LIMIT (see CAPACITY_POLICY below), so
        # every token is dispatched and `overflow` is a genuine measurement of
        # zero, not a placeholder. `max_load_fraction` is the real capacity
        # pressure: the largest share of tokens any single expert received.
        # `expert_evaluations` is the honest compute count: how many
        # (token, expert) forward passes actually happened. Sparse does N;
        # dense does N * E. Without this field both paths report the same
        # `dispatched` count and the dense ablation silently understates its
        # own cost by a factor of E, which would make the sparse-vs-dense
        # comparison meaningless.
        route_stats = {
            "tokens":            float(N),
            "dispatched":        float(N),
            "expert_evaluations": float(N) if self.routing_mode == CANONICAL_ROUTING_MODE
                                  else float(N * self.num_experts),
            "overflow":          0.0,
            "max_load_fraction": float(load.max().item()) if N > 0 else 0.0,
            "experts_called":    float((load > 0).sum().item()),
            # T4.2 / CLAUDE.md 4: the two halves of the balance objective must be
            # REPORTED separately. They move in opposite directions (entropy up is
            # good, switch-aux up is bad), so the single `balance_loss` number can
            # sit flat while both components drift. Detached -- these are
            # diagnostics; the gradient path is `balance_loss` itself.
            "entropy_term":      float(entropy_term.detach().item()),
            "switch_aux_term":   float(switch_aux.detach().item()),
        }

        # Return router_logits (pre-softmax, after noise) so callers can
        # apply oracle CE directly on the router — much shorter gradient path
        # than going through step_cls_head after pooling.
        return out, balance_loss, expert_idx, router_logits, route_stats


class MoREWrapper(nn.Module):
    """
    Wraps a MoEBlock in an active-token-mask loop of up to `max_depth`
    iterations.  Each depth step:
      1. Processes only ACTIVE tokens through the MoEBlock (Top-1).
      2. Updates their hidden state with a residual connection + LayerNorm.
      3. Per-expert halt heads output p_t = sigmoid(Linear(d_model, 1)) per
         active token. Each expert has its own head:
           Expert 0 (ADD/SUB) → learns to fire at depth=1 (simple, 1-step ops).
           Expert 5 (SORT/STAT) → learns to recurse deeper (complex aggregation).
      4. CANONICAL ACT (Graves 2016 / Universal Transformer, plan.md 5.2 --
         "do not invent a novel halting equation"). Cumulative halt mass c_t;
         a token halts at the step where c_t + p_t would exceed
         `halt_threshold = 1 - eps`; its final step carries the remainder
         R = 1 - c_t instead of p_t. The output is the weighted sum of the
         per-step states, and the weights of the steps a token takes sum to
         EXACTLY 1 -- a convex combination.
      5. Any tokens still active when depth == max_depth are force-exited (their
         weight is also the remainder, so the sum-to-1 property holds there too).
      6. Pad tokens (step_mask == False) are excluded from depth=1 onward.

    Why the convex combination matters. The previous implementation wrote the
    state at exit time into an output buffer, so the halt probability affected
    only WHICH state was copied -- a discrete choice with no gradient. The halt
    heads then received gradient from nothing at all: measured before the fix,
    all 24 halt-head parameters had `grad is None` after a full backward pass.
    Multiplying each step's state by its halt weight is what puts the halt
    parameters on the path from the TASK loss (T3.2 / T3.4).

    Gradients flow through every depth step because we use boolean masks
    (no in-place scatter on leaf tensors with grad).

    Args:
        d_model     : hidden dimension
        max_depth   : maximum recursion depth (default 7)
        num_experts : number of expert networks (default: the canonical six)
        dropout     : dropout in experts
    """

    def __init__(
        self,
        d_model: int = 256,
        max_depth: int = 7,                          # recursion budget, NOT E
        num_experts: int = NUM_EXPERTS_CANONICAL,    # T5.1: from the manifest
        dropout: float = 0.1,
        fixed_depth: bool = False,
        routing_mode: str = CANONICAL_ROUTING_MODE,
        router_noise: str = CANONICAL_ROUTER_NOISE,
        router_noise_init: float = ROUTER_NOISE_INIT_SCALE_DEFAULT,
        router_noise_anneal_steps: int = ROUTER_NOISE_ANNEAL_STEPS_DEFAULT,
        ffn_mult: int = 4,
        attention: bool = False,
        n_heads: int = 4,
    ):
        super().__init__()
        self.max_depth   = max_depth
        self.num_experts = num_experts
        self.d_model     = d_model
        self.fixed_depth = fixed_depth

        # ACT halting threshold = 1 - eps (Graves 2016 / Universal Transformer).
        # A token halts once its cumulative halt mass would cross this, which
        # guarantees the per-token step weights sum to exactly 1. eps > 0 is what
        # makes a single-step exit reachable: with threshold exactly 1.0 a token
        # whose p_1 = 0.999 would still be forced to take a second step.
        self.halt_eps       = 0.01
        self.halt_threshold = 1.0 - self.halt_eps

        self.moe_block = MoEBlock(
            num_experts, d_model, dropout,
            routing_mode=routing_mode,
            router_noise=router_noise,
            router_noise_init=router_noise_init,
            router_noise_anneal_steps=router_noise_anneal_steps,
            ffn_mult=ffn_mult,
        )

        # Per-expert halting heads — replaces the single shared halting_router.
        # Each expert independently learns when its type of operation is "done".
        # Expert 0 (ADD/SUB) learns: halt fast   (1-step operations).
        # Expert 5 (SORT/STAT) learns: recurse    (needs multiple refinements).
        self.expert_halt_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(num_experts)
        ])

        # LayerNorm applied after residual at each depth step
        self.layer_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # T-L4.0 / T-L4.1  ONE causal self-attention sublayer, shared over depth
        # ------------------------------------------------------------------
        #
        # WHY IT IS REQUIRED (plan_language.md §6.1). Without attention no token
        # ever sees another, so the best achievable language model is the unigram
        # distribution: every arm of the MoE/MoR/MoRE matrix would converge to
        # `unigram_ce` and the comparison would measure nothing. For arithmetic it
        # was the right minimal design; for causal LM its absence is degenerate.
        #
        # WHY EXACTLY ONE MODULE, CONSTRUCTED HERE. `updated_rules.md` §2.1: the
        # same recursive block parameters at every recursion depth. A per-depth
        # attention module would make depth a stack of depth-specific networks
        # instead of computation through time, which is the invariant that makes
        # the MoR and MoRE arms mean what they claim. This mirrors the Universal
        # Transformer layout -- shared attention sublayer, then shared transition
        # sublayer -- with the transition replaced by the Top-1 MoE block.
        #
        # WHY ABSENCE RATHER THAN A BYPASSED MODULE (T-L4.1, §6.5). Same reasoning
        # as `router_noise_scale` above: with `attention=False` there is NO
        # attribute, so the arithmetic `state_dict` key set, the arithmetic
        # parameter counts and Gate 4's "no unexpected parameter" check are
        # provably untouched rather than believed to be. A module bypassed by a
        # forward-time flag would still be in the state_dict and still be in the
        # optimizer.
        self.attention = bool(attention)
        self.n_heads   = int(n_heads)
        if self.attention:
            if d_model % self.n_heads != 0:
                raise ValueError(
                    f"d_model {d_model} is not divisible by n_heads "
                    f"{self.n_heads}. nn.MultiheadAttention splits the model "
                    "dimension across heads, so this is a config error, not a "
                    "case to round."
                )
            self.attn = nn.MultiheadAttention(
                d_model, self.n_heads, dropout=dropout, batch_first=True)
            # Its own norm: the attention sublayer and the transition sublayer are
            # separate residual blocks in the Universal Transformer layout, and
            # sharing `layer_norm` between them would tie two different residual
            # streams to one set of statistics.
            self.attn_norm = nn.LayerNorm(d_model)
            # Causal masks, cached per (S, device). A PLAIN DICT, not a buffer:
            # a registered buffer would enter `state_dict()` and make a language
            # checkpoint structurally dependent on the sequence length it last ran
            # at. Same reasoning as `_router_noise_step`.
            self._causal_cache: dict = {}

    def causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Upper-triangular bool mask, `True` = DISALLOWED. T-L4.2.

        Orientation, stated because getting it backwards is silent: entry `[i, j]`
        is `True` when `j > i`, so query position `i` may attend to key positions
        `j <= i` and to nothing later. `nn.MultiheadAttention` treats a `True` entry
        in a bool `attn_mask` as "not allowed to attend".

        `is_causal=True` IS DELIBERATELY NOT USED. In PyTorch's API it is a *hint*:
        whether it is honoured depends on which backend kernel gets selected at
        runtime, so a shape, dtype or version change could silently make the model
        non-causal. Everywhere else in this model a silent no-op would cost accuracy;
        here it would mean TARGET LEAKAGE, and the run would look excellent with
        nothing else out of place. An explicit mask goes through the same code path
        on every backend, and Gate L1 perturbs future positions to check it.
        """
        key = (int(seq_len), str(device))
        m = self._causal_cache.get(key)
        if m is None:
            m = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1)
            self._causal_cache[key] = m
        return m

    def forward(
        self,
        x: torch.Tensor,
        step_mask: torch.Tensor,
        step_experts: torch.Tensor | None = None,
    ):
        """
        Args:
            x            : [B, S, d_model]
            step_mask    : [B, S] bool — True for real steps, False for padding.
                           Pad tokens are permanently excluded from the depth loop.
            step_experts : [B, S] long — oracle expert index per token (-1 for pad).
                           When provided, oracle CE is computed directly on router
                           logits at each depth step (short gradient path to router).
        Returns (11-tuple):
            final_output      : [B, S, d_model] — ACT convex combination of the
                                per-step states, NOT the state at exit time.
            total_bal_loss    : scalar — MEAN of the per-call balance loss over
                                the depth steps this wrapper actually ran, not
                                the sum (T4.1 / plan.md 6.1). Summed, it grew
                                with max_depth and confounded every depth
                                comparison.
            ponder_cost       : scalar, DIFFERENTIABLE. Normalized (N + R) /
                                (max_depth + 1) over real tokens. The gradient
                                reaches the halt heads through R, not N (N is a
                                discrete count). Replaced the old
                                `total_halt_loss`, which was
                                active_mask.float().mean() * 0.05 on a bool
                                tensor and therefore had requires_grad = False
                                (plan.md 5.1 / T3.2).
            depth_exits       : [B*S, max_depth] float — 1 where token exited
            all_expert_idx    : list[Tensor[N_active]] per depth step
            avg_depth         : scalar float (mean recursion steps, real tokens only)
            oracle_routing_ce : scalar — direct CE on router logits (0 if no labels)
            first_route_flat  : [B*S] long — expert chosen at depth 1 (-1 for pad)
            route_stat_totals : dict — dispatch/capacity accounting (T2.3)
            expected_depth    : [B*S] DIFFERENTIABLE N + R per token. This is the
                                quantity the curriculum supervision regresses
                                (metrics.halting_supervision_loss).
            halt_stats        : dict — forced/early exit counts, mean remainder,
                                mean halt mass. Diagnostics only, no grad.
        """
        B, S, D = x.shape
        N = B * S

        flat_x        = x.reshape(N, D)
        current_state = flat_x.clone()

        # ACT output accumulator: sum over steps of state_t * halt_weight_t.
        # Replaces the old "write the state at exit time" buffer. The weights of
        # the steps a token takes sum to exactly 1, so this is a convex
        # combination and the halt probabilities affect the OUTPUT, which is the
        # gradient path from the task loss into the halt heads (plan.md 5.2).
        accumulator = torch.zeros_like(flat_x)

        # Initialise active_mask from step_mask so pad tokens NEVER enter the loop
        active_mask = step_mask.reshape(N)   # [N] bool — False for pad positions

        # ---- ACT state, all over the full [N] token space -----------------
        # halting_prob : cumulative halting mass c_t per token
        # remainders   : the leftover mass R assigned at the halt step. THIS is
        #                the differentiable part of the ponder cost.
        # n_updates    : integer count of steps each token actually took.
        halting_prob = torch.zeros(N, device=x.device, dtype=flat_x.dtype)
        remainders   = torch.zeros(N, device=x.device, dtype=flat_x.dtype)
        n_updates    = torch.zeros(N, device=x.device, dtype=flat_x.dtype)
        forced_exits = 0
        early_exits  = 0

        # Flat oracle labels — [N], -1 for pad.  None if not provided.
        flat_experts = step_experts.reshape(N) if step_experts is not None else None


        total_bal_loss    = torch.tensor(0.0, device=x.device)
        # T4.1: number of MoEBlock calls this wrapper made == number of balance
        # terms produced. The balance loss is divided by this at the end so its
        # magnitude does not grow with recursion depth (plan.md 6.1).
        bal_calls         = 0
        oracle_routing_ce = torch.tensor(0.0, device=x.device)
        oracle_depth_count = 0

        # [N, max_depth] — records at which depth step each token exited
        depth_exits     = torch.zeros(N, self.max_depth, device=x.device)
        all_expert_idx  = []          # list of [N_active] per depth step
        tokens_per_step = []          # diagnostic: active count per step
        first_route_flat = torch.full((N,), -1, dtype=torch.long, device=x.device)

        # T2.3 dispatch accounting, accumulated over the depth loop.
        route_stat_totals = {
            "tokens": 0.0, "dispatched": 0.0, "overflow": 0.0,
            "expert_evaluations": 0.0,
            "max_load_fraction": 0.0, "experts_called": 0.0,
            "dispatch_steps": 0.0,
            # T4.2: summed here, divided by `dispatch_steps` at the end so the
            # reported term is a mean over the same calls the loss averages over.
            "entropy_term": 0.0, "switch_aux_term": 0.0,
        }

        # T-L4.3: attention query waste, accumulated over the depth loop. Attention
        # is computed at ALL S positions and its output discarded at halted ones, so
        # some of the work is thrown away. That is a real cost and it is REPORTED
        # rather than hidden -- `plan_language.md` §6.3 accepts it for clarity, on
        # the grounds that a variable-length gather of queries would complicate the
        # mask and the exactness of the depth accounting for no scientific gain.
        attn_query_slots   = 0.0
        attn_query_wasted  = 0.0

        for depth in range(1, self.max_depth + 1):
            if not active_mask.any():
                break

            # ---- 0. Causal self-attention over the FULL sequence ---------
            # T-L4.0/4.2/4.3. Shared parameters at every depth: `self.attn` is
            # constructed once, so this is the same tensors at depth 1 and depth 7.
            #
            # HALTED TOKENS STAY ATTENDABLE, AND STAY FROZEN. Attention runs over
            # all S positions using each position's CURRENT state, so a halted
            # position remains a valid key and value -- position 4 halting early
            # must not blind position 40 to the word at position 4, which would
            # couple one token's prediction to another token's halting decision.
            # But nothing is WRITTEN at a halted position: the `torch.where` keeps
            # `current_state` there, which is `updated_rules.md` §2.2's "halted
            # states are frozen and are not recursively recomputed", enforced by
            # the same mask the MoE write-back uses.
            #
            # No `key_padding_mask`: language packs to exactly `seq_len` with the
            # trailing partial block dropped (T-L2.2), so `step_mask` is all True
            # and a fully-masked attention row -- the NaN path -- is unreachable.
            if self.attention:
                seq = current_state.reshape(B, S, D)
                attn_out, _ = self.attn(
                    seq, seq, seq,
                    attn_mask=self.causal_mask(S, x.device),
                    need_weights=False,
                )
                attn_flat = self.attn_norm(current_state + attn_out.reshape(N, D))
                current_state = torch.where(
                    active_mask.unsqueeze(-1), attn_flat, current_state)
                attn_query_slots  += float(N)
                attn_query_wasted += float(N - int(active_mask.sum()))

            # ---- 1. Gather active tokens --------------------------------
            active_inputs = current_state[active_mask]    # [N_active, D]
            N_active      = active_inputs.shape[0]
            tokens_per_step.append(N_active)
            active_positions = active_mask.nonzero(as_tuple=True)[0]   # [N_active]

            # ---- 2. MoE forward ----------------------------------------
            (moe_out, b_loss, expert_idx,
             router_logits, step_route_stats) = self.moe_block(active_inputs)
            all_expert_idx.append(expert_idx)

            # T2.3: accumulate dispatch counts across depth steps. Summed, not
            # averaged, because "how many tokens were dropped" is a count over
            # every dispatch the block performed -- a token routed at depth 1
            # and again at depth 4 is two dispatches, each of which could in
            # principle overflow. max_load_fraction is kept as the WORST step,
            # since capacity pressure is a peak quantity, not a mean.
            route_stat_totals["tokens"]     += step_route_stats["tokens"]
            route_stat_totals["dispatched"] += step_route_stats["dispatched"]
            route_stat_totals["overflow"]   += step_route_stats["overflow"]
            route_stat_totals["expert_evaluations"] +=                 step_route_stats["expert_evaluations"]
            route_stat_totals["max_load_fraction"] = max(
                route_stat_totals["max_load_fraction"],
                step_route_stats["max_load_fraction"],
            )
            route_stat_totals["experts_called"] = max(
                route_stat_totals["experts_called"],
                step_route_stats["experts_called"],
            )
            route_stat_totals["dispatch_steps"] += 1.0
            route_stat_totals["entropy_term"]    += step_route_stats["entropy_term"]
            route_stat_totals["switch_aux_term"] += step_route_stats["switch_aux_term"]

            if depth == 1:
                first_route_flat = first_route_flat.clone()
                first_route_flat[active_positions] = expert_idx

            # ---- 3. Residual + LayerNorm --------------------------------
            updated = self.layer_norm(active_inputs + moe_out)

            # Write updated state back; clone to avoid in-place on grad tensors
            current_state    = current_state.clone()
            current_state[active_positions] = updated

            # T4.1 (plan.md 6.1): SUMMED here, then divided by the number of
            # calls after the loop. It is accumulated rather than averaged
            # in-place because the number of depth steps is not known until the
            # loop ends -- ACT may break early when every token has halted.
            total_bal_loss = total_bal_loss + b_loss
            bal_calls      = bal_calls + 1

            # ---- Oracle routing CE — direct on router logits -----------
            # This is the SHORT gradient path: router_logits → oracle CE.
            # The router learns immediately which expert each op type belongs to,
            # without waiting for gradients to travel back through the full
            # MoREWrapper depth loop via step_cls_head.
            if flat_experts is not None:
                active_oracle = flat_experts[active_positions]    # [N_active]
                valid_oracle  = (active_oracle >= 0) & (active_oracle < self.num_experts)
                if valid_oracle.any():
                    oracle_routing_ce = oracle_routing_ce + F.cross_entropy(
                        router_logits[valid_oracle], active_oracle[valid_oracle]
                    )
                    oracle_depth_count += 1

            # ---- 4. ACT halting  (T3.1, plan.md 5.2) --------------------
            # Canonical Adaptive Computation Time (Graves 2016) as used in the
            # Universal Transformer. This is the standard formulation, not a
            # novel one -- plan.md 5.2 explicitly forbids inventing a halting
            # equation.
            #
            # Per token, per step:
            #     p_t                = sigmoid(halt_head(state))
            #     cumulative mass    c_t = sum of p over steps taken so far
            #     halts when         c_t + p_t > 1 - eps
            #     remainder          R = 1 - c_t   (the leftover mass)
            #     step weight        w_t = p_t while running, R at the halt step
            #
            # The weights of every step a token takes sum to EXACTLY 1, so the
            # output is a convex combination of the per-step states. That is
            # what carries halt-head gradient into the TASK loss: change p_t and
            # you change how much of each step's state reaches the output. The
            # boolean `> threshold` decides dispatch only, which updated_rules.md
            # 2.2 permits, while the objective keeps a differentiable path.
            halt_logits = torch.zeros(N_active, device=x.device)
            for e, halt_head in enumerate(self.expert_halt_heads):
                emask = (expert_idx == e)
                if emask.any():
                    halt_logits = halt_logits.clone()
                    halt_logits[emask] = halt_head(updated[emask]).squeeze(-1)
            halt_probs = torch.sigmoid(halt_logits)                 # [N_active]

            cum_active = halting_prob[active_positions]             # [N_active]
            last_step  = (depth == self.max_depth)

            if self.fixed_depth:
                # No adaptive halting: every token runs the full depth and exits
                # at the last step with weight 1. Halt probabilities are computed
                # but do not gate anything, so the halt heads receive no gradient
                # -- correct, because this configuration has no halting to train.
                stop_local = torch.ones(N_active, dtype=torch.bool, device=x.device) \
                    if last_step else torch.zeros(N_active, dtype=torch.bool,
                                                  device=x.device)
                step_weight = stop_local.to(x.dtype)
                new_cum     = cum_active + step_weight
                remainder_c = step_weight
            else:
                # A token halts when the cumulative mass would cross the
                # threshold, or when it is forced out at max_depth.
                would_cross = (cum_active + halt_probs) > self.halt_threshold
                stop_local  = would_cross | last_step

                cont = (~stop_local).to(x.dtype)
                stop = stop_local.to(x.dtype)

                # Remainder for halting tokens: the mass left unassigned. Carries
                # gradient through cum_active, hence through every earlier p.
                remainder_c = stop * (1.0 - cum_active)
                # Continuing tokens contribute p_t and accumulate it.
                step_weight = cont * halt_probs + remainder_c
                new_cum     = cum_active + cont * halt_probs + remainder_c

            # Scatter the updated cumulative mass back into the [N] buffer.
            halting_prob = halting_prob.clone()
            halting_prob[active_positions] = new_cum

            remainders = remainders.clone()
            remainders[active_positions] = (
                remainders[active_positions] + remainder_c
            )

            # n_updates counts one step per token per step actually taken,
            # whether it continued or halted here. Differentiable only through
            # `remainders`; the count itself is discrete, which matches ACT.
            n_updates = n_updates.clone()
            n_updates[active_positions] = n_updates[active_positions] + 1.0

            # ---- 5. Accumulate the ACT-weighted output -----------------
            # accumulator += state_t * w_t. Weights sum to 1 per token, so this
            # is mean-preserving and cannot rescale the regression target.
            accumulator = accumulator.clone()
            accumulator[active_positions] = (
                accumulator[active_positions]
                + updated * step_weight.unsqueeze(-1)
            )

            # ---- 6. Record exits and evict halted tokens ---------------
            # Halted tokens leave the active set, so their state is FROZEN: it is
            # never gathered, recomputed or overwritten again (updated_rules.md
            # 2.2). Verified bit-identically by the Phase 3 gate.
            stop_global = active_positions[stop_local]               # indices in [N]
            depth_exits[stop_global, depth - 1] = 1.0
            if last_step:
                forced_exits = forced_exits + int(stop_local.sum().item())
            else:
                early_exits = early_exits + int(stop_local.sum().item())

            active_mask = active_mask.clone()
            active_mask[stop_global] = False

        # --- T4.1 Balance-loss normalization  (plan.md 6.1) ----------------
        # MEAN over the (depth) calls this wrapper actually made, not the sum.
        # Summed, the term's magnitude was a function of `max_depth`: a
        # 7-step model received ~7x the balance penalty of a 1-step model for
        # IDENTICAL routing behaviour, so the balance objective silently
        # dominated the task loss in deep configurations and any depth or
        # block-count comparison was mathematically confounded (plan.md 6.1).
        # Dividing by the realised call count -- not by max_depth -- is what makes
        # it depth-invariant under ACT, where the loop can break early.
        if bal_calls > 0:
            total_bal_loss = total_bal_loss / float(bal_calls)
            route_stat_totals["entropy_term"]    /= float(bal_calls)
            route_stat_totals["switch_aux_term"] /= float(bal_calls)

        # T-L4.3: the attention query waste, as a MEASURED fraction. Emitted only
        # when attention actually ran -- on the arithmetic path there is no
        # attention sublayer, so there is no waste to report, and a 0.0 there would
        # be a sentinel presented as a measurement (CLAUDE.md §4). Its consumers
        # must treat the key's absence as "not applicable", exactly as they do for
        # the routing keys at E == 1.
        if self.attention and attn_query_slots > 0:
            route_stat_totals["attn_query_slots"] = attn_query_slots
            route_stat_totals["attn_query_wasted"] = attn_query_wasted
            route_stat_totals["attn_query_waste_fraction"] = (
                attn_query_wasted / attn_query_slots)

        # Normalise oracle CE by number of depth steps that contributed labels
        if oracle_depth_count > 0:
            oracle_routing_ce = oracle_routing_ce / oracle_depth_count

        # --- Compute average recursion depth (real tokens only) --------
        real_token_mask = step_mask.reshape(N)   # [N] bool
        exit_steps = (
            depth_exits
            * torch.arange(1, self.max_depth + 1, device=x.device, dtype=torch.float32)
            .unsqueeze(0)
        ).sum(dim=-1)                            # [N]
        real_exit_steps = exit_steps[real_token_mask]
        avg_depth = (
            real_exit_steps.mean()
            if real_exit_steps.numel() > 0
            else torch.tensor(0.0, device=x.device)
        )

        # --- Differentiable ponder cost  (T3.2, plan.md 5.3) --------------
        # Canonical ACT ponder cost is N(t) + R(t): the number of steps taken
        # plus the remainder. N(t) is discrete and contributes no gradient; R(t)
        # does, and R shrinks as a token's earlier halt probabilities grow, so
        # minimising this term pushes p_t up and depth down. That is the
        # mechanism by which the cost "discourages unconditional max-depth
        # recursion" (plan.md 5.3) rather than merely reporting a number.
        #
        # Normalised by (max_depth + 1) -- the largest value N + R can take --
        # so the reported figure is in (0, 1] and is comparable across depth
        # settings. Without this the term's magnitude would grow with max_depth,
        # the same defect Phase 4 fixes for the balance loss.
        expected_depth = n_updates + remainders           # [N], differentiable via R
        if real_token_mask.any():
            ponder_cost = (
                expected_depth[real_token_mask].mean() / float(self.max_depth + 1)
            )
        else:
            ponder_cost = torch.zeros((), device=x.device, dtype=flat_x.dtype)

        halt_stats = {
            "forced_exits":     float(forced_exits),
            "early_exits":      float(early_exits),
            "real_tokens":      float(real_token_mask.sum().item()),
            "mean_remainder":   float(remainders[real_token_mask].mean().item())
                                if real_token_mask.any() else 0.0,
            "mean_halt_mass":   float(halting_prob[real_token_mask].mean().item())
                                if real_token_mask.any() else 0.0,
        }

        return (
            accumulator.reshape(B, S, D),
            total_bal_loss,
            ponder_cost,
            depth_exits,
            all_expert_idx,
            avg_depth,
            oracle_routing_ce,
            first_route_flat,
            route_stat_totals,
            expected_depth,
            halt_stats,
        )


class MoREModel(nn.Module):
    """
    Full MoRE model stacking `num_blocks` MoREWrapper layers.

    Input projection:
      step_proj: Linear(step_feat_dim → d_model) + LayerNorm + GELU
      Applied per-row to [B, max_steps, step_feat_dim] → [B, max_steps, d_model].
      S is now real (1–7 steps), NOT always 1.

    After MoRE blocks, masked mean-pooling over real steps → [B, d_model].

    Three output heads:
      regression_head : d_model → 1             — normalised scalar prediction
      cls_head        : d_model → num_families  — whole-program family class
      step_cls_head   : d_model → num_families  — per-step expert-family class
                        (used for step_routing_ce supervision; applied before pool)

    T8.3 -- why the two auxiliary heads are sized by `num_families` and NOT by
    `num_experts`. Their targets are the ORACLE FAMILY LABELS, which are a
    property of the DATASET, not of the model: data/script.py stamps
    `"family": "E1".."E6"` into every record and families.py:FAMILY_TO_IDX maps
    them to 0..5. That label space is 6 whatever the model's expert count is --
    measured identical at num_experts=6 and num_experts=1.

    While both heads were `Linear(d_model, num_experts)`, MoR (num_experts=1)
    fed 0..5 targets into a 1-class cross-entropy and died on a CUDA
    device-side assert (`t >= 0 && t < n_classes`, engine.py:464) in the first
    batch of the first epoch -- i.e. MoR could not train at all, so one third of
    the Phase 9 matrix was unexecutable. The range guard that would have caught
    it already existed at ONE of the three oracle-label call sites
    (model.py:570, `active_oracle < self.num_experts`) and not at the other two.

    This does NOT violate "all tensor dims derive from num_experts"
    (CLAUDE.md 2): the ROUTER still derives from num_experts, and the rest of
    that same rule requires "all labels derive from one family manifest" --
    which is exactly what sizing these two label heads by the manifest means.

    Total params target: ~10-20M.
    """

    def __init__(
        self,
        step_feat_dim: int,
        d_model: int = 256,
        num_experts: int = NUM_EXPERTS_CANONICAL,    # T5.1: from the manifest
        max_depth: int = 7,                          # recursion budget, NOT E
        num_blocks: int = 2,
        dropout: float = 0.1,
        fixed_depth: bool = False,
        num_op_types: int = NUM_OP_TYPES,
        routing_mode: str = CANONICAL_ROUTING_MODE,
        router_noise: str = CANONICAL_ROUTER_NOISE,
        router_noise_init: float = ROUTER_NOISE_INIT_SCALE_DEFAULT,
        router_noise_anneal_steps: int = ROUTER_NOISE_ANNEAL_STEPS_DEFAULT,
        ffn_mult: int = 4,
        # T8.3: the ORACLE LABEL SPACE, from the family manifest. Deliberately a
        # separate argument from num_experts so the two can never be silently
        # unified again; a caller that passes the model's expert count here
        # re-introduces the MoR crash.
        num_families: int = NUM_EXPERTS_CANONICAL,
        attention: bool = False,
        n_heads: int = 4,
        max_seq_len: int | None = None,
        task: str = TASK_ARITHMETIC,
        vocab_size: int | None = None,
    ):

        super().__init__()
        self.d_model      = d_model
        self.num_experts  = num_experts
        self.routing_mode = routing_mode
        self.router_noise = router_noise
        self.ffn_mult     = ffn_mult

        # Per-step projection: each step row [step_feat_dim] → [d_model].
        # PyTorch applies Linear to the last dim, so [B, S, F] → [B, S, d_model].
        #
        # T-L5.2: ARITHMETIC ONLY. The language input adapter is a token embedding
        # (see the head block below), so `step_proj` and `op_embed` have no meaning
        # there. Not constructed rather than constructed-and-skipped, for the same
        # reason as `regression_head`: an unused module still contributes parameters
        # to the budget comparison. `op_embed` would additionally be
        # `nn.Embedding(0, d_model)` on language -- `lang_families.NUM_OP_TYPES` is
        # 0 because there is no operation axis -- which is legal to build and raises
        # only when indexed, i.e. the worst kind of latent trap.
        if str(task) != TASK_LANGUAGE:
            self.step_proj = nn.Sequential(
                nn.Linear(step_feat_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

            # Operation embedding (T1.2). Since Phase 1 the feature row carries
            # only numeric arguments, so WHICH function to compute must be told to
            # the model somehow. It is told as an embedding over OPERATIONS (16 op
            # codes), which is legitimate task information — never as the oracle
            # expert index in an input slot, which is prohibited
            # (updated_rules.md 3). Note the size is num_op_types, not
            # num_experts: the mapping op → expert is exactly what routing is
            # supposed to discover, so it must not be handed to the model through
            # the embedding table's shape either.
            self.op_embed = nn.Embedding(num_op_types, d_model)

        # Stack of MoRE blocks
        self.blocks = nn.ModuleList([
            MoREWrapper(d_model, max_depth, num_experts, dropout, fixed_depth,
                        routing_mode=routing_mode,
                        router_noise=router_noise,
                        router_noise_init=router_noise_init,
                        router_noise_anneal_steps=router_noise_anneal_steps,
                        ffn_mult=ffn_mult,
                        attention=attention,
                        n_heads=n_heads)
            for _ in range(num_blocks)
        ])

        # ------------------------------------------------------------------
        # T-L4.0  Learned absolute position embeddings
        # ------------------------------------------------------------------
        #
        # Under the same flag as the attention sublayer, and for the same reason it
        # is `absence` rather than a zeroed parameter: with `attention=False` there
        # is no `pos_embed` attribute, so the arithmetic state_dict is untouched.
        #
        # WHY LEARNED ABSOLUTE, NOT RoPE OR ALiBi (§6.6). It is the simplest scheme
        # that CANNOT interact with recursion depth. RoPE and ALiBi both modify
        # attention scores, and any positional signal that varied with step count
        # would confound the depth analysis -- the one measurement this study is
        # built to make. A `[seq_len, d_model]` table added once, before the depth
        # loop, is depth-invariant by construction.
        #
        # Added ONCE before the blocks, not per depth step: position is a property
        # of the token's place in the sequence, not of how much computation it has
        # received, and re-adding it every step would make the positional signal
        # grow with depth.
        self.attention = bool(attention)
        if self.attention:
            if max_seq_len is None:
                raise ValueError(
                    "attention=True requires max_seq_len: the positional table is "
                    "[max_seq_len, d_model] and a default would silently truncate "
                    "or oversize it. Pass the dataset's seq_len "
                    "(dataset_meta.json:seq_len)."
                )
            self.max_seq_len = int(max_seq_len)
            self.pos_embed = nn.Embedding(self.max_seq_len, d_model)
            # Same scale as the token embedding, for the same reason: the two are
            # ADDED, so a positional table at N(0, 1) beside a token table at
            # N(0, 0.02) would make position 50x louder than identity at
            # initialisation and the first epochs would be spent attenuating it.
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

        # T8.3: the oracle label space, kept on the module so engine.py can
        # reshape step_cls_out without re-deriving the width from num_experts --
        # which is the exact mistake that made MoR unrunnable.
        self.num_families = int(num_families)

        # ------------------------------------------------------------------
        # T-L5.0 / T-L5.2 / T-L5.3  Heads, per task
        # ------------------------------------------------------------------
        #
        # `plan_language.md` §7.1's table, made structural. The two paths construct
        # DIFFERENT heads rather than sharing a superset with some of them weighted
        # to zero, for the T-L5.2 reason: an unused head still contributes
        # parameters to the budget comparison and still invites a future reader to
        # weight it. Absence is the enforceable form of "has no meaning here".
        self.task = str(task)
        if self.task == TASK_LANGUAGE:
            if vocab_size is None:
                raise ValueError(
                    "task='language' requires vocab_size: the token embedding and "
                    "the tied LM head are both [vocab_size, d_model], and a "
                    "default would silently disagree with the corpus the manifest "
                    "was built for."
                )
            self.vocab_size = int(vocab_size)

            # T-L5.0. WEIGHT TYING, and `lm_head.weight` IS `tok_embed.weight` --
            # the same tensor object, not a copy kept in sync. Standard practice,
            # and here it is also what keeps the §6 parameter budget a statement
            # about the EXPERT STACK: two independent V x d_model tables would be
            # 2 x 2.1 M parameters at V = 8192, d_model = 256, which is most of a
            # 10-20 M budget spent on lookup. Tying also puts the embedding under
            # gradient from both the input and the output path.
            #
            # `bias=False` is required, not stylistic: a bias would be a per-token
            # logit offset with no counterpart in the embedding, so the head would
            # no longer be the transpose of the input map.
            self.tok_embed = nn.Embedding(self.vocab_size, d_model)
            self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
            self.lm_head.weight = self.tok_embed.weight

            # T-L5.1: TYING MAKES THE EMBEDDING INIT SCALE AN OUTPUT-LOGIT SCALE,
            # and `nn.Embedding`'s default is N(0, 1). Measured with that default:
            # `task_loss` at initialisation was **167.6 nats**, against the uniform
            # floor ln(8192) = 9.011 -- an 18x overshoot and a perplexity of 1e73.
            # The arithmetic is direct: logits = h @ W.T, so with h of unit RMS the
            # logit spread is ~std(W) * sqrt(d_model) = 1 * 16, and a 16-nat spread
            # over 8192 classes is a confidently wrong distribution rather than a
            # uniform one. A model starting there spends its first epochs undoing
            # its own initialisation, which shows up as a suspiciously steep early
            # loss curve rather than as an error.
            #
            # std=0.02 is the GPT-2 convention and puts the logit spread at ~0.32,
            # so initialisation sits at the uniform floor -- which is what T-L5.1's
            # `task_loss ~ ln(V)` check verifies, and which is exactly why that
            # check is in the ledger.
            nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)

            # T-L5.3. A READ-ONLY PROBE, and the detach happens at its INPUT in
            # `forward`, not here. It reports how linearly decodable the POS family
            # is from the trunk's representation WITHOUT being able to create that
            # decodability -- see the forward-pass comment for why that distinction
            # is the whole point of §7.3.
            self.family_probe = nn.Linear(d_model, self.num_families)
        else:
            # Regression head: predict normalised output scalar (from pooled repr)
            self.regression_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

            # Whole-program auxiliary classification head (whole-program family).
            # T5.1 (plan.md 7.1): width is NOT a literal. It was hard-coded to 7
            # while the canonical setup runs six families, so class 6 was an
            # unreachable logit that softmax still had to normalise over -- every
            # family probability was scaled down by an expert that cannot exist,
            # and the argmax could in principle land on it.
            # T8.3: width is num_families (the DATASET's label space, always 6),
            # not num_experts (the MODEL's expert count, 1 for MoR). See the class
            # docstring for why MoR crashed while these were the same argument.
            self.cls_head = nn.Linear(d_model, self.num_families)

            # Per-token step classification head — applied per-step token BEFORE
            # pooling.  Supervised by per-step oracle labels (step_routing_ce
            # loss). This teaches the router: ADD token → Expert 0, MULT → 1, etc.
            # Same T5.1 and T8.3 reasoning, and it matters more here: this head is
            # what the routing-accuracy and confusion-matrix metrics read.
            #
            # NOT shared with the language path's `family_probe`, deliberately: on
            # arithmetic this head IS in the objective at weight 0.5 and shapes the
            # trunk, which is the confound §7.3 refuses to inherit. Two names for
            # two different scientific roles.
            self.step_cls_head = nn.Linear(d_model, self.num_families)


    def forward(
        self,
        x: torch.Tensor,
        step_mask: torch.Tensor,
        step_experts: torch.Tensor | None = None,
        step_ops: torch.Tensor | None = None,
    ):
        """
        Args:
            x            : [B, max_steps, step_feat_dim] — numeric ARGUMENTS only.
                           Contains no step result and no oracle expert index
                           (plan.md 3.1, 3.2).
            step_mask    : [B, max_steps] bool
            step_experts : [B, max_steps] long — oracle expert labels (-1 for pad).
                           Threaded into each MoREWrapper to compute direct oracle
                           CE on router logits (short, clean gradient path). This
                           is a supervision TARGET, not an input feature.
            step_ops     : [B, max_steps] long — operation code per step (-1 pad).
                           Required: it is the model's only signal for WHICH
                           function each step computes.
        Returns (13-tuple):
            reg_out           : [B, 1]
            cls_out           : [B, num_families]
            step_cls_out      : [B, max_steps, num_families]  (diagnostic head)
            total_bal_loss    : scalar — AVERAGED over blocks, each block's value
                                already averaged over its depth calls (T4.1).
                                Depth- and block-invariant by construction.
            total_ponder_cost : scalar, differentiable. AVERAGED over blocks, not
                                summed: CLAUDE.md 2 requires the halting cost's
                                magnitude not grow with block count, otherwise a
                                deeper model is penalised for its shape rather
                                than its behaviour.
            depth_exits       : [B*max_steps, max_depth]  (from last block)
            avg_depth         : scalar float              (from last block)
            expert_idx_last   : list of routing indices from last block
            oracle_routing_ce : scalar — direct CE on router logits across all blocks
            first_route_flat  : [B*max_steps] long — block-0 depth-1 router choice
            route_stats_total : dict — dispatch/capacity accounting (T2.3)
            expected_depth    : [B*max_steps] differentiable N + R, from the LAST
                                block. Curriculum supervision target (T3.3).
            halt_stats_total  : dict — exit counts and rates aggregated over
                                blocks, with forced/early rates recomputed from
                                the summed counts.
        """
        # [B, max_steps, step_feat_dim] → [B, max_steps, d_model]
        if self.task == TASK_LANGUAGE:
            # T-L5.0. `x` is `input_ids [B, S]` long, not a float feature matrix.
            # `step_ops` is all -1 for language (there is no operation axis) and is
            # accepted but unused, so the arithmetic caller's argument order is
            # untouched.
            if x.dtype not in (torch.long, torch.int32, torch.int64):
                raise TypeError(
                    f"task='language' expects integer input_ids, got {x.dtype}. "
                    "A float tensor here would be silently embedded by index "
                    "truncation."
                )
            h = self.tok_embed(x)
        else:
            h = self.step_proj(x)

            # Add the operation embedding (T1.2). Since the feature row is numeric
            # arguments only, omitting this would leave the task unlearnable in
            # principle: "compute f(args)" with f unknown. Padded steps use index 0
            # and are then zeroed, so the pad op contributes nothing.
            if step_ops is None:
                raise ValueError(
                    "MoREModel.forward requires step_ops. Operation identity is "
                    "the only remaining signal for WHICH function each step "
                    "computes (the oracle expert index was removed from the input "
                    "in Phase 1, plan.md 3.2). Passing None would silently train a "
                    "model that cannot know the operation."
                )
            op_valid = (step_ops >= 0)
            op_emb   = self.op_embed(step_ops.clamp(min=0))
            h = h + op_emb * op_valid.unsqueeze(-1).to(h.dtype)

        # Zero out pad positions after projection so they carry no signal
        h = h * step_mask.unsqueeze(-1).float()

        # T-L4.0: positions added ONCE, before the depth loop -- see the
        # `pos_embed` construction comment for why not per step. After the pad
        # zeroing so a pad position stays exactly zero rather than carrying a
        # position vector into attention as a key.
        if self.attention:
            _S = h.shape[1]
            if _S > self.max_seq_len:
                raise ValueError(
                    f"sequence length {_S} exceeds max_seq_len {self.max_seq_len}: "
                    "the positional table has no entry for those positions. "
                    "Rebuild the model with the dataset's seq_len rather than "
                    "letting the table wrap."
                )
            _pos = torch.arange(_S, device=h.device)
            h = h + self.pos_embed(_pos).unsqueeze(0)
            h = h * step_mask.unsqueeze(-1).float()

        total_bal_loss       = torch.tensor(0.0, device=x.device)
        total_ponder_cost    = torch.tensor(0.0, device=x.device)
        total_oracle_routing = torch.tensor(0.0, device=x.device)
        depth_exits_last     = None
        avg_depth_last       = None
        expert_idx_last      = None
        first_route_block0   = None
        expected_depth_last  = None
        halt_stats_total     = {"forced_exits": 0.0, "early_exits": 0.0,
                                "real_tokens": 0.0, "mean_remainder": 0.0,
                                "mean_halt_mass": 0.0}
        route_stats_total    = {
            "tokens": 0.0, "dispatched": 0.0, "overflow": 0.0,
            "expert_evaluations": 0.0,
            "max_load_fraction": 0.0, "experts_called": 0.0,
            "dispatch_steps": 0.0,
            "entropy_term": 0.0, "switch_aux_term": 0.0,
        }

        for block_idx, block in enumerate(self.blocks):
            (
                h, b_loss, block_ponder, depth_exits, block_expert_idx,
                avg_depth, block_oracle_ce, first_route, block_route_stats,
                block_expected_depth, block_halt_stats,
            ) = block(h, step_mask, step_experts)
            # T4.1: each block's balance loss is already a mean over its own
            # depth calls, so summing here would reintroduce the magnitude
            # defect along the block axis. Divided by the block count below --
            # the same mean-of-means treatment the ponder cost gets.
            total_bal_loss       = total_bal_loss       + b_loss
            # Ponder cost is AVERAGED over blocks, not summed. Each block's cost
            # is already normalised to (0, 1] by its own max_depth, so summing
            # would make the term's magnitude scale with num_blocks -- exactly
            # the depth/block magnitude defect plan.md 6.1 forbids.
            total_ponder_cost    = total_ponder_cost    + block_ponder
            total_oracle_routing = total_oracle_routing + block_oracle_ce
            depth_exits_last     = depth_exits
            avg_depth_last       = avg_depth
            expert_idx_last      = block_expert_idx
            expected_depth_last  = block_expected_depth
            # Exit counts sum over blocks (each block makes its own exit
            # decisions); the mass diagnostics are averaged at the end.
            halt_stats_total["forced_exits"]   += block_halt_stats["forced_exits"]
            halt_stats_total["early_exits"]    += block_halt_stats["early_exits"]
            halt_stats_total["real_tokens"]    += block_halt_stats["real_tokens"]
            halt_stats_total["mean_remainder"] += block_halt_stats["mean_remainder"]
            halt_stats_total["mean_halt_mass"] += block_halt_stats["mean_halt_mass"]
            # T2.3: dispatch counts sum over blocks (each block dispatches
            # independently); capacity pressure is the worst block.
            for k in ("tokens", "dispatched", "overflow", "dispatch_steps",
                      "expert_evaluations"):
                route_stats_total[k] += block_route_stats[k]
            # T4.2: already per-call means inside the block; averaged over blocks
            # below so the reported term matches the normalized loss.
            for k in ("entropy_term", "switch_aux_term"):
                route_stats_total[k] += block_route_stats[k]
            for k in ("max_load_fraction", "experts_called"):
                route_stats_total[k] = max(route_stats_total[k],
                                           block_route_stats[k])
            # T-L4.3: attention query waste, summed over blocks. Present only when
            # the blocks actually have an attention sublayer, so the key's ABSENCE
            # is what the arithmetic path reports -- not a 0.0, which would be a
            # sentinel standing in for "not applicable" (CLAUDE.md §4). The
            # fraction is recomputed from the summed slots below rather than
            # averaged, because averaging per-block fractions would weight a block
            # that ran two depth steps the same as one that ran seven.
            for k in ("attn_query_slots", "attn_query_wasted"):
                if k in block_route_stats:
                    route_stats_total[k] = (route_stats_total.get(k, 0.0)
                                            + block_route_stats[k])
            if block_idx == 0:
                first_route_block0 = first_route

        _nb = max(len(self.blocks), 1)
        total_ponder_cost = total_ponder_cost / _nb
        # T4.1 (plan.md 6.1): mean over blocks, so the balance term is invariant
        # to BOTH axes -- recursion depth (divided inside each wrapper) and block
        # count (divided here). Each block contributes equally regardless of how
        # deep it recursed; this coincides with a flat mean over all (block,
        # depth) calls whenever the blocks run the same number of steps, and
        # equal-per-block weighting is the deliberate choice where they do not,
        # matching how the ponder cost is aggregated.
        total_bal_loss    = total_bal_loss / _nb
        route_stats_total["entropy_term"]    /= _nb
        route_stats_total["switch_aux_term"] /= _nb
        # T-L4.3: one fraction over the summed slots, so a block that ran seven
        # depth steps contributes seven steps' worth of waste rather than one
        # block's worth.
        if route_stats_total.get("attn_query_slots", 0.0) > 0:
            route_stats_total["attn_query_waste_fraction"] = (
                route_stats_total["attn_query_wasted"]
                / route_stats_total["attn_query_slots"])
        halt_stats_total["mean_remainder"] /= _nb
        halt_stats_total["mean_halt_mass"] /= _nb
        _ex = halt_stats_total["forced_exits"] + halt_stats_total["early_exits"]
        # Rates, not raw counts, so they are comparable across batch sizes.
        # Measured zeros -- a model that never force-exits genuinely reports 0.0.
        halt_stats_total["forced_exit_rate"] = (
            halt_stats_total["forced_exits"] / _ex if _ex > 0 else 0.0
        )
        halt_stats_total["early_exit_rate"] = (
            halt_stats_total["early_exits"] / _ex if _ex > 0 else 0.0
        )

        # Overflow RATE as a fraction of all dispatches -- a real measurement,
        # never a sentinel (T2.3). Under CAPACITY_POLICY = "no_capacity_limit"
        # this is a measured 0.0 because nothing is dropped, which is a
        # different statement from "unmeasured".
        _disp = route_stats_total["dispatched"]
        route_stats_total["overflow_rate"] = (
            route_stats_total["overflow"] / _disp if _disp > 0 else 0.0
        )
        route_stats_total["capacity_policy"] = CAPACITY_POLICY
        # Expert evaluations per dispatched token: 1.0 for true Top-1 sparse,
        # num_experts for the dense ablation. This is the single number that
        # distinguishes the two paths in any log or table.
        route_stats_total["evals_per_token"] = (
            route_stats_total["expert_evaluations"] / _disp if _disp > 0 else 0.0
        )
        route_stats_total["routing_mode"]    = self.routing_mode

        # Per-token step classification head — diagnostic / secondary supervision.
        # step_cls_head applies to h after MoRE blocks; used to monitor whether
        # the learned representations are separable by expert class.
        #
        # T-L5.3 -- THE DETACH IS THE WHOLE POINT, AND IT IS HERE. On arithmetic
        # this head sits in the objective at weight 0.5, so its gradient shapes the
        # trunk, and "MoRE's representation separates operation families" is
        # permanently weaker for it: some of the observed structure could be this
        # head's gradient rather than the router's own behaviour. The arithmetic
        # write-up handles that with the `no_family_supervision` ablation, which is
        # the right remedy after the fact.
        #
        # Language does not inherit the confound. `h.detach()` cuts the trunk out of
        # the probe's backward graph entirely, so the probe can REPORT how linearly
        # decodable the POS family is without being able to CREATE that
        # decodability. Every routing/AMI/purity number then measures emergent
        # structure with no family signal anywhere in the trunk's objective, and the
        # probe's loss weight is scientifically inert rather than tuned.
        if self.task == TASK_LANGUAGE:
            step_cls_out = self.family_probe(h.detach())
        else:
            step_cls_out = self.step_cls_head(h)   # [B, max_steps, num_families]

        if self.task == TASK_LANGUAGE:
            # T-L5.0 / T-L5.2. No pooling: every position is a prediction, so
            # collapsing the sequence would throw the task away. `reg_out` carries
            # the per-position vocabulary logits and `cls_out` is None -- a packed
            # LM block has no whole-sequence family, so there is no such head to
            # call. The tuple ARITY is unchanged so `engine.py` needs no second
            # training loop (plan.md §9); the slots' meanings are documented in the
            # `MoRELanguageDataset` docstring's table and in §7.1.
            logits = self.lm_head(h)               # [B, S, vocab_size]
            return (
                logits,
                None,
                step_cls_out,
                total_bal_loss,
                total_ponder_cost,
                depth_exits_last,
                avg_depth_last,
                expert_idx_last,
                total_oracle_routing,
                first_route_block0,
                route_stats_total,
                expected_depth_last,
                halt_stats_total,
            )

        # Masked mean-pool over real steps (exclude pad positions)
        mask_f   = step_mask.unsqueeze(-1).float()                    # [B, S, 1]
        h_pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        reg_out = self.regression_head(h_pooled)   # [B, 1]
        cls_out = self.cls_head(h_pooled)           # [B, num_families]

        return (
            reg_out,
            cls_out,
            step_cls_out,
            total_bal_loss,
            total_ponder_cost,
            depth_exits_last,
            avg_depth_last,
            expert_idx_last,
            total_oracle_routing,
            first_route_block0,
            route_stats_total,
            expected_depth_last,
            halt_stats_total,
        )


