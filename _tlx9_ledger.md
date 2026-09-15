
- [x] **T-LX.9 The training loop retained every routing index tensor of the epoch,
  and MoR paid ~7.4 GiB for a statistic it reports as `N/A`.** The epoch loop kept
  `epoch_all_expert_idx`, a list of every `[N_active]` int64 argmax tensor, for every
  depth step, of every batch, alive until the epoch ended — solely to feed
  `compute_expert_load_entropy` once. On canonical language that is
  `7 depths x 12,288 tokens x 8 B x 10,965 batches = ~7.4 GiB` of **live** tensors in
  ~77 k separate allocations. **MoR OOM'd on a rented 16 GB V100 at the first backward
  of epoch 2**, and MoR has `num_experts = 1`, where T6.4 makes the entropy `None` by
  definition — so the arm was paying 7.4 GiB to feed a number it never reports.
  **Verify:** `test_lang_recursion.py` asserts the entropy computed from the
  incrementally accumulated count vector is **bit-identical** to the entropy computed
  from the retained index tensors, at E = 1 (both `None`), 2 and 6, using the engine's
  own float32 accumulation order; that no `epoch_all_expert_idx` accumulator survives
  in executable engine code; and that the engine emits per-epoch peak GPU memory.
  **Evidence (2026-09-14, CPU interpreter): `test_lang_recursion.py` 17 passed / 0
  failed / 0 skipped** (was 11; TLX.9a–d are this task), bit-identity at E=2
  `0.9997463226318359` and E=6 `0.9999266266822815` on both paths. Gate L0 **TOTAL 356
  356 0 0, ALL GATES PASS**.
  - **Why `.cpu()` on the retained tensors is not the fix.** The obvious one-line patch
    — `epoch_all_expert_idx.extend(t.detach().cpu() for t in batch_expert_idx)` — moves
    the same 7.4 GiB into host RAM, on a box with 21 GB, and leaves ~77 k Python objects
    and a CPU-side `(idx == e)` scan per expert per tensor at epoch end. It relieves the
    symptom on the GPU and keeps the redundancy. **The list has no reader that the
    counts cannot serve**: `epoch_expert_counts` at engine.py's epoch head already
    accumulates `(idx_tensor == e).float().sum()` over exactly the same tensors, and
    that count vector is the *complete* input to `H / log(E)`. The list is deleted, not
    relocated.
  - **The refactor is numerically inert, which is what makes it safe to land mid-study.**
    `metrics.expert_load_entropy_from_counts(counts, E)` holds the one definition of the
    statistic; `compute_expert_load_entropy(depth_exits, idx_list, E)` is kept because
    Gate L0's G4.16 exercises that signature, and it now counts and delegates. Same
    float32 accumulator, same per-batch addition order, so every published
    `val/routing_load_entropy_norm` keeps its meaning and no prior run is
    reinterpreted. `perf/*` and metric keys are not config fields, so **no
    `config_hash` moves and the five finished MoE runs stay admissible.**
  - **The empty-epoch branch had to be preserved explicitly.** The old guard was
    `if epoch_all_expert_idx:` — falsy when nothing was routed. With the list gone the
    guard is `epoch_expert_counts.sum() > 0`. Without it, an unrouted epoch would report
    `-0.000000` rather than `N/A`, which is exactly the sentinel-as-measurement failure
    CLAUDE.md §4 forbids; TLX.9b pins that all-zero counts produce a finite number and
    therefore that the guard, not a NaN, is what yields N/A.
  - **`perf/gpu_peak_alloc_gib` and `perf/gpu_peak_reserved_gib`, reset per epoch.** The
    OOM had to be diagnosed from a traceback and an arithmetic estimate because **no run
    in this repo recorded how much memory it used**, so "epoch 1 fits, epoch 2 batch 0
    does not" could not be told apart from "the batch never fit" without buying another
    GPU hour. Both keys are needed and they answer different questions: `allocated` is
    live tensor bytes (a **rising** series across epochs = retention), `reserved` is what
    the caching allocator holds from the driver, and the gap between them is
    fragmentation — an OOM can be caused by either and the remedies differ.
  - **What was ruled out, so the next OOM is not re-diagnosed from scratch.**
    `torch.save(model.state_dict(), ...)` saves and discards, retaining nothing.
    `depth_hist` is a CPU tensor. `epoch_halt_stats`, `epoch_route_stats` and
    `depth_allocation_error` carry Python floats (`float(...)` / `.item()` at the model
    boundary), not graph-bearing tensors. The validation accumulators
    (`_lang_depth_ed/_ids/_loss`, `_lang_route`, `_lang_ids_full`) are already `.cpu()`
    and the val split is 1,098 blocks, so they are ~MB. **`wandb.watch(model,
    log="gradients", log_freq=50)` is deliberately left in place**: it is part of the
    logging pipeline, removing it changes what a canonical run records, and it is not
    implicated by a per-epoch growth pattern. If the instrumented run shows a rising
    `perf/gpu_peak_alloc_gib` with this fix in, `wandb.watch` is the next thing to test —
    as a labelled variant, never as a silent edit.
  - **Batch size was not touched.** `batch_size = 48` is a shared `enforced_field` frozen
    for the 8 GB card (T-L7.0/T-L7.1); lowering it to fit memory would invalidate the
    five MoE runs and silently change the optimization problem. When a run does not fit,
    the defect gets fixed, not the protocol.
