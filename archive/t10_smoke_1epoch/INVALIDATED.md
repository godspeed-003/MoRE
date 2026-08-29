# T10 smoke runs — 1 epoch, NOT experiments

Six run directories produced while bringing up `automated/phase10_ablations.py`.
Every one was launched with `--epochs 1` against a Phase 10 arm config, purely to
answer *does this arm construct and complete a step at all* before committing
~6.4 h of GPU time to the real batch.

**These may never appear in any table.** `training.epochs = 1` against the
canonical 50 is a proxy in the strictest sense (CLAUDE.md §6), and
`one_field_check()` in the driver correctly reports them as differing from the
canonical MoRE baseline in **two** fields — the ablated field *and* `epochs`.
That is what they were used to demonstrate:

```
t10smoke_two_blocks_seed42 -> unintended second difference training.epochs: baseline 50 vs arm 1
```

They are archived rather than deleted only because the repository rule is to
archive evidence. Nothing here is a measurement.

## What they found

`t10smoke_router_noise_seed42__03f023ba` has **no metrics.json** — it crashed.
The arm had been declared as `model.router_noise = "gaussian"`, which is not a
legal value; `more/model.py:71` defines `ROUTER_NOISE_MODES = ("none",
"fixed_annealed", "trainable")` and the `MoEBlock` constructor raised.

The failure mode worth recording: `resolve_variant` had already produced the
label `router_noise_gaussian` without complaint, because it string-formats
whatever value it is handed and validates nothing. So the label layer will
happily name a configuration that cannot run. The driver now validates enum
fields against `more.model`'s own tuples at config-generation time
(`LEGAL_VALUES` in `automated/phase10_ablations.py`), which is why
`t10smoke_router_noise_seed42__55ebfccc` — the `fixed_annealed` retry — exists
and completed.

`fixed_annealed` and not `trainable` was chosen deliberately: CLAUDE.md §2 states
the L2-on-trainable-noise-scale mechanism is not a proven fix, so ablating
`trainable` would confound router noise with that unproven machinery.

## Throughput observed (1 epoch, used only to size the real batch)

| arm | tok/s | implied min/run at 50 epochs |
|---|---|---|
| dense_routing | 3052 | 17.7 |
| routing_supervision | 2744 | 19.6 |
| fixed_depth | 2193 | 24.6 |
| router_noise | 2074 | 26.0 |
| two_blocks | 1315 | 41.0 |

Canonical MoRE measured 2073 tok/s over the T9.1 matrix, which is what these were
scaled against. Note dense routing came out *faster* than Top-1 sparse: the
workload is launch-overhead-bound, so evaluating all six experts as one batched
matmul can cost less wall clock than gathering and scattering for one. Treat that
as a hypothesis to confirm on the real arm, not a result — one epoch on a
downclocked laptop GPU is not a throughput measurement.
