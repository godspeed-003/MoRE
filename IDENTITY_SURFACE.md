# Identity surface on the pushed branch

**Branch.** `claude/english-language-dataset-migration-92946e`
**Scanned with.** `vedan|Vedant|Ayan|Users\\Hp|godspeed|vedantkorade|rshisode607|`
`mumbai-university|government-college|DESKTOP-1OH84V8|Vast\.ai|RunPod`
**Status.** 31 files remain, down from 393. The remainder is listed in full below.

---

## What was removed, and why it was safe

| group | files | action | why it was safe |
|---|---:|---|---|
| `archive/` | 4,016 | untracked from git | Invalidated history, already declared void. Its raw W&B payloads carried two author email addresses, two W&B entity names embedding an institution, a machine hostname and the GitHub account. The tree stays on disk. |
| `.env`, `.claude/settings.json` | 2 | gitignored | Live HuggingFace and Anthropic tokens. **Rotate both** — a token that has been in a working tree is a token to replace, not a token to hide. |
| root `acl_submission.pdf`, `tmlr_submission.pdf` | 2 | untracked | Four-day-stale copies of the submission PDFs. A reviewer opening the root one would have read the pre-T-LA.25 layout. |
| `automated/phase10_*.log` | 5 | untracked | Machine-written console logs carrying the W&B entity name, which embeds an institution. The per-run provenance in `runs/` is the citable record. |
| exploratory run configs | 332 | absolute paths made relative | Not admitted, not citable, and **quoted by no derived artifact**: `automated/phase10_ablations_result.json` names run directories and carries zero config hashes. |
| `code/`, `data/lang/` usage docstrings | ~20 | interpreter path -> `python` | The path is an example of how to invoke a script. The reader has their own interpreter. |
| `code/compile_papers.py` | 1 | two hardcoded temp roots -> `expanduser` | They were fallbacks behind the `TEMP`/`TMP` environment variables. |
| `.claude/skills/`, `.agents/skills/` | 3 | interpreter and worktree paths -> relative | The skill packs document how to run the gate, and the worktree path was machine-specific. |
| `SETUP.md` | 1 | machine named -> capacity described | "Ayan's RTX 4060 Laptop, 8 GB" -> "an 8 GB card". The VRAM headroom argument is what the sentence was for. |

---

## The 31 that remain

### A. The 20 files inside the 15 canonical run cells — a decision, not a task

| file | hit |
|---|---|
| `runs/langB_MoE_seed42__51000018/config.json` | `C:\\Users\\Hp\\Desktop\\Waste\\MoRE\\data\\*.jsonl` |
| `runs/langB_MoE_seed42__51000018/resolved_config.json` | same |
| `runs/langB_MoE_seed43__7d85fe81/config.json` | same |
| `runs/langB_MoE_seed43__7d85fe81/resolved_config.json` | same |
| `runs/langB_MoE_seed44__c348b39a/config.json` | same |
| `runs/langB_MoE_seed44__c348b39a/resolved_config.json` | same |
| `runs/langB_MoE_seed45__64800f08/config.json` | same |
| `runs/langB_MoE_seed45__64800f08/resolved_config.json` | same |
| `runs/langB_MoE_seed46__3d9d12a9/config.json` | same |
| `runs/langB_MoE_seed46__3d9d12a9/resolved_config.json` | same |
| `runs/langB_MoR_seed42__c93f0fe8__r2/config.json` | same |
| `runs/langB_MoR_seed42__c93f0fe8__r2/resolved_config.json` | same |
| `runs/langB_MoR_seed43__8b54a65f/config.json` | same |
| `runs/langB_MoR_seed43__8b54a65f/resolved_config.json` | same |
| `runs/langB_MoR_seed44__c11db02d/config.json` | same |
| `runs/langB_MoR_seed44__c11db02d/resolved_config.json` | same |
| `runs/langB_MoR_seed45__4041a890/config.json` | same |
| `runs/langB_MoR_seed45__4041a890/resolved_config.json` | same |
| `runs/langB_MoRE_seed42__9cfd3f1c/...` | **none** (Linux container path) |
| `runs/langB_MoRE_seed43__dbe38cbd/...` | **none** |
| `runs/langB_MoRE_seed44__16027752/...` | **none** |
| `runs/langB_MoRE_seed45__f6c47448/...` | **none** |
| `runs/langB_MoRE_seed46__d069325c/...` | **none** |

(The five MoRE cells are listed for completeness: they are already clean.)

**Why these are the hard ones.** The path is one field group -- `data` -- and
`config_hash` covers the `data` block. Only `provenance` and `logging` are excluded
from the hash. So rewriting the path changes the recorded `config_hash` and
desynchronises it from the run-directory name, which is derived from the hash's
first eight hex digits.

**What is actually exposed.** A two-letter Windows account name (`Hp`), a folder
name (`Waste`), and the repository name. No full name, no email, no institution.
The MoRE cells were trained on a rented V100 in a Linux container, so only the
MoE and MoR cells show a path at all -- which is itself the paper's disclosed
two-GPU split.

**The three options.**

1. **Leave it.** The leak is a username, not a name, and these files are the
   canonical provenance a reviewer will read. Cost: a determined reviewer who
   already knows the co-author can de-anonymise. Nothing else is affected.
2. **Rewrite the path identically across each arm's five seeds.** The
   seed-blind config-equality check still passes, because it compares arms to each
   other, not to a recorded hash. The recorded `config_hash` then no longer
   recomputes from the rewritten file. The exporter does not recompute it -- it
   reads the recorded value -- so the export still works, and the paper's numbers
   do not move. Cost: the recorded hash stops being verifiable against the file
   it came from, which is a real loss of a self-checking property.
3. **Rewrite the path and recompute the hash, then rename the run directories
   and update every reference to them.** The hash stays self-consistent, but the
   run names change, and those names appear in `results/language/results.json`,
   the paper's provenance list, the exporter's admission records, and the
   appendix. Cost: the largest of the three, and it touches the frozen spec.

**Recommendation.** Option 1, for a stated reason rather than by default: the
exposed string is a two-letter account name, it appears in exactly the files a
reviewer would inspect anyway, and every alternative either breaks a hash
self-check or renames the runs the paper cites. If the venue's anonymisation
policy is stricter than that, option 2 is the cheap fallback and it does not
move a single number in the paper.

### B. The 11 non-run files

| file | hit | note |
|---|---|---|
| `AGENTS.md` | both authors' names and machines | Section 9, "Two machines, two authors". Judgement call: how much of the machine record to keep. |
| `CLAUDE.md` | same | Same content as `AGENTS.md`; the two are near-duplicates under different names. |
| `TASKS_LANGUAGE.md` | the other author's name | Historical task ledger, kept deliberately as the project's changelog. |
| `changelog.md` | both authors' names | 43 hits. The full change history. |
| `automated/phase10_configs/_base_more.json` | `C:\Users\Hp\Desktop\Waste\MoRE` | As a **config value**. Rewriting it changes that config's hash. |
| `automated/phase10_configs/dense_routing.json` | same | same |
| `automated/phase10_configs/fixed_depth.json` | same | same |
| `automated/phase10_configs/ponder_cost_low.json` | same | same |
| `automated/phase10_configs/router_noise.json` | same | same |
| `automated/phase10_configs/routing_supervision.json` | same | same |
| `automated/phase10_configs/two_blocks.json` | same | same |

**On the seven sweep configs.** They are the *driver* configs for the arithmetic
ablation arms, not the arms' own resolved configs (those were scrubbed with the
other 332). Nothing quotes their hashes. Rewriting the path is safe in the sense
that nothing breaks, but it silently changes a config's identity, and these are
the one place where a changed hash could be noticed later by someone re-deriving
an ablation. They are left as they are.

**On the four prose files.** These name both authors and both machines, and they
are the project's working record rather than a submitted artifact. Two questions
for whoever does the anonymisation: whether the two-machine section is needed at
all once the artifact is anonymous, and whether `changelog.md`'s 43 name
mentions are worth a mechanical pass (they are all "Vedant's machine" / "Ayan's
machine" in running prose, so a find-and-replace would suffice) or whether the
history is better kept un-anonymised and the artifact built from a filtered
copy.
