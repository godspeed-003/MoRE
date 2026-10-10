"""fix_refs.py -- one-shot bibliography repair (T-LA.4). Safe to re-run.

Applies the three verification outcomes from 2026-10-10:
  * leong2023ewok  -> ivanova2024ewok   (the placeholder was wrong on author,
                      year AND title; verified against the arXiv API)
  * vinh2010ami    -> add number = {95} (from JMLR's own BibTeX export)
  * raposo2024mod  -> confirmed arXiv-only; title cased as arXiv cases it
"""
import io
import re
import sys

P = 'paper/refs.bib'
s = io.open(P, encoding='utf-8').read()

# ---------------------------------------------------------------- EWoK --------
# The placeholder entry spanned from its warning comment block to the closing
# brace. Match it whole so no fragment of the fabricated author survives.
start = s.index('% !! UNVERIFIED -- DO NOT SUBMIT WITHOUT CHECKING !!')
end = s.index('}', s.index('note    = {UNVERIFIED ENTRY')) + 1
new_ewok = """% VERIFIED 2026-10-10 against the arXiv API (export.arxiv.org, id 2405.09605).
% THE PREVIOUS PLACEHOLDER WAS WRONG IN EVERY FIELD THAT MATTERS: it guessed
% first author "Leong", year 2023, and the title "A Benchmark for Measuring
% Systematic Generalization of Multi-domain World Knowledge". The real paper is
% first-authored by Anna A. Ivanova, dated 2024-05-15, and titled as below. The
% old key `leong2023ewok` is therefore retired rather than corrected in place --
% keeping it would preserve a wrong surname in every citation.
@article{ivanova2024ewok,
  title   = {Elements of World Knowledge ({EWoK}): A Cognition-Inspired Framework
             for Evaluating Basic World Knowledge in Language Models},
  author  = {Ivanova, Anna A. and Sathe, Aalok and Lipkin, Benjamin
             and Kumar, Unnathi and Radkani, Setayesh and Clark, Thomas H.
             and Kauf, Carina and Hu, Jennifer and Pramod, R. T.
             and Grand, Gabriel and Paulun, Vivian and Ryskina, Maria
             and Aky{\\\"u}rek, Ekin and Wilcox, Ethan and Rashid, Nafisa
             and Choshen, Leshem and Levy, Roger and Fedorenko, Evelina
             and Tenenbaum, Joshua and Andreas, Jacob},
  journal = {arXiv preprint arXiv:2405.09605},
  year    = {2024}
}"""
s = s[:start] + new_ewok + s[end:]

# ---------------------------------------------------------------- AMI ---------
old_vinh = """@article{vinh2010ami,
  title   = {Information Theoretic Measures for Clusterings Comparison:
             Variants, Properties, Normalization and Correction for Chance},
  author  = {Vinh, Nguyen Xuan and Epps, Julien and Bailey, James},
  journal = {Journal of Machine Learning Research},
  volume  = {11},
  pages   = {2837--2854},
  year    = {2010}
}"""
new_vinh = """% VERIFIED 2026-10-10 against JMLR's own BibTeX export
% (jmlr.org/papers/v11/vinh10a.bib). Only `number` was missing.
@article{vinh2010ami,
  title   = {Information Theoretic Measures for Clusterings Comparison:
             Variants, Properties, Normalization and Correction for Chance},
  author  = {Vinh, Nguyen Xuan and Epps, Julien and Bailey, James},
  journal = {Journal of Machine Learning Research},
  volume  = {11},
  number  = {95},
  pages   = {2837--2854},
  year    = {2010}
}"""
assert old_vinh in s, 'vinh entry not found as expected'
s = s.replace(old_vinh, new_vinh)

# ---------------------------------------------------------------- MoD ---------
old_mod = """@article{raposo2024mod,
  title   = {Mixture-of-Depths: Dynamically Allocating Compute in
             Transformer-Based Language Models},"""
new_mod = """% VERIFIED 2026-10-10 against the arXiv API (id 2404.02258, submitted
% 2024-04-02). Author list and ID confirmed exactly; no journal_ref, so it
% remains arXiv-only. Title cased as arXiv cases it.
@article{raposo2024mod,
  title   = {Mixture-of-Depths: Dynamically allocating compute in
             transformer-based language models},"""
assert old_mod in s, 'raposo entry not found as expected'
s = s.replace(old_mod, new_mod)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

keys = re.findall(r'@[a-z]+\{([A-Za-z0-9_]+),', s)
assert 'leong2023ewok' not in keys, 'old EWoK key survived'
assert 'ivanova2024ewok' in keys, 'new EWoK key missing'
assert 'UNVERIFIED' not in s, 'an UNVERIFIED marker survived'
print(f'refs.bib repaired: {len(keys)} entries, no UNVERIFIED markers remain')
print('  retired key : leong2023ewok')
print('  new key     : ivanova2024ewok')
