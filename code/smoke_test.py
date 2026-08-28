import torch, sys
sys.path.insert(0, '.')
from more import MoREDataset, MoREModel, OP_TO_EXPERT, NUM_EXPERTS_CANONICAL

# T5.1: the expert count comes from the family manifest, never a literal. The 7s
# below that remain are MAX_STEPS (steps per program), a different quantity that
# happens to share the value; conflating them is the defect Phase 5 removed.
E = NUM_EXPERTS_CANONICAL
MAX_STEPS = 7

print('=== Test 1: Dataset shapes ===')
ds = MoREDataset('../data/dummy-train.jsonl', max_steps=MAX_STEPS, step_feat_dim=12, max_val=1e6, pad_value=0.0, num_experts=E)
x, sm, se, so, fam, dep, tgt = ds[0]
print(f'x.shape      : {x.shape}')
print(f'step_mask    : {sm.tolist()}')
print(f'step_experts : {se.tolist()}')
print(f'step_ops     : {so.tolist()}')
print(f'family={fam.item()}, depth={dep.item()}, target={tgt.item():.4f}')

print()
print('=== Test 2: Model forward pass ===')
m = MoREModel(step_feat_dim=12, d_model=64, num_experts=E, max_depth=3, num_blocks=1, dropout=0.0)
xb  = torch.randn(4, MAX_STEPS, 12)
smb = torch.ones(4, MAX_STEPS, dtype=torch.bool)
smb[:, 4:] = False
# step_experts is a supervision target; step_ops carries operation identity and
# is required since Phase 1 removed the oracle expert index from the input.
seb = torch.randint(0, E, (4, MAX_STEPS)); seb[:, 4:] = -1
sob = torch.randint(0, 16, (4, MAX_STEPS)); sob[:, 4:] = -1
out = m(xb, smb, seb, sob)
names = ['reg_out','cls_out','step_cls_out','bal_loss','halt_loss','depth_exits','avg_depth','expert_idx','oracle_routing_ce','first_route']
for n, o in zip(names, out):
    if hasattr(o, 'shape'):
        shape = tuple(o.shape)
    elif o is None:
        shape = 'None'
    elif isinstance(o, list):
        shape = f'list[{len(o)}]'
    else:
        shape = repr(o)
    print(f'{n:20s}: {shape}')

print()
print('=== Test 3: OP_TO_EXPERT ===')
print(sorted(OP_TO_EXPERT.keys()))

print()
print('ALL TESTS PASSED')
