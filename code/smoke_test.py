import torch, sys
sys.path.insert(0, '.')
from train import MoREDataset, MoREModel, OP_TO_EXPERT

print('=== Test 1: Dataset shapes ===')
ds = MoREDataset('../data/dummy-train.jsonl', max_steps=7, step_feat_dim=12, max_val=1e6, pad_value=0.0, num_experts=7)
x, sm, se, so, fam, dep, tgt = ds[0]
print(f'x.shape      : {x.shape}')
print(f'step_mask    : {sm.tolist()}')
print(f'step_experts : {se.tolist()}')
print(f'step_ops     : {so.tolist()}')
print(f'family={fam.item()}, depth={dep.item()}, target={tgt.item():.4f}')

print()
print('=== Test 2: Model forward pass ===')
m = MoREModel(step_feat_dim=12, d_model=64, num_experts=7, max_depth=3, num_blocks=1, dropout=0.0)
xb  = torch.randn(4, 7, 12)
smb = torch.ones(4, 7, dtype=torch.bool)
smb[:, 4:] = False
out = m(xb, smb)
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
