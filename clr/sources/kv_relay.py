"""Untrained, same-weight Qwen3 KV relay. K is post-RoPE at physical positions.

Both calls use a frozen backbone, sequentially. Sender sees public + private;
receiver receives public + prefix only. No gold/meta enters either model.
Idea A utilities are imported read-only after checking the frozen generator.
"""
import hashlib
import json
from pathlib import Path
import sys
import torch

CORE = ('latdef/data/synthetic_v2.py', 'latdef/data/synthetic.py', 'latdef/data/schema.py')
PINNED_CORE_SHA256 = {
    CORE[0]:'30879ec2915fc3ef18f3322374c4dff3ea06aac11054c9592269162dcd4bf01b',
    CORE[1]:'8621ebd00d532071b138377d03e99cfc7fbb7a81091aba810940933c63a1ecab',
    CORE[2]:'3ff77c5e553809738d8f025839c90562bcbaf7b046537f9c80f5ea27716ed815'}

def verify_idea_a(root):
    sys.dont_write_bytecode = True
    root = Path(root)
    frozen = json.loads((root/'docs/v2-design-freeze.json').read_text())
    hashes = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in CORE}
    if any(hashes[p] != PINNED_CORE_SHA256[p] or frozen['source_sha256'][p]!=PINNED_CORE_SHA256[p] for p in CORE):
        raise RuntimeError('Frozen synthetic v2 core source SHA256 mismatch')
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    import yaml
    if yaml.safe_load((root/'configs/synthetic/v2.yaml').read_text())!=frozen['dataset_config']:
        raise RuntimeError('Idea A v2 generator recipe differs from frozen m1-v2-r2 configuration')
    return hashes


def sender_text(example):
    return example.x_pub + '\n\nPrivate records:\n' + example.x_priv


class KVRelay:
    def __init__(self, config):
        self.config = config
        self.core_hashes = verify_idea_a(config['idea_a'])
        from latdef.s2.model import FrozenS2
        from latdef.s2.budgets import BudgetArm
        self.s2 = FrozenS2.from_pretrained(config['model_path'], device=config['device'],
                                         max_context=config['max_context'])
        mode = config.get('answer_mode', 'direct')
        self.arm = BudgetArm(mode, config['model_path'], mode, config.get('max_new_tokens', 0))
        self.forward_calls = 0
        self.forward_examples = 0
        self.prefill_tokens = 0

    @torch.no_grad()
    def encode_texts(self, texts):
        from latdef.inject.cache import PrefixKV, left_pad
        model, tok = self.s2.model, self.s2.tokenizer
        sequences = [tok.encode(t, add_special_tokens=False) for t in texts]
        if max(map(len, sequences)) > self.config['max_context']:
            raise ValueError('Sender exceeds context; truncation forbidden')
        ids, mask = left_pad(sequences, tok.pad_token_id, self.s2.device)
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand(len(texts), -1)
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        self.forward_calls += 1; self.forward_examples += len(texts)
        self.prefill_tokens += sum(map(len, sequences))
        prefix = PrefixKV(tuple((x.keys, x.values) for x in output.past_key_values.layers),
                          mask, frozenset(range(model.config.num_hidden_layers)))
        prefix.validate(model.config, len(texts))
        return prefix

    def encode(self, examples):
        return self.encode_texts([sender_text(e) for e in examples])

    @torch.no_grad()
    def score(self, examples, prefix=None, *, sender=False, answer_mode=None):
        from latdef.handoff.base import S2Input
        from latdef.s2.budgets import BudgetArm
        arm = self.arm
        if sender and self.config.get('sender_answer_mode'):
            mode = self.config['sender_answer_mode']
            arm = BudgetArm(mode, self.config['model_path'], mode, self.config['sender_max_new_tokens'])
        if answer_mode is not None:
            arm=BudgetArm(answer_mode,self.config['model_path'],answer_mode,512 if answer_mode=='cot' else 0)
        inputs = [S2Input(e.x_pub, tuple(e.options), extra_text=e.x_priv if sender else '') for e in examples]
        out = self.s2.answer_batch(inputs, arm, prefix=prefix)
        decode_calls = max(out['n_gen_tokens']) if arm.mode == 'cot' else 0
        self.forward_calls += 1 + decode_calls
        self.forward_examples += len(examples) * (1 + decode_calls)
        self.prefill_tokens += sum(out['score_prefill_tokens'])
        if arm.mode == 'cot':
            self.prefill_tokens += sum(len(p) for p in self.s2._prepare(inputs,arm,prefix)[0])
        if sender:
            from clr.data.m7 import format_commitment
            self.last_sender=[{'mode':arm.mode,'pred':int(out['pred'][i]),
                               'probs':out['probs'][i].float().cpu().tolist(),
                               'commitment':format_commitment(e.id,int(out['pred'][i])),
                               'generated_ids':list(out['cached_cot'].generated_ids[i]) if arm.mode=='cot' else [],
                               'generated_text':self.s2.tokenizer.decode(out['cached_cot'].generated_ids[i],skip_special_tokens=False) if arm.mode=='cot' else e.options[int(out['pred'][i])],
                               'correct':int(out['pred'][i])==e.label} for i,e in enumerate(examples)]
        return torch.stack(out['probs']).float().cpu()

    def commitments(self, examples):
        if self.config.get('commitment_policy') == 'programmatic_verified':
            from latdef.data.synthetic_v2 import solve_v2_text
            from clr.data.m7 import format_commitment
            # Explicit synthetic protocol: the sending pipeline computes the
            # visible statement symbolically. This is not Qwen generation.
            labels=[solve_v2_text(e) for e in examples]
            probs=torch.nn.functional.one_hot(torch.tensor(labels),num_classes=4).float()
            return [format_commitment(e.id,y) for e,y in zip(examples,labels)],probs
        # Visible conclusion is the sender's actual prediction, never generator gold.
        probs = self.score(examples, sender=True)
        return [f'COMMITMENT: choose {e.options[int(p.argmax())]}.' for e, p in zip(examples, probs)], probs


def permute_prefix(prefix, order):
    from latdef.inject.cache import PrefixKV
    idx = torch.as_tensor(order, device=prefix.mask.device)
    return PrefixKV(tuple((k[idx], v[idx]) for k, v in prefix.layers), prefix.mask[idx],
                    prefix.active_layers, prefix.gates)


def audit_tensor(prefix, batch_index, layers):
    """Layer-major [L*S, 2*H*D], excluding padding; values are actual cache outputs."""
    valid = prefix.mask[batch_index].bool()
    chunks = []
    for layer in layers:
        k, v = prefix.layers[layer]
        chunks.append(torch.cat([k[batch_index, :, valid].permute(1,0,2).flatten(1),
                                 v[batch_index, :, valid].permute(1,0,2).flatten(1)], -1))
    z = torch.cat(chunks).detach().to(device='cpu', dtype=torch.float16).contiguous()
    return z, {'layers': list(layers), 'tokens_per_layer': int(valid.sum()),
               'token_range': [int(valid.nonzero()[0]), int(valid.nonzero()[-1])+1],
               'mask': valid.cpu().tolist(), 'active_layers': sorted(prefix.active_layers),
               'gates': None if prefix.gates is None else prefix.gates.detach().float().cpu().tolist(),
               'head_concat': 'K_heads_then_V_heads', 'K': 'post-RoPE', 'dtype': 'float16'}
