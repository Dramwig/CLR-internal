"""Group uncertainty for the deterministic C1 validation panel."""
import numpy as np
from scipy.special import softmax


def interval(mean, draws):
    return {'mean': float(mean), 'lo': float(np.quantile(draws, .025)),
            'hi': float(np.quantile(draws, .975))}


def auc_group_draws(labels,scores,group_index,weights,batch_size=128):
    """Exact weighted AUROC draws; all rows in a group share its weight."""
    labels=np.asarray(labels,int);scores=np.asarray(scores);gi=np.asarray(group_index,int)
    if not set(labels).issubset({0,1}) or len(set(labels))!=2:
        raise ValueError('AUROC requires two binary labels')
    order=np.argsort(scores,kind='stable');yy=labels[order];gg=gi[order];ss=scores[order]
    starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1];result=[]
    for offset in range(0,len(weights),batch_size):
        sample=weights[offset:offset+batch_size,gg].astype(np.float64)
        pos=np.add.reduceat(sample*(yy==1)[None],starts,axis=1)
        neg=np.add.reduceat(sample*(yy==0)[None],starts,axis=1)
        numerator=(pos*(np.cumsum(neg,axis=1)-neg+.5*neg)).sum(1)
        denominator=pos.sum(1)*neg.sum(1)
        if np.any(denominator==0):raise ValueError('AUC bootstrap draw lacks one label')
        result.append(numerator/denominator)
    return np.concatenate(result)


def gate_intervals(prediction, seed=20261107, repeats=10000):
    """Resample original/CF groups together, conditional on the fitted probe."""
    result = {}
    groups = sorted(set(prediction['A4']['group_ids']))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)),
                              size=repeats).astype(np.float32)
    for task in ['A1', 'A4']:
        item = prediction[task]
        gids = np.asarray(item['group_ids'])
        logits = item['logits'].float().numpy()
        labels = item['labels'].numpy()
        if sorted(set(gids)) != groups:
            raise ValueError('C1 tasks use different validation groups')
        correct = logits.argmax(1) == labels
        values = np.array([correct[gids == g].mean() for g in groups])
        result[task] = {'accuracy': interval(values.mean(), weights @ values / len(groups))}
        if task == 'A1':
            p = softmax(logits, axis=1)[:, 1]
            positive = [p[(gids == g) & (labels == 1)] for g in groups]
            negative = [p[(gids == g) & (labels == 0)] for g in groups]
            if (len(set(map(len, positive))) != 1 or
                    len(set(map(len, negative))) != 1 or not len(positive[0]) or
                    not len(negative[0])):
                raise ValueError('C1 A1 requires balanced, nonempty labels per group')
            pos, neg = np.stack(positive), np.stack(negative)
            # Rows are groups of true-fact questions; columns are groups of
            # false-fact questions. Both axes preserve original/CF pairing.
            matrix = np.zeros((len(groups), len(groups)), dtype=np.float32)
            for a in range(pos.shape[1]):
                for b in range(neg.shape[1]):
                    matrix += ((pos[:, a, None] > neg[None, :, b]) +
                               .5 * (pos[:, a, None] == neg[None, :, b]))
            matrix /= pos.shape[1] * neg.shape[1]
            draws = np.einsum('bi,ij,bj->b', weights, matrix, weights,
                              optimize=True) / len(groups) ** 2
            result[task]['AUROC'] = interval(matrix.mean(), draws)
    return {'groups': len(groups), 'repeats': repeats,
            'resampling_unit': 'original/CF group',
            'conditional_on': 'fixed seed-11 fitted checkpoint and validation panel',
            'metrics': result}
