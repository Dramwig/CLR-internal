"""Attack-side selection and fixed-denominator robustness accounting."""
import math

def selection_key(row):
    """Held-out audit predictions are deliberately absent from this key."""
    score=row.get('selection_nll')
    return (not row['target_achieved'],float('inf') if score is None else score,-row['target_logp'],row['lambda'])

def select_candidate(rows):
    if not rows:raise ValueError('No attack candidates')
    return min(rows,key=selection_key)

def failure(row,audit_key,alpha=.05):
    return bool(row['changed_to_target'] and not row['audit'][audit_key][str(alpha)])

def ratio_interval(receiver,reader):
    """Interval for a cost ratio; preserve right censoring and zero baselines."""
    if receiver['already_target'] or reader['already_target']:return None
    if receiver['censored']:return {'lower':None,'upper':None,'status':'receiver_censored'}
    if reader['censored']:
        return {'lower':reader['lower']/receiver['upper'],'upper':None,'status':'reader_censored'}
    return {'lower':reader['lower']/receiver['upper'],
            'upper':reader['upper']/receiver['lower'] if receiver['lower']>0 else None,
            'estimate':reader['upper']/receiver['upper'],'status':'finite_search'}
