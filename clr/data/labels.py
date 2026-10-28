"""Frozen M0 answer extraction; invalid predictions never match invalid gold."""
import re
from decimal import Decimal, InvalidOperation

def numeric_answer(text):
    values=re.findall(r'-?\d+\.?\d*',text.replace(',',''))
    if not values:return None
    try:return str(Decimal(values[-1]).normalize())
    except InvalidOperation:return None

def score_answer(pred,gold,task):
    if task=='gsm8k':
        target=numeric_answer(gold)
        if target is None:raise ValueError('Invalid gold numerical answer')
        return numeric_answer(pred)==target
    if task=='prosqa':return pred.split('###')[-1].strip()==gold.strip()
    raise ValueError(task)

def canonical_answer(pred,task):
    return numeric_answer(pred) if task=='gsm8k' else pred.split('###')[-1].strip()
