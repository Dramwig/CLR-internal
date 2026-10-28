"""Conservative source-training contamination audit; text only, never labels."""
import re,unicodedata
from collections import defaultdict

def tokens(text):
    s=unicodedata.normalize('NFKC',text).casefold().replace('’',"'").replace('‘',"'")
    return re.findall(r"[+-]?\d+(?:\.\d+)?|[^\W\d_]+(?:'[^\W\d_]+)*",s,flags=re.UNICODE)

def normalize(text):return ' '.join(tokens(text))
def grams(text,n=13):
    t=tokens(text)
    return {tuple(t[i:i+n]) for i in range(max(0,len(t)-n+1))}

def audit(train,targets,n=13):
    exact=defaultdict(set);index=defaultdict(set);result={}
    for row in targets:
        key=row['id'];text=row['question'];g=grams(text,n)
        exact[normalize(text)].add(key)
        for gram in g:index[gram].add(key)
        result[key]={'exact_training_rows':[],'ngram_training_row_count':0,'ngram_witnesses':[], 'matched_grams':set(),'total_target_grams':len(g)}
    for i,row in enumerate(train):
        text=row['question']
        for key in exact.get(normalize(text),()):result[key]['exact_training_rows'].append(i)
        hits=defaultdict(set)
        for gram in grams(text,n):
            for key in index.get(gram,()):hits[key].add(gram)
        for key,matched in hits.items():
            rec=result[key];rec['ngram_training_row_count']+=1;rec['matched_grams'].update(matched)
            if len(rec['ngram_witnesses'])<5:
                rec['ngram_witnesses'].append({'training_row':i,'training_question':text,'matched_gram':' '.join(sorted(matched)[0])})
    for key,rec in result.items():
        rec['matched_gram_count']=len(rec.pop('matched_grams'))
        rec['overlap']=bool(rec['exact_training_rows'] or rec['ngram_training_row_count'])
    return result
