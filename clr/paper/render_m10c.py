"""M10c frozen CSV renderers; no model environment required."""
from clr.paper.render import Context, escape, number, interval, one


def cost(c):
    f=c.read('cost'); rows=[]
    for q in ['Q3','Q4']:
        for m in ['B3','B6','CLR-lite','B5c']:
            a=one(f,model=m,qtype=q,batch=1); b=one(f,model=m,qtype=q,batch=64)
            acc={'mean':a.test_accuracy,'lo':a.accuracy_lo,'hi':a.accuracy_hi}
            rows.append([q,escape(m),interval(acc,100),number(a.trainable_parameters/1e6),number(a.total_parameters/1e6),
                         number(a.flops_per_query/1e9)+' / '+number(b.flops_per_query/1e9),
                         number(a.source_step_ratio)+' / '+number(b.source_step_ratio),
                         number(a.latency_per_batch_ms),number(a.queries_per_second),
                         number(b.latency_per_batch_ms),number(b.queries_per_second),
                         number(a.historical_training_gpu_hours_3seeds)])
    c.write_data(f)
    c.table(['Type','Reader','Test acc. (\\%)','Trainable (M)','Total (M)','GFLOPs/query (1/64)','Source-step ratio (1/64)',
             'b1 ms/batch','b1 query/s','b64 ms/batch','b64 query/s','Training GPUh (3 seeds)'],rows,
            'Fixed seed11 weights, shared RTX4090 GPU4; 64 reader_val problems shared across models. Timings: median of three rounds, three inner loops after two warmups; feature preparation, tokenization and CPU-to-GPU transfer excluded. GPU is shared with concurrent jobs: these are contended, implementation-dependent timings, not isolated hardware limits. FLOPs: matrix and attention operations, 2 FLOP/MAC; no lookup, activation, norm or softmax. B5c includes full frozen Qwen3-1.7B forward, injection, unmerged rank16 LoRA and final vocabulary head. Padding changes b64 FLOPs. Source denominator: mean cached one-token CODI-GPT2 step, unmerged LoRA, projection and vocabulary head, no prefill. B6 total includes frozen option embedding table (38.7M); its trainable parameters and B3 parameters are per-type models. CLR/B5c share Q3/Q4 weights. Historical completed resident training costs include loading/validation: CLR and B6 train Q1-Q4 (B6 includes both learning-rate sweeps), B5c trains Q3-Q4 for one pass. B3 isolated training cost is unavailable (NA); entire original B0-B4 multi-type/three-seed fitting/calibration/evaluation run costs 0.534 GPUh. Costs are repeated across types, not additive. Accuracy uses frozen three-seed test predictions, problem-bootstrap 95% CI; no new test inference.')


def conformal(c):
    f=c.read('conformal');rows=[]
    for split in ['reader_val','test']:
        for q in ['Q3','Q4']:
            rows.append([escape(split),q]+[interval(one(f,split=split,qtype=q,metric=m),scale) for m,scale in [
                ('coverage_clr',100),('coverage_b5c',100),('coverage_difference',100),('size_clr',1),('size_b5c',1),('size_difference',1)]])
    c.write_data(f)
    c.table(['Split','Type','CLR cover. (\\%)','B5c cover. (\\%)','B5c$-$CLR (pp)','CLR size','B5c size','Paired size difference'],rows,
            'Existing cal_B Mondrian thresholds at alpha=0.1; no recalibration. Exact problem/step/seed pairing of the frozen selected query instances. Differences are B5c minus CLR; 10000 problem-bootstrap draws preserve steps and the three fixed seeds. Coverage differs; set-size differences do not establish dominance at equal coverage. Reader_val is diagnostic, not a fresh exchangeable test sample.')


def main():
    import argparse,yaml
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();c=Context(yaml.safe_load(Path(a.config).read_text()))
    {'t6_cost_precision':cost,'t6_conformal':conformal}[c.config['artifact']](c);c.finish()
