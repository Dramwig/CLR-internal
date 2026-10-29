# M0b CODI-LLaMA 复现 v2

作者checkpoint与ModelScope底模均已校验。官方脚本未修改；strict加载只用于拒绝遗漏权重。

运行清单：
```json
{
  "source_smoke": "/home/me/PaperField/CalReader/runs/20261029T013734197681Z_7bcbbe6fbd3d_77ffd6",
  "official_smoke": "/home/me/PaperField/CalReader/runs/20261029T013822186629Z_e9f09c818e4c_64a9d7",
  "determinism": "/home/me/PaperField/CalReader/runs/20261029T013901994681Z_66cf21a10c33_09328c",
  "official_full": "/home/me/PaperField/CalReader/runs/20261029T013954709535Z_f594d64e7e97_b69ebd",
  "source_full": "/home/me/PaperField/CalReader/runs/20261029T014056589002Z_261f9f964915_639303"
}
```

官方原始1319题准确率及按题10k bootstrap、清洁1312子集：
```json
{
  "status": "completed",
  "run_id": "20261029T013954709535Z_f594d64e7e97_b69ebd",
  "config_sha256": "f594d64e7e97f33f82c9269d7940a18a147a56585d3f18843a59fb4211a3c5d9",
  "questions": 1319,
  "accuracy": {
    "mean": 0.5549658832448825,
    "ci95": [
      0.5284306292645944,
      0.5815011372251706
    ],
    "n_questions": 1319,
    "bootstrap_resamples": 10000,
    "bootstrap_rng_seed": 20261028
  },
  "clean_subset_accuracy": {
    "mean": 0.5533536585365854,
    "ci95": [
      0.5259146341463414,
      0.5800304878048781
    ],
    "n_questions": 1312,
    "bootstrap_resamples": 10000,
    "bootstrap_rng_seed": 20261028
  },
  "wall_seconds": 53.35860576410778,
  "gpu_hours": 0.014821834934474382,
  "source_forward_calls": 176,
  "source_forward_sample_evaluations": 21104
}
```

CLR batch16准确率及与官方逐题一致率：
```json
{
  "status": "completed",
  "run_id": "20261029T014056589002Z_261f9f964915_639303",
  "config_sha256": "261f9f96491530b7e9f86fba81f1ecc88118e9d4b09fafc0dbb7d7d705f616ec",
  "batch_sizes_finished": [
    "16"
  ],
  "results": {
    "16": {
      "accuracy": {
        "mean": 0.5526914329037149,
        "ci95": [
          0.5261561789234268,
          0.579226686884003
        ],
        "n_questions": 1319,
        "bootstrap_resamples": 10000,
        "bootstrap_rng_seed": 20261028
      },
      "agreement_first_batch": {
        "mean": 1.0,
        "ci95": [
          1.0,
          1.0
        ],
        "n_questions": 1319,
        "bootstrap_resamples": 10000,
        "bootstrap_rng_seed": 20261028
      },
      "agreement_official": {
        "mean": 0.979529946929492,
        "ci95": [
          0.9719294920394239,
          0.9871114480667172
        ],
        "n_questions": 1319,
        "bootstrap_resamples": 10000,
        "bootstrap_rng_seed": 20261028
      }
    }
  },
  "protocol_sha256": "259432a7b666368c0b9b80c01c9c0f0c26cd6d6ab7f92ff5672c4cbe55a0c197",
  "wall_seconds": 128.65166056412272,
  "gpu_hours": 0.03573657237892298,
  "source_forward_calls": 1284
}
```

与既有M0报告暂列论文参考51.9%差值为3.597pp。若绝对差>2pp，须继续核验checkpoint/论文配置对应关系、prompt及精度协议，不能通过改划分/挑seed追平。该数值不是已完成原因分析。

确定性逐题潜状态差见determinism run；本轮没有训练/噪声曲线结果，不宣称M0b整体验收。
