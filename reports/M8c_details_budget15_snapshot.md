# M8c 预算关卡详细证据

未完成；test封闭，未创建final_lock.json。
用户修订见docs/M8c_TASK.md；15 GPUh不足至少200组时停下，以下不代表最终test结论。

## 实测与预测方法
Val five-epsilon group durations fit by nonnegative fixed/D0-step/R4-step costs; predict only the three requested epsilons. Lock rate is 97.5% bootstrap upper mean plus 20% timing margin. All remaining budget is offered to primary E3 before secondary items. Forecast is not a runtime guarantee.
新主协议预期最大n取floor((15−已花成本−加载预留)/规划组成本)，上限500；不读取test样本、标签、预测或根据test调整。按点吞吐也不足200组，故未执行主矩阵。
被截断的val最后一组不进入吞吐拟合与19组描述性失效表，已明确保留；可能使预测偏乐观。GPU计费为驻留墙钟，与利用率无关。
## 预算完整JSON
```json
{
  "run": "/home/me/PaperField/CalReader/runs/20261109T054410952270Z_fc76a4d3d035_de2212",
  "test_opened": false,
  "passed": false,
  "maximum_feasible_groups": 124,
  "maximum_at_point_rate": 181,
  "minimum_main_groups": 200,
  "budget_gpu_hours": 15,
  "spent_gpu_hours": 6.431750062113893,
  "remaining_gpu_hours": 8.568249937886108,
  "primary_200_gpu_hours": 13.73385587859723,
  "total_200_gpu_hours": 20.21560594071112,
  "point_primary_200_gpu_hours": 9.401632336806747,
  "point_total_200_gpu_hours": 15.88338239892064,
  "additional_gpu_hours_for_200": 5.2156059407111215,
  "primary_fraction_of_remaining_when_planning": 1.0,
  "loading_reserve_gpu_hours": 0.05,
  "rate_gpu_hours_per_group": 0.06866927939298616,
  "point_rate_gpu_hours_per_group": 0.04700816168403373,
  "point_rate_bootstrap_ci": [
    0.03851331674435525,
    0.05722439949415513
  ],
  "throughput_margin": 0.2,
  "bootstrap_resamples": 10000,
  "val_complete_groups": 19,
  "val_expected_groups": 20,
  "val_partial_groups": {
    "val-b7bae149ac3ef95a25265e01d0c5fcae815f409b8a3126b6b4eb2fafa413099b": 10
  },
  "val_run": "runs/20261109T041333353840Z_482eec68a407_a050a8",
  "val_status": "failed",
  "val_gpu_hours": 1.5000085215144725,
  "new_main_matrix": {
    "kind": "E3",
    "lambdas": [
      1.0
    ],
    "epsilons": [
      0.01,
      0.1,
      1.0
    ],
    "designs": [
      "D0",
      "R4"
    ],
    "seeds": [
      11,
      22,
      33
    ],
    "steps": 200
  },
  "model_seconds_coefficients": {
    "fixed_per_group": 8.075777966856247,
    "D0_per_PGD_step": 0.47587887342173724,
    "R4_per_PGD_step": 0.4322793667412195
  },
  "model_residual_RMSE_seconds": 12.448884603485814,
  "forecast_note": "Val five-epsilon group durations fit by nonnegative fixed/D0-step/R4-step costs; predict only the three requested epsilons. Lock rate is 97.5% bootstrap upper mean plus 20% timing margin. All remaining budget is offered to primary E3 before secondary items. Forecast is not a runtime guarantee.",
  "secondary_protocol": {
    "groups": 100,
    "order": [
      "lambda=.1/10 at epsilon=.1",
      "minimum search at most 6 bisections",
      "E1/E2 at epsilon=.1"
    ],
    "executed": false
  },
  "cropped_settings": {
    "epsilons": [
      0.03,
      0.3
    ],
    "old_test60_plan": "superseded by user; never locked or executed"
  },
  "input_sha256": {
    "runs/20261109T041333353840Z_482eec68a407_a050a8/results.jsonl": "98c3ab71e2ea08aa128d51eb7c933959a637617d115b9e28793bce376cc16a32",
    "runs/20261109T041333353840Z_482eec68a407_a050a8/metrics.json": "96f2ce05f1b48461ad5036e829244cef1e7d10d20ab82f2d914e98e650ce7010",
    "reports/M8c_evidence/val_main_v2.log": "f3a0d9c5573ebe4f24833006f1801fb22ad4c9320990f72967fe1c2793981d80",
    "docs/M8c_TASK.md": "3f74582eeeda29fd6da585e08e1b03502ead709ff3095b09e61642c71bc1089e"
  }
}
```
## 三seed曝光/QA/准入与val部分结果
```json
{
  "run": "/home/me/PaperField/CalReader/runs/20261109T055408308916Z_b752762ef390_3f7ccf",
  "complete": false,
  "test_opened": false,
  "budget": {
    "run": "/home/me/PaperField/CalReader/runs/20261109T054410952270Z_fc76a4d3d035_de2212",
    "test_opened": false,
    "passed": false,
    "maximum_feasible_groups": 124,
    "maximum_at_point_rate": 181,
    "minimum_main_groups": 200,
    "budget_gpu_hours": 15,
    "spent_gpu_hours": 6.431750062113893,
    "remaining_gpu_hours": 8.568249937886108,
    "primary_200_gpu_hours": 13.73385587859723,
    "total_200_gpu_hours": 20.21560594071112,
    "point_primary_200_gpu_hours": 9.401632336806747,
    "point_total_200_gpu_hours": 15.88338239892064,
    "additional_gpu_hours_for_200": 5.2156059407111215,
    "primary_fraction_of_remaining_when_planning": 1.0,
    "loading_reserve_gpu_hours": 0.05,
    "rate_gpu_hours_per_group": 0.06866927939298616,
    "point_rate_gpu_hours_per_group": 0.04700816168403373,
    "point_rate_bootstrap_ci": [
      0.03851331674435525,
      0.05722439949415513
    ],
    "throughput_margin": 0.2,
    "bootstrap_resamples": 10000,
    "val_complete_groups": 19,
    "val_expected_groups": 20,
    "val_partial_groups": {
      "val-b7bae149ac3ef95a25265e01d0c5fcae815f409b8a3126b6b4eb2fafa413099b": 10
    },
    "val_run": "runs/20261109T041333353840Z_482eec68a407_a050a8",
    "val_status": "failed",
    "val_gpu_hours": 1.5000085215144725,
    "new_main_matrix": {
      "kind": "E3",
      "lambdas": [
        1.0
      ],
      "epsilons": [
        0.01,
        0.1,
        1.0
      ],
      "designs": [
        "D0",
        "R4"
      ],
      "seeds": [
        11,
        22,
        33
      ],
      "steps": 200
    },
    "model_seconds_coefficients": {
      "fixed_per_group": 8.075777966856247,
      "D0_per_PGD_step": 0.47587887342173724,
      "R4_per_PGD_step": 0.4322793667412195
    },
    "model_residual_RMSE_seconds": 12.448884603485814,
    "forecast_note": "Val five-epsilon group durations fit by nonnegative fixed/D0-step/R4-step costs; predict only the three requested epsilons. Lock rate is 97.5% bootstrap upper mean plus 20% timing margin. All remaining budget is offered to primary E3 before secondary items. Forecast is not a runtime guarantee.",
    "secondary_protocol": {
      "groups": 100,
      "order": [
        "lambda=.1/10 at epsilon=.1",
        "minimum search at most 6 bisections",
        "E1/E2 at epsilon=.1"
      ],
      "executed": false
    },
    "cropped_settings": {
      "epsilons": [
        0.03,
        0.3
      ],
      "old_test60_plan": "superseded by user; never locked or executed"
    },
    "input_sha256": {
      "runs/20261109T041333353840Z_482eec68a407_a050a8/results.jsonl": "98c3ab71e2ea08aa128d51eb7c933959a637617d115b9e28793bce376cc16a32",
      "runs/20261109T041333353840Z_482eec68a407_a050a8/metrics.json": "96f2ce05f1b48461ad5036e829244cef1e7d10d20ab82f2d914e98e650ce7010",
      "reports/M8c_evidence/val_main_v2.log": "f3a0d9c5573ebe4f24833006f1801fb22ad4c9320990f72967fe1c2793981d80",
      "docs/M8c_TASK.md": "3f74582eeeda29fd6da585e08e1b03502ead709ff3095b09e61642c71bc1089e"
    }
  },
  "admission": {
    "message": {
      "mean": 0.93,
      "low": 0.895,
      "high": 0.965,
      "groups": 200
    },
    "none": {
      "mean": 0.27,
      "low": 0.21,
      "high": 0.33,
      "groups": 200
    },
    "mismatch": {
      "mean": 0.26,
      "low": 0.2,
      "high": 0.325,
      "groups": 200
    }
  },
  "QA": [
    {
      "design": "D0",
      "task": "A1",
      "split": "reader_val",
      "mean": 0.772777777777778,
      "low": 0.7536111111111111,
      "high": 0.7908333333333334,
      "groups": 200
    },
    {
      "design": "D0",
      "task": "A4",
      "split": "reader_val",
      "mean": 0.9566666666666667,
      "low": 0.9316666666666665,
      "high": 0.9766666666666667,
      "groups": 200
    },
    {
      "design": "R4",
      "task": "A1",
      "split": "reader_val",
      "mean": 0.9419444444444446,
      "low": 0.9297222222222223,
      "high": 0.9533333333333335,
      "groups": 200
    },
    {
      "design": "R4",
      "task": "A4",
      "split": "reader_val",
      "mean": 0.9983333333333334,
      "low": 0.995,
      "high": 1.0,
      "groups": 200
    }
  ],
  "QA_per_seed": [
    {
      "design": "D0",
      "seed": 11,
      "task": "A1",
      "accuracy": 0.7016666666666665
    },
    {
      "design": "D0",
      "seed": 11,
      "task": "A4",
      "accuracy": 0.96
    },
    {
      "design": "D0",
      "seed": 22,
      "task": "A1",
      "accuracy": 0.8308333333333334
    },
    {
      "design": "D0",
      "seed": 22,
      "task": "A4",
      "accuracy": 0.96
    },
    {
      "design": "D0",
      "seed": 33,
      "task": "A1",
      "accuracy": 0.7858333333333334
    },
    {
      "design": "D0",
      "seed": 33,
      "task": "A4",
      "accuracy": 0.95
    },
    {
      "design": "R4",
      "seed": 11,
      "task": "A1",
      "accuracy": 0.8583333333333334
    },
    {
      "design": "R4",
      "seed": 11,
      "task": "A4",
      "accuracy": 0.995
    },
    {
      "design": "R4",
      "seed": 22,
      "task": "A1",
      "accuracy": 1.0
    },
    {
      "design": "R4",
      "seed": 22,
      "task": "A4",
      "accuracy": 1.0
    },
    {
      "design": "R4",
      "seed": 33,
      "task": "A1",
      "accuracy": 0.9675
    },
    {
      "design": "R4",
      "seed": 33,
      "task": "A4",
      "accuracy": 1.0
    }
  ],
  "exposure": [
    {
      "seed": 11,
      "selected_exposure": 4368000,
      "total_exposure": 4592000,
      "best_epoch": 78,
      "checkpoint": "/home/me/PaperField/CalReader/runs/20261109T024558213482Z_1dae2969485e_450190/best.pt"
    },
    {
      "seed": 22,
      "selected_exposure": 4312000,
      "total_exposure": 4592000,
      "best_epoch": 77,
      "checkpoint": "/home/me/PaperField/CalReader/runs/20261109T025644521699Z_72b615e5e669_fceb9d/best.pt"
    },
    {
      "seed": 33,
      "selected_exposure": 4256000,
      "total_exposure": 4592000,
      "best_epoch": 76,
      "checkpoint": "/home/me/PaperField/CalReader/runs/20261109T025644494507Z_c0339ed44d8e_027434/best.pt"
    }
  ],
  "partial_val_E3": [
    {
      "design": "D0",
      "epsilon": 0.01,
      "split": "reader_val_partial",
      "mean": 0.9122807017543859,
      "low": 0.7894736842105262,
      "high": 1.0,
      "groups": 19
    },
    {
      "design": "R4",
      "epsilon": 0.01,
      "split": "reader_val_partial",
      "mean": 0.8771929824561402,
      "low": 0.7368421052631579,
      "high": 0.982456140350877,
      "groups": 19
    },
    {
      "design": "D0",
      "epsilon": 0.1,
      "split": "reader_val_partial",
      "mean": 0.9122807017543859,
      "low": 0.7894736842105263,
      "high": 1.0,
      "groups": 19
    },
    {
      "design": "R4",
      "epsilon": 0.1,
      "split": "reader_val_partial",
      "mean": 0.7192982456140351,
      "low": 0.5964912280701754,
      "high": 0.8245614035087719,
      "groups": 19
    },
    {
      "design": "D0",
      "epsilon": 1.0,
      "split": "reader_val_partial",
      "mean": 0.9298245614035087,
      "low": 0.8070175438596491,
      "high": 1.0,
      "groups": 19
    },
    {
      "design": "R4",
      "epsilon": 1.0,
      "split": "reader_val_partial",
      "mean": 0.6491228070175439,
      "low": 0.5263157894736841,
      "high": 0.7719298245614035,
      "groups": 19
    }
  ]
}
```
## 运行索引
- val主矩阵（failed，90分钟）：runs/20261109T041333353840Z_482eec68a407_a050a8
- 旧val敏感性（按用户修订中止）：runs/20261109T052533568845Z_f645a165c902_91ae80；原心跳metrics_before_user_interrupt.json、interrupt_reconciliation.json及cost_upper_additions.json保留。
- 此报告run：/home/me/PaperField/CalReader/runs/20261109T055408308916Z_b752762ef390_3f7ccf
