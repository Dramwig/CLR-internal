# M9 详细证据

完成待审查；唯一审查交付为reports/M9_REVIEW.md。

## 协议与预算

累计GPUh=7.38169113611；源前向0。模型为原M6 MultiLite，两源适配器/source token与共享主体；每源完整冻结训练集，逆(source,qtype)频率损失，按全reader_val平均NLL选checkpoint，min_epochs=8。

旧控制器只在等待子进程时停止；seed11原训练保持。未重启训练、未改变batch/精度/曝光。原控制器保留running历史快照，执行补充说明其已停止；它没有GPU计费。

沿用的M6实现仅从epoch>=8的候选保存best checkpoint；epoch1–7的val损失只作曲线诊断，不能事后选更早epoch。联合模型的训练规模/加权损失与单源不同，不能据差异作纯机制因果归因。

## 模型与曝光（保存的实际记录）

### multi_lite_11

run：/home/me/PaperField/CalReader/runs/20261110T035645180399Z_8587a7c4ac1a_9e3b58

checkpoint SHA：fb7fee5764cd7029c4dfe7bb0c7380e759a39a28a49e8cb36dc373408fcf836d

{
  "best": {
    "epoch": 8,
    "val_nll": {
      "gpt2:Q1": 0.5194767904089753,
      "gpt2:Q2": 0.5461948405674526,
      "gpt2:Q3": 0.37035797126644016,
      "gpt2:Q4": 0.6140732040317906,
      "llama:Q1": 0.4733306989936321,
      "llama:Q2": 0.4652299107142857,
      "llama:Q3": 0.7838716022848149,
      "llama:Q4": 0.9381200964147741
    },
    "selection_nll": 0.5888318893352706,
    "gpu_hours": 2.10541769560918
  },
  "metrics": {
    "status": "completed",
    "run_id": "20261110T035645180399Z_8587a7c4ac1a_9e3b58",
    "config_sha256": "8587a7c4ac1ae3d425204f4b5ded6c14d1715fdfc53f946b5b94779730356266",
    "gpu_hours": 2.3572813193244575,
    "source_forward_calls": 0,
    "best_val_nll": 0.5888318893352706,
    "test_opened": false,
    "examples": {
      "gpt2": {
        "Q1": 6662052,
        "Q2": 1260000,
        "Q3": 7742052,
        "Q4": 2514960
      },
      "llama": {
        "Q1": 2271591,
        "Q2": 429471,
        "Q3": 2639709,
        "Q4": 857052
      }
    },
    "optimizer_steps": 23832,
    "trainable_parameters": 19345667,
    "training_seconds": 7655.582091073971,
    "validation_seconds": 649.0427387282252,
    "validation_examples": 2730042
  }
}

### multi_lite_22

run：/home/me/PaperField/CalReader/runs/20261110T052536832497Z_f2d8f6011c79_51cf47

checkpoint SHA：5b55e6660da6fded590ec3503c99c139f49b2f76a53cbab63bba010f0284450c

{
  "best": {
    "epoch": 8,
    "val_nll": {
      "gpt2:Q1": 0.4695876452374982,
      "gpt2:Q2": 0.5250856563023159,
      "gpt2:Q3": 0.3626225326376566,
      "gpt2:Q4": 0.586983066147117,
      "llama:Q1": 0.4153394570714482,
      "llama:Q2": 0.4146642107282366,
      "llama:Q3": 0.747741704170376,
      "llama:Q4": 0.8824520713863976
    },
    "selection_nll": 0.5505595429601308,
    "gpu_hours": 2.0266517695402984
  },
  "metrics": {
    "status": "completed",
    "run_id": "20261110T052536832497Z_f2d8f6011c79_51cf47",
    "config_sha256": "f2d8f6011c79cbc5cc95f26e048055eebb04b5b243f499df251e20c3f9380de2",
    "gpu_hours": 2.270156481185857,
    "source_forward_calls": 0,
    "best_val_nll": 0.5505595429601308,
    "test_opened": false,
    "examples": {
      "gpt2": {
        "Q1": 6662052,
        "Q2": 1260000,
        "Q3": 7742052,
        "Q4": 2514960
      },
      "llama": {
        "Q1": 2271591,
        "Q2": 429471,
        "Q3": 2639709,
        "Q4": 857052
      }
    },
    "optimizer_steps": 23832,
    "trainable_parameters": 19345667,
    "training_seconds": 7335.42949673289,
    "validation_seconds": 645.2910019550472,
    "validation_examples": 2730042
  }
}

### multi_lite_33

run：/home/me/PaperField/CalReader/runs/20261110T052537082287Z_e28aff3f6df6_2d0975

checkpoint SHA：18a3645848203db835b893a429b3ca792d178c0177ce99117d6f8072f225c7c0

{
  "best": {
    "epoch": 9,
    "val_nll": {
      "gpt2:Q1": 0.5104125691626834,
      "gpt2:Q2": 0.5745721697126116,
      "gpt2:Q3": 0.3772167408314803,
      "gpt2:Q4": 0.6592991679028047,
      "llama:Q1": 0.4110128886351376,
      "llama:Q2": 0.4693818141392299,
      "llama:Q3": 0.7816744689389654,
      "llama:Q4": 0.923852205651522
    },
    "selection_nll": 0.5884277531218044,
    "gpu_hours": 2.36635520418443
  },
  "metrics": {
    "status": "completed",
    "run_id": "20261110T052537082287Z_e28aff3f6df6_2d0975",
    "config_sha256": "e28aff3f6df66b254bcaa60184d4e17c2e66924d225c89953e3284badd73106a",
    "gpu_hours": 2.622924629112499,
    "source_forward_calls": 0,
    "best_val_nll": 0.5884277531218044,
    "test_opened": false,
    "examples": {
      "gpt2": {
        "Q1": 7402280,
        "Q2": 1400000,
        "Q3": 8602280,
        "Q4": 2794400
      },
      "llama": {
        "Q1": 2523990,
        "Q2": 477190,
        "Q3": 2933010,
        "Q4": 952280
      }
    },
    "optimizer_steps": 26480,
    "trainable_parameters": 19345667,
    "training_seconds": 8490.70857353299,
    "validation_seconds": 766.5462680589408,
    "validation_examples": 3033380
  }
}

## 完整统计与图表

analysis_A/metrics.csv含Q1 reader_val clean/traj/all及所有类型；multisource_paired.csv是joint−single配对统计；T1补Q2 AUROC的题bootstrap；T2的部署集合覆盖率与size配对另列。

所有脚本仅从paper_inputs冻结CSV读数；其provenance.json给出原始CSV/JSON/已保存预测SHA。表图文件与数值派生CSV逐字节重建。PDF固定日期、字体和元数据；没有光栅数据图层。

T2的size配对不能宣称同覆盖率；GSM8K匹配覆盖率描述来自M5b原冻结统计，使用评估标签，不是部署阈值。

F1左图沿用原seed-t CI，右两图为固定三seed的题bootstrap；B5b仅N100，缺预算不画假点。F2搜索预算包络由val全网格得出，只作描述，非独立选型/测试证据。

F3包络取已执行较小半径的最高均值；浅色为原始点，不重新运行攻击。weak D0为M8a四任务分数，R1–R4为M8b统一A1/A4，不能把D0与其微小差异作同口径改进。

T5从关闭的里程碑账本与历史保存记录导出，复用模型不重复计原训练；legacy行只给可追溯下界。物理batch forward、搜索per-lane等价计算和梯度调用不是同一单位。

## 新增工程修复

M0.1确定/随机CI字段分别解析；旧预测从冻结QA按problem/t/traj/template/replicate补回选项，重建conformal逐项校验。首次未通过导出不产生模型调用。协议元数据v1保留但不用于论文；v2选择实际噪声协议。

## 历史限制

各前序review已记录的测试暴露、synthetic programmatic commitment、弱receiver、适配器目标混杂、偏移可交换性与有限PGD限制全部保留。没有通过新划分、选seed或新的攻击实验改善结果。

## 机器主张索引

paper/claims.json与reports/FINAL_SUMMARY.md按四贡献/两个负面发现组织，每条都含证据及边界；不能支持的措辞独立列出。
