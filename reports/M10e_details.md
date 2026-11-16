# M10e 详细证据



# M10e 外部基线（2026-11-12）

用户授权仅新增：HSRM-style 两源三seed Q1/Q2，CLUE 两源确定性 Q1/Q2，LLaMA M3 B2 三seed Q1–Q4；旧冻结只读。HSRM先读2608.30841第3节，CLUE读2510.01591原文。
输入仅实际观测前缀；HSRM用Q1独立轨迹Bernoulli监督，Q2复用标量但cal_A独立温度。CLUE原始末层delta=z_t-z_0，reader_train成功/失败质心，cal_A分别logistic校准。B2完全沿用M3并适配2048维。
只用reader_val选超参数，所有模型/配置hash统一锁定后才读test；smoke≤5%/15min仅train/val代理。指标test/clean/seen/各t等权/三seed均值；并列reader_val且Q1分clean/traj。
用户后续修订：预算自行确定并可自动扩充；论文不直接修改，新建一个Markdown交接文档，写明表格行、setup两句、number_audit来源及编译检查，供另一个对话完成。
审查reports/M10e_REVIEW.md≤2000字节、固定(a)–(f)。完成后再次冻结data/M10e_v1；不启动其他实验。

预设：HSRM使用原文2层/256宽/4头/FF1024/dropout0.1、masked mean→LN→scalar，无额外位置编码；按用户要求用BCE替代原文组内排序。LR={0.0001,0.0003}，8epoch，reader_val NLL逐epoch选优。B2 LR={0.001,0.0003}/8epoch/WD0.0001/batch4096同M3。seed11/22/33。
CLUE按原文完整经验轨迹构建一对质心：reader_train每个clean/扰动轨迹的终点delta仅计一次；评估各t使用可观测prefix delta。只用冻结末层而非原文全层，按用户输入约束明确作为适配。



## 运行协议与审计


原文与SHA：data/M10e_v1/literature/sources.json。HSRM第3节及附录C、CLUE第3节及附录A已读；无源前向。

HSRM两源原始末层观测prefix输入，2层256宽/4头/FF1024/dropout0.1；有效位置均值后LN标量。BCE监督Q1独立Bernoulli；Q2复用同一标量，Q1/Q2分别在cal_A温度缩放。未输入源问题、标签或split标识。

CLUE每条reader_train完整clean/扰动轨迹终点只计一次；每类算术平均raw delta。评分=到失败质心欧氏距离−到成功质心距离；各t只用当前可观测delta=z_t−z_0。cal_A对Q1/Q2独立无正则logistic，数值标准化仅帮助优化，与原始距离差的仿射模型等价。单次确定性，不以重复运行伪造seed CI。

B2延用M3输入/模板/数值选项/训练归一化/优化器/2LR/8epoch/每qtype选择；只扩2048维，Q3原始sqrt(768)缩放保留。旧GPT2 B2仅从冻结M3 statistics取clean/seen/三seed/各t等权，legacy_gpt2_B2.json保存来源hash。

HSRM完整搜索2LR×8epoch×3seed，各epoch全量reader_val NLL选优；后期val NLL上升如实保存在history.json，不改变预设训练网格。全部训练结束后才统一training_lock，cal_A随后拟合并保存，test_access.json记录最终访问锁。

训练启动快照由各run实际保存config/git/environment；data/M10e_v1/code_before为开发阶段保全，非每run源码快照。训练期间仅补充冻结与测试入口的配置/依赖hash完整性校验，未改训练计算；当前代码SHA在测试前锁定。

smoke每项20训练/20val题；GPT2比例0.100%/1.00%，LLaMA 0.293%/2.00%，均≤5%且≤15min。具体时间/数据审计见smoke_*.json对应run和cost_*.json。

统计：test/clean/seen，各t内micro、t及固定seed等权。95%CI为10,000次题目cluster bootstrap，题目重采样在t/seed间共享；条件于固定训练模型，不覆盖重新训练seed的不确定性。reader_val Q1 clean/traj及Q2–Q4分列summary.csv，所有逐t/seed指标在最终run statistics.json。

CPU汇总v1 run 20261112T072248432462Z_189260e52f8b_40c2ab 因B2 reader_val Q4 online float32与导出float64 accuracy差2.38e-5触发审计中止。近0.5处float32 softmax造成额外argmax平局。原脚本、失败日志、预测、温度和在线statistics全部保留；m10e_summarize_v2分别重放原生精度并核对原statistics，再以既有scripts/m9_export.py的float64 serialized-logit口径独立核对最终汇总，没有放宽一致性阈值、重训、重校准或重推断。新增近零logit回归测试，合计5项通过。精度差逐项见aggregation_checks.json。

成本0.942317202071 GPUh，RTX4090，共享GPU。包含GPU任务完整wall（加载、smoke、训练、校准和最终预测），保守高于实际GPU kernel占用。CLUE CPU时间单列cost_ledger.json；源前向0。

旧输入前后4865项hash均一致；核验范围是相关冻结dump/QA和各freeze清单原字节，另核验复用的M3统计文件。不宣称消除历史冻结日志漂移。

用户后续明确改为交付Markdown而非修改论文；故未改/编译manuscripts，未声称新版8页已验收。reports/M10e_PAPER_HANDOFF.md含完整三行LaTeX、两句setup、原值/CI/来源、文献及编译要求。


## 主表及区间


| source | model | split | qtype | kind | metric | mean | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | HSRM | reader_val | Q1 | clean | nll | 0.402773516 | 0.389786775 | 0.416140589 |
| gpt2 | HSRM | reader_val | Q1 | clean | accuracy | 0.829142196 | 0.818250061 | 0.839912117 |
| gpt2 | HSRM | reader_val | Q1 | clean | brier | 0.126374677 | 0.121104421 | 0.131776789 |
| gpt2 | HSRM | reader_val | Q1 | traj | nll | 0.41834338 | 0.404778661 | 0.432142554 |
| gpt2 | HSRM | reader_val | Q1 | traj | accuracy | 0.816032243 | 0.804793375 | 0.82714211 |
| gpt2 | HSRM | reader_val | Q1 | traj | brier | 0.132834061 | 0.127298111 | 0.13842691 |
| gpt2 | HSRM | reader_val | Q2 | clean | nll | 0.458082341 | 0.445782558 | 0.470569937 |
| gpt2 | HSRM | reader_val | Q2 | clean | accuracy | 0.772285714 | 0.761738095 | 0.782905357 |
| gpt2 | HSRM | reader_val | Q2 | clean | brier | 0.14968677 | 0.144619464 | 0.154820293 |
| gpt2 | HSRM | test | Q1 | clean | nll | 0.558777822 | 0.53768978 | 0.579799428 |
| gpt2 | HSRM | test | Q1 | clean | accuracy | 0.720149254 | 0.701195881 | 0.739259802 |
| gpt2 | HSRM | test | Q1 | clean | brier | 0.188416909 | 0.179705924 | 0.197194005 |
| gpt2 | HSRM | test | Q2 | clean | nll | 0.581768313 | 0.563163479 | 0.600367551 |
| gpt2 | HSRM | test | Q2 | clean | accuracy | 0.694100924 | 0.675527126 | 0.712959014 |
| gpt2 | HSRM | test | Q2 | clean | brier | 0.198765299 | 0.190788689 | 0.206746715 |
| gpt2 | HSRM | reader_val | Q2 | clean | auroc | 0.84421095 | 0.831413553 | 0.856435862 |
| gpt2 | HSRM | test | Q2 | clean | auroc | 0.834324409 | 0.816724832 | 0.85227497 |
| llama | HSRM | reader_val | Q1 | clean | nll | 0.380802166 | 0.370002467 | 0.392039175 |
| llama | HSRM | reader_val | Q1 | clean | accuracy | 0.933082022 | 0.919893562 | 0.945203681 |
| llama | HSRM | reader_val | Q1 | clean | brier | 0.104926671 | 0.0999987038 | 0.110068708 |
| llama | HSRM | reader_val | Q1 | traj | nll | 0.379071136 | 0.36725643 | 0.391634049 |
| llama | HSRM | reader_val | Q1 | traj | accuracy | 0.925306017 | 0.910718437 | 0.938747569 |
| llama | HSRM | reader_val | Q1 | traj | brier | 0.104493799 | 0.0990915881 | 0.110273386 |
| llama | HSRM | reader_val | Q2 | clean | nll | 0.478021799 | 0.46910065 | 0.487168263 |
| llama | HSRM | reader_val | Q2 | clean | accuracy | 0.881190476 | 0.866284524 | 0.895428571 |
| llama | HSRM | reader_val | Q2 | clean | brier | 0.146849376 | 0.142601496 | 0.151205022 |
| llama | HSRM | test | Q1 | clean | nll | 0.624759219 | 0.605133261 | 0.644317614 |
| llama | HSRM | test | Q1 | clean | accuracy | 0.61746624 | 0.591767132 | 0.643574168 |
| llama | HSRM | test | Q1 | clean | brier | 0.218423619 | 0.209273366 | 0.227507922 |
| llama | HSRM | test | Q2 | clean | nll | 0.653255195 | 0.639742934 | 0.666817723 |
| llama | HSRM | test | Q2 | clean | accuracy | 0.570244018 | 0.545319829 | 0.595120824 |
| llama | HSRM | test | Q2 | clean | brier | 0.231123791 | 0.224603764 | 0.237634871 |
| llama | HSRM | reader_val | Q2 | clean | auroc | 0.832531349 | 0.799262059 | 0.861545893 |
| llama | HSRM | test | Q2 | clean | auroc | 0.834295261 | 0.814161027 | 0.853242278 |
| gpt2 | CLUE | reader_val | Q1 | clean | nll | 0.851370315 | 0.842551899 | 0.859967715 |
| gpt2 | CLUE | reader_val | Q1 | clean | accuracy | 0.247659828 | 0.231680292 | 0.264125402 |
| gpt2 | CLUE | reader_val | Q1 | clean | brier | 0.326954593 | 0.322671628 | 0.331118795 |
| gpt2 | CLUE | reader_val | Q1 | traj | nll | 0.831477213 | 0.82149137 | 0.841232851 |
| gpt2 | CLUE | reader_val | Q1 | traj | accuracy | 0.293788459 | 0.276605859 | 0.311442894 |
| gpt2 | CLUE | reader_val | Q1 | traj | brier | 0.317109054 | 0.312272512 | 0.321826793 |
| gpt2 | CLUE | reader_val | Q2 | clean | nll | 0.881648326 | 0.870146875 | 0.892974501 |
| gpt2 | CLUE | reader_val | Q2 | clean | accuracy | 0.317142857 | 0.302214286 | 0.332357143 |
| gpt2 | CLUE | reader_val | Q2 | clean | brier | 0.338837778 | 0.333394659 | 0.344176934 |
| gpt2 | CLUE | test | Q1 | clean | nll | 0.648457871 | 0.634354247 | 0.662721285 |
| gpt2 | CLUE | test | Q1 | clean | accuracy | 0.640245203 | 0.614631752 | 0.665538602 |
| gpt2 | CLUE | test | Q1 | clean | brier | 0.228120414 | 0.221276182 | 0.235055033 |
| gpt2 | CLUE | test | Q2 | clean | nll | 0.62438363 | 0.606148809 | 0.642752972 |
| gpt2 | CLUE | test | Q2 | clean | accuracy | 0.675195451 | 0.651741294 | 0.698081023 |
| gpt2 | CLUE | test | Q2 | clean | brier | 0.216683654 | 0.208119494 | 0.225345962 |
| gpt2 | CLUE | reader_val | Q2 | clean | auroc | 0.61168941 | 0.594820616 | 0.628622336 |
| gpt2 | CLUE | test | Q2 | clean | auroc | 0.575343994 | 0.552145368 | 0.599366656 |
| llama | CLUE | reader_val | Q1 | clean | nll | 0.465799225 | 0.444258727 | 0.488333499 |
| llama | CLUE | reader_val | Q1 | clean | accuracy | 0.80610198 | 0.78925094 | 0.822689661 |
| llama | CLUE | reader_val | Q1 | clean | brier | 0.153796503 | 0.14548559 | 0.16242134 |
| llama | CLUE | reader_val | Q1 | traj | nll | 0.43122533 | 0.406210979 | 0.457468869 |
| llama | CLUE | reader_val | Q1 | traj | accuracy | 0.78946875 | 0.770625276 | 0.80781636 |
| llama | CLUE | reader_val | Q1 | traj | brier | 0.139361415 | 0.129622094 | 0.149655979 |
| llama | CLUE | reader_val | Q2 | clean | nll | 0.54478264 | 0.52086775 | 0.569938031 |
| llama | CLUE | reader_val | Q2 | clean | accuracy | 0.683285714 | 0.668282143 | 0.698285714 |
| llama | CLUE | reader_val | Q2 | clean | brier | 0.184860599 | 0.176095704 | 0.193914297 |
| llama | CLUE | test | Q1 | clean | nll | 0.544663269 | 0.523876149 | 0.566058055 |
| llama | CLUE | test | Q1 | clean | accuracy | 0.705170576 | 0.688983209 | 0.72122468 |
| llama | CLUE | test | Q1 | clean | brier | 0.184557198 | 0.176466258 | 0.192744475 |
| llama | CLUE | test | Q2 | clean | nll | 0.52480995 | 0.50383588 | 0.546402151 |
| llama | CLUE | test | Q2 | clean | accuracy | 0.738592751 | 0.723525231 | 0.753375977 |
| llama | CLUE | test | Q2 | clean | brier | 0.175651841 | 0.16772948 | 0.183883187 |
| llama | CLUE | reader_val | Q2 | clean | auroc | 0.765285222 | 0.732104638 | 0.79444001 |
| llama | CLUE | test | Q2 | clean | auroc | 0.792658825 | 0.776386134 | 0.808108114 |
| llama | B2 | reader_val | Q1 | clean | nll | 0.36774111 | 0.356074901 | 0.380304552 |
| llama | B2 | reader_val | Q1 | clean | accuracy | 0.929601496 | 0.916127627 | 0.942361472 |
| llama | B2 | reader_val | Q1 | clean | brier | 0.100121549 | 0.0948329523 | 0.105810291 |
| llama | B2 | reader_val | Q1 | traj | nll | 0.350943548 | 0.337819335 | 0.365283719 |
| llama | B2 | reader_val | Q1 | traj | accuracy | 0.925012635 | 0.910094888 | 0.938792108 |
| llama | B2 | reader_val | Q1 | traj | brier | 0.0941531116 | 0.0882669741 | 0.100610505 |
| llama | B2 | reader_val | Q2 | clean | nll | 0.461453547 | 0.450333373 | 0.472856464 |
| llama | B2 | reader_val | Q2 | clean | accuracy | 0.853666667 | 0.839285714 | 0.867714286 |
| llama | B2 | reader_val | Q2 | clean | brier | 0.142037851 | 0.136938299 | 0.147261134 |
| llama | B2 | reader_val | Q3 | clean | nll | 0.979422109 | 0.951276615 | 1.00750355 |
| llama | B2 | reader_val | Q3 | clean | accuracy | 0.566571429 | 0.549238095 | 0.584238095 |
| llama | B2 | reader_val | Q3 | clean | brier | 0.544745251 | 0.530085251 | 0.559404327 |
| llama | B2 | reader_val | Q4 | clean | nll | 0.693156082 | 0.693120951 | 0.693191155 |
| llama | B2 | reader_val | Q4 | clean | accuracy | 0.499427999 | 0.49201523 | 0.506792507 |
| llama | B2 | reader_val | Q4 | clean | brier | 0.250004451 | 0.249986885 | 0.250021987 |
| llama | B2 | test | Q1 | clean | nll | 0.614866818 | 0.593459009 | 0.6358585 |
| llama | B2 | test | Q1 | clean | accuracy | 0.637834636 | 0.612369403 | 0.663237459 |
| llama | B2 | test | Q1 | clean | brier | 0.214065731 | 0.204229218 | 0.223649331 |
| llama | B2 | test | Q2 | clean | nll | 0.62926315 | 0.614308211 | 0.644106995 |
| llama | B2 | test | Q2 | clean | accuracy | 0.644491827 | 0.624496565 | 0.664723999 |
| llama | B2 | test | Q2 | clean | brier | 0.219283942 | 0.212424879 | 0.225962061 |
| llama | B2 | test | Q3 | clean | nll | 1.01907773 | 0.988193531 | 1.04989188 |
| llama | B2 | test | Q3 | clean | accuracy | 0.540298507 | 0.523004027 | 0.557972045 |
| llama | B2 | test | Q3 | clean | brier | 0.565116479 | 0.54895966 | 0.580863892 |
| llama | B2 | test | Q4 | clean | nll | 0.693149018 | 0.693115788 | 0.693181948 |
| llama | B2 | test | Q4 | clean | accuracy | 0.501440092 | 0.494335637 | 0.508424539 |
| llama | B2 | test | Q4 | clean | brier | 0.250000919 | 0.249984304 | 0.250017384 |
| llama | B2 | reader_val | Q2 | clean | auroc | 0.783623692 | 0.752794911 | 0.813438206 |
| llama | B2 | test | Q2 | clean | auroc | 0.76397426 | 0.744227088 | 0.783579494 |


## 训练/评估索引


- gpt2_HSRM: training `/home/me/PaperField/CalReader/runs/20261112T064306345959Z_4bcf3841b35e_88e027`; final `/home/me/PaperField/CalReader/runs/20261112T072105541188Z_c424abe93f1c_855eba`

```json
[
  {
    "qtype": "Q1",
    "seed": 11,
    "lr": 0.0001,
    "epoch": 1,
    "val_nll": 0.4113233983516693,
    "train_rows": 740228,
    "train_seconds": 42.706544294022024,
    "parameters": 1777153
  },
  {
    "qtype": "Q1",
    "seed": 22,
    "lr": 0.0001,
    "epoch": 1,
    "val_nll": 0.41703465580940247,
    "train_rows": 740228,
    "train_seconds": 43.47761186293792,
    "parameters": 1777153
  },
  {
    "qtype": "Q1",
    "seed": 33,
    "lr": 0.0001,
    "epoch": 1,
    "val_nll": 0.3965524733066559,
    "train_rows": 740228,
    "train_seconds": 43.155819035018794,
    "parameters": 1777153
  }
]

```

- llama_HSRM: training `/home/me/PaperField/CalReader/runs/20261112T064307543891Z_687e6047be1e_0a60c2`; final `/home/me/PaperField/CalReader/runs/20261112T072128848127Z_25c1ef10e404_c4b772`

```json
[
  {
    "qtype": "Q1",
    "seed": 11,
    "lr": 0.0001,
    "epoch": 1,
    "val_nll": 0.2527472674846649,
    "train_rows": 252399,
    "train_seconds": 14.800284065073356,
    "parameters": 2104833
  },
  {
    "qtype": "Q1",
    "seed": 22,
    "lr": 0.0003,
    "epoch": 1,
    "val_nll": 0.26353225111961365,
    "train_rows": 252399,
    "train_seconds": 14.543806664994918,
    "parameters": 2104833
  },
  {
    "qtype": "Q1",
    "seed": 33,
    "lr": 0.0001,
    "epoch": 1,
    "val_nll": 0.2580872178077698,
    "train_rows": 252399,
    "train_seconds": 14.724744061939418,
    "parameters": 2104833
  }
]

```

- gpt2_CLUE: training `/home/me/PaperField/CalReader/runs/20261112T064309946881Z_b6ec3d307d3d_b13766`; final `/home/me/PaperField/CalReader/runs/20261112T072148985727Z_973d83d558f5_63a60c`

- llama_CLUE: training `/home/me/PaperField/CalReader/runs/20261112T064311096388Z_1da1c20014b5_2ef680`; final `/home/me/PaperField/CalReader/runs/20261112T072201154413Z_6ef05ca41d6f_82a22f`

- llama_B2: training `/home/me/PaperField/CalReader/runs/20261112T064308529098Z_5292adbd6387_617dde`; final `/home/me/PaperField/CalReader/runs/20261112T072214078101Z_293d13eb2f34_80aa4f`

```json
[
  {
    "qtype": "Q1",
    "seed": 11,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.23874305188655853,
    "train_rows": 252399,
    "train_seconds": 0.589145238045603,
    "parameters": 2123
  },
  {
    "qtype": "Q1",
    "seed": 22,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.2436419576406479,
    "train_rows": 252399,
    "train_seconds": 0.5667166319908574,
    "parameters": 2123
  },
  {
    "qtype": "Q1",
    "seed": 33,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.25129207968711853,
    "train_rows": 252399,
    "train_seconds": 0.560250220936723,
    "parameters": 2123
  },
  {
    "qtype": "Q2",
    "seed": 11,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.35222041606903076,
    "train_rows": 47719,
    "train_seconds": 0.1173309920122847,
    "parameters": 2123
  },
  {
    "qtype": "Q2",
    "seed": 22,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.35817354917526245,
    "train_rows": 47719,
    "train_seconds": 0.11359719093888998,
    "parameters": 2123
  },
  {
    "qtype": "Q2",
    "seed": 33,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.36669597029685974,
    "train_rows": 47719,
    "train_seconds": 0.11773040902335197,
    "parameters": 2123
  },
  {
    "qtype": "Q3",
    "seed": 11,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.9781721234321594,
    "train_rows": 293301,
    "train_seconds": 0.5222448679851368,
    "parameters": 18507
  },
  {
    "qtype": "Q3",
    "seed": 22,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.9645346999168396,
    "train_rows": 293301,
    "train_seconds": 0.7598338429816067,
    "parameters": 18507
  },
  {
    "qtype": "Q3",
    "seed": 33,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.9642353653907776,
    "train_rows": 293301,
    "train_seconds": 0.627293610945344,
    "parameters": 18507
  },
  {
    "qtype": "Q4",
    "seed": 11,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.6996030807495117,
    "train_rows": 95228,
    "train_seconds": 0.24321209592744708,
    "parameters": 2123
  },
  {
    "qtype": "Q4",
    "seed": 22,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.6990970373153687,
    "train_rows": 95228,
    "train_seconds": 0.26744526100810617,
    "parameters": 2123
  },
  {
    "qtype": "Q4",
    "seed": 33,
    "lr": 0.001,
    "epoch": 8,
    "val_nll": 0.6960568428039551,
    "train_rows": 95228,
    "train_seconds": 0.28835349599830806,
    "parameters": 2123
  }
]

```
