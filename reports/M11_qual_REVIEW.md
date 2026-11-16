### (a) 状态一句话
M11-qual导出完成待审查；无训练/重校准/PGD，旧冻结输入SHA一致。

### (b) 关卡清单
通过：val smoke 53.8秒，规模≤5%；12项单测；两源各3题共6题冻结概率逐位一致；必需概率/集合、NPZ形状与有限值检查。
GPU保守累计0.0849h（含失败/重复导出），RTX4090；源模型前向0。

### (c) 文件与规模
- `export/gpt2/qual_reading.json`、`llama/qual_reading.json`：各12题，t0–6。
- `embed_reasoning.npz`：300题×4步；含兄弟部分，另存`embed_siblings.npz`，40题×2步×17候选，三seed状态/值均保留。
- `embed_audit.npz`：600组×2消息，四层特征/A4表示与三seed概率/Γ/报警。
- `dependent_scores.json`：100组、600条ε=.1已有E3记录。
- `frozen_predictions.json`、selection/checks/provenance及逐文件README；数据160MB，ZIP见同目录。

### (d) 新增偏离与阻塞
M8c未保存扰动张量及同一攻击的另一审计器分数，按授权导出现有分数，其余显式null。附加LLaMA线性基线无冻结conformal阈值；必需方法齐全。M5b原排序logits保留，另用冻结温度导出概率。
两次val smoke格式失败已修复并计费；原日志保留。

### (e) 需审查者决定
无。

### (f) 证据路径
`data/M11_qual_v1/export/{selection,checks,provenance}.json`；`costs.json`、`inventory.json`、`reproduce/`；`reports/M11_qual_tests.log`。
选择：题目仅按gold可解析且中间量≥3过滤；其余按指定总体；统一sha256(ID+"qual")取前N，不看结果。Q4固定template0，负例补足供体见selection；Q3原模板/选项/冻结cal_B α=.1。
