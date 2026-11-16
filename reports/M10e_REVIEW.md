### (a) 状态
M10e三项实验完成待审查，再次冻结；论文修改按用户指令交接。
### (b) 关卡
通过：5项val smoke、5项单测、18权重+2质心测试前hash锁、10,000次按题bootstrap、4865项旧科学输入SHA一致。新增0.942GPUh、源前向0。
未执行：论文改稿/编译/8页验收，已按用户修订移交。
### (c) 主表
|源/方法|val Q1 clean/traj|test Q1 NLL|Q2 AUC|Q3/Q4 acc|
|---|---|---|---|---|
|gpt2/HSRM|0.403/0.418|0.559|0.834|—|
|llama/HSRM|0.381/0.379|0.625|0.834|—|
|gpt2/CLUE|0.851/0.831|0.648|0.575|—|
|llama/CLUE|0.466/0.431|0.545|0.793|—|
|llama/B2|0.368/0.351|0.615|0.764|0.540/0.501|
CI及reader_val Q2/Q3/Q4见summary.csv；CLUE确定性单次，其余三seed。
### (d) 新增偏离与阻塞
HSRM用BCE、CLUE用末层prefix delta，未冒称原文完整复现。CPU汇总曾因近0.5舍入差中止；v2按既有表1 float64导出并双精度重放通过，原预测/温度不改。
### (e) 待决定
无。
### (f) 证据
reports/M10e_PAPER_HANDOFF.md；reports/M10e_details.md；data/M10e_v1/{summary.csv,training_lock.json,freeze.json}；final_*.json索引真实runs及预测。
