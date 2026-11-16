# M10e 外部基线（2026-11-12）

用户授权仅新增：HSRM-style 两源三seed Q1/Q2，CLUE 两源确定性 Q1/Q2，LLaMA M3 B2 三seed Q1–Q4；旧冻结只读。HSRM先读2608.30841第3节，CLUE读2510.01591原文。
输入仅实际观测前缀；HSRM用Q1独立轨迹Bernoulli监督，Q2复用标量但cal_A独立温度。CLUE原始末层delta=z_t-z_0，reader_train成功/失败质心，cal_A分别logistic校准。B2完全沿用M3并适配2048维。
只用reader_val选超参数，所有模型/配置hash统一锁定后才读test；smoke≤5%/15min仅train/val代理。指标test/clean/seen/各t等权/三seed均值；并列reader_val且Q1分clean/traj。
用户后续修订：预算自行确定并可自动扩充；论文不直接修改，新建一个Markdown交接文档，写明表格行、setup两句、number_audit来源及编译检查，供另一个对话完成。
审查reports/M10e_REVIEW.md≤2000字节、固定(a)–(f)。完成后再次冻结data/M10e_v1；不启动其他实验。

预设：HSRM使用原文2层/256宽/4头/FF1024/dropout0.1、masked mean→LN→scalar，无额外位置编码；按用户要求用BCE替代原文组内排序。LR={0.0001,0.0003}，8epoch，reader_val NLL逐epoch选优。B2 LR={0.001,0.0003}/8epoch/WD0.0001/batch4096同M3。seed11/22/33。
CLUE按原文完整经验轨迹构建一对质心：reader_train每个clean/扰动轨迹的终点delta仅计一次；评估各t使用可观测prefix delta。只用冻结末层而非原文全层，按用户输入约束明确作为适配。
