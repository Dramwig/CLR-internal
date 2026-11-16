# M10d：补齐 tab:main 四格（2026-11-12）

用户授权：旧冻结结果/模型/阈值只读；GPT2 B6 Q1 NLL/Q2 AUROC先查M10现有评估，已有即复用。
LLaMA B4先查M6/M9/M10；缺失才按M3标量z_t→Q1轨迹Bernoulli协议补训，Q2复用标量、单独cal_A温度；6817题M6 reader_train/reader_val，LR/epoch搜索同M3，seed11/22/33，全部训练锁hash后才test。
新增总成本≤2 GPUh，不调用源模型；smoke≤5%/15min，仅训练/val代理。
四格统一test/clean/已见模板/各t等权/三seed均值；仅替换论文TODO和必要最优加粗，不改措辞；number_audit.md追加来源，编译确认8页。
论文位于用户本机 C:\Users\zixi-\Desktop\Paper\CVPR27-CLR；服务器当前无该目录，须核实访问能力，不能声称已编辑/编译。
唯一审查reports/M10d_REVIEW.md≤2000字节：四数、来源、复用/新跑、成本。
