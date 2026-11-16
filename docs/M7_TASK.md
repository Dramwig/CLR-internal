# M7：潜通信管线与语义审计（非自适应）

用户授权：2026-11-06，北京时间。M7 与 M6 独立，允许不同 GPU 并行；不训练攻击，M8 未授权。
优先级 A → B1+B4 → C → D synthetic v2 → E 主比较 → 可选项。预算含 smoke/失败/重启 ≤40 GPUh。

A：独立 `.venv-comm`，transformers 4.57.1 及相关依赖与 Idea A 一致；不改主环境、Idea A、冻结产物。
生成器直接 import `/home/me/PaperField/LatDefer`，每次启动核验 synthetic_v2.py/synthetic.py/schema.py 冻结 SHA。
B：Qwen3-4B（Idea A 的 qwen4 固定 revision）同权重 sender/receiver 分时调用；sender 读 x_pub+x_priv，receiver 只读 x_pub+KV。sender 固定格式短 commitment，不输入真实标签。
独立核验固定 episode 的历史消息/输出及数值误差；有/无/错配消息任务准确率。历史未保存 raw KV 时不得把新复放冒充历史张量逐项比对。
学习式消息仅 Idea A 完整 val 通过才接入；C2C 可选，不阻塞。
C：z=实际消息 KV；等间隔层，逐 token 各头 K/V 拼接，fp16；manifest 含层、token范围、mask、门控、post-RoPE。smoke 后估算，>15GiB 仅减少层数/episode数，调整须明示。
D：目标 reader_train≥8000、reader_val1000、cal1000（A/B各半）、test2000，按反事实 group ID 划分。先只做 synthetic v2；HiddenBench 核实可得性/格式。
clean=真消息+匹配 commitment；mismatch=异 episode 消息+原 commitment；counterfactual=同组只改单个私有事实重算KV+原commitment；corrupted=截断/层置零/范数约束噪声。
审计 A1事实 yes-no（正负取值边际对齐）、A2结论 yes-no、A3消息与 commitment 一致 yes-no、A4私有变量四选一；每类≥6改写，2留出。标签来自生成器；reader/基线不看x_priv或x_pub原文，允许commitment。
E：CLR-lite复用M4.1架构新增KV适配器；B0、池化KV×选项MLP B3、线性探针、层范数检测、仅commitment+receiver输出的文本审计。3 seed；可选M6初始化不依赖。
一致性=随机k∈{1,4,16}问答与可见commitment/已知约束一致的平均log概率；cal_A温度校准，cal_B clean独立消息定α∈{.01,.05} conformal阈值。不得把经验FPR写成无条件≤α保证。
报告A1–A4问答指标，按消息类别检测率@FPR1%/5%和AUROC，counterfactual为主结果；并列reader_val/test；3seed与按group bootstrap10000次95%CI。

smoke ≤5%数据且≤15分钟，只用val代理；训练smoke仅train，最终test前冻结所有配置/checkpoint/阈值。失败直接记录，不换模型/设置。
交付唯一 `reports/M7_REVIEW.md`（≤8000字节、(a)–(f)、≤3主表、3位有效数字），详细证据索引到run/CSV，冻结完成产物，状态持久化到docs/M7_LIVE.json。
最后回答：(1)独立管线验收及私有信息因果证据；(2)counterfactual CLR/范数/文本/B3检测率@5%；(3)A1/A4相对B3是否与GSM8K Q3/Q4一致。

## 2026-11-06 用户明确修订：commitment 主协议
所有 clean/mismatch/counterfactual/corrupted 都用生成器真实事实/结论的程序化 commitment，同一套≥3表述变体随机使用；manifest写 `commitment_source=programmatic`。格式、字段顺序和措辞与 sender 输出的格式化接口一致，不用文本风格泄漏类别。counterfactual仍保留原episode承诺。
模型承诺仅诊断：reader_val/test每个基础episode记录sender direct与cot512输出/正误。训练与阈值锁定后，额外natural_commitment分析clean消息配模型承诺，按模型结论对/错分组报告CLR A3一致性分数分布；不进主训练、指标、调参或阈值。
文本基线仍读承诺+receiver输出；CF检测只能依靠receiver输出变化。BLOCKERS须记direct33.8%（固定val80）/cot51250.0%（val8 smoke）及程序化承诺与部署模型承诺的差距，两个数样本量不同。
