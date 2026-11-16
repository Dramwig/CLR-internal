# M5a 工程预检与预算冲突（未完成）

本文件不报告 val300 搜索准确率，不回答两项研究问题；唯一审查文件为 `reports/M5a_REVIEW.md`。
完整任务、输入/标签/成本口径见 `docs/M5a_TASK.md`。没有打开test数据、生成test预测或查看本次test指标。

## 已完成

实现S0/S1/S2/S3，N4/8/16、beam2/4、sigma1.5、全步/早期范围、B4/B3(Q1)/oracle K16。
复用冻结M3三seed checkpoint及normalization；B3只接收模板0与训练时的dummy选项`['0']`数值特征。
先扰动反馈、源模型前向，再读真实观测；每条分支保留自身KV。只从冻结val manifest读题，入口allowlist在读文件前拒绝其他数据路径。
单位测试：9 passed in 6.36s。覆盖共享策略续算、观测/反馈区分、K16终点退化、合法答案投票与并列、Q1特征、预算规则及val路径隔离。

## 完整网格预算

正确特征/UUID绑定/批量iid采样版本固定5题×单seed完整网格，每题61个配置，均值50.182457秒/题/seed（不含一次模型加载与核对），逐题时间[63.41138466700795, 51.9666182709916, 43.138505482987966, 45.2113159270084, 47.18446142799803]。
300题×3seed直接外推12.545614GPUh；加20%裕量、0.05GPUh公共开销和已有预检保守计费0.353686GPUh，总估算15.458423GPUh，上限10。
因此完整预算关卡未通过，没有启动完整val300。固定小面板计时不能代表严谨吞吐置信区间；没有以此伪造三seed研究结果。

预算备选已形成配置`configs/M5a_oracle_full_proposal_v12.yaml`并只做smoke：B4/B3和S1/S2全部原设置不变；仅S3 oracle固定N16、b4且只做全步范围。
其固定5题×单seed均值21.595840秒/题/seed，逐题时间[26.528567323024618, 21.799848752998514, 19.22443239801214, 20.332258502021432, 20.09409436400165]；直接外推5.398960GPUh，含同样裕量/已有成本约6.882438GPUh。
备选未获授权应用于完整实验；没有擅自删减原任务。另一选择是将原网格预算明确修订至16GPUh。

## 优化、失败和限制

- 全程源权重fp32、temperature0、同一cohort左padding；没有TF32/降精度、模型替换或N/b/K回退。
- 显式LoRA合并推理核对：16匹配轨迹答案一致，状态最大绝对差0.000175476。不是逐位等价，更不是全val300数值验收；未来完整运行须保留该限制。
- 编译版数值核对通过（max0.000253677），但计算阶段76.713444秒/题/seed，未见吞吐收益；编译开销计入成本，未选择它跑全量。编译缓存触顶/grad模式重编译警告保留。
- 批量iid协议直接调用同一个`clr.policy.RelativeNoise`处理整批z，仍为每lane独立相对高斯。改变seed寻址和物理batch，不能声称重现原M1逐trajectory随机流。单元测试核对带噪一步输出，实际smoke零噪声续算与原policy一致。
- 原逐lane随机流版本与M1的16轨迹状态逐元素一致。所有工程版本都有启动时源码、配置、环境、git/权重hash快照；不是事后补造。
- Q1 dummy特征最初误用全零；已修复并有与冻结训练接口逐元素相等的测试。早期smoke全部作工程记录，不作科研结果；修正版独立smoke通过。
- 一次实际OOM、若干显存预检查失败及一次入口局部import错误均记录traceback。首轮exec返回143，原因未观测，末尾成本缺失；保守计费0.080GPUh，最终源计数只能给下界。
- 早期`merged128_v2`的合并核对发生在lane计数器安装前；其GPU墙钟/物理API调用包含核对，但lane计数不含核对。总lane计数仅下界，不补造最终计数。
- 共享GPU序号有歧义，最终用GPU UUID显式绑定。保留全部资源快照；没有终止其他项目作业。
- 后续预检门槛显式从14GiB调为11GiB，依据已测8.55GiB峰值；设置CUDA allocator expandable_segments以控制缓存碎片。硬件参数调整不缩小源模型/batch/N/b/K。备选删减的oracle网格另外明确标注，未用于全量。

## 待执行的统计与比较

源前向主轴含prefill、潜步骤、答案解码；probe FLOPs换算由smoke的`compute_profile.json`给出；oracle内部所有K16单列且计入主轴。
独立长S1池按每题累计前向成本匹配搜索；同一预算内取最长前缀，残余不足一条轨迹的预算须报告。4096轨迹上限不够时标缺，不声称匹配。
固定预算点128..32768和全部原始配置点均须保留；不按准确率挑N/b/seed。还需生成accuracy–compute曲线以及按题配对10000次bootstrap CI。
需同时说明终点oracle是使用gold的特权0/1选择；中间K16有MC误差和自适应选择偏差，不能称严格数学上界。
比例分母的oracle增益没有可靠正证据时，不能报告“取得上限百分比”。本预检没有足以支持任何收益/比例的结果。

## 证据

- 正确完整网格smoke：`/home/me/PaperField/CalReader/runs/20261105T031751654418Z_212fc9c51870_18a67f`。
- 备选smoke：`/home/me/PaperField/CalReader/runs/20261105T032311389452Z_6ad9e6171e14_54d560`。
- 完整逐配置成本：`reports/M5a_preflight_costs.csv`；不包含accuracy列，避免将1题smoke误作val300结论。
- 预算和成本：`data/M5a_preflight_v0/budget.json`、`data/M5a_preflight_v0/cost_ledger.json`。
- 运行目录都有config/git/environment/source_snapshot/input_sha256/metrics；中断版本缺失项已标识。
