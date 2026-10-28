# M0b：第三方 Coconut checkpoint 核实（仅资料盘点）

核实日期：2026-10-28。按 M0.1 指令，Coconut 继续暂缓；本次没有下载、加载、使用任何权重，也没有训练。GPU 小时 = 0，源模型前向次数 = 0。下述准确率全部是上传者/论文作者声称，**不是 CLR 独立复现结果**，没有本项目置信区间；不能据此关闭 M0 的 Coconut 复现阻塞。

## 与原始 Coconut GPT-2 最直接相关的发布

| HF 仓库与固定 revision | 确切权重文件（仅核对目录元数据） | 来源及声称准确率 |
|---|---|---|
| [connordilgren/gpt2-gsm8k-coconut](https://huggingface.co/connordilgren/gpt2-gsm8k-coconut/tree/24f1422fcf6d557fd359fc818a4d1d0fde4aac7f)；`24f1422fcf6d557fd359fc818a4d1d0fde4aac7f` | `checkpoint_33`，原始 PyTorch state dict | Dilgren / Wiegreffe 的第三方复现；论文 Table 1 报告 GPT-2 Coconut **33.1%**，括号中的 **34.1% 是原 Coconut 论文数字**，不是该发布的结果。|
| [connordilgren/gpt2-prosqa-coconut](https://huggingface.co/connordilgren/gpt2-prosqa-coconut/tree/b8be0b21d5b4d8465ef305debdc9c4a4ac3e60a0)；`b8be0b21d5b4d8465ef305debdc9c4a4ac3e60a0` | `checkpoint_40`，原始 PyTorch state dict | 同一论文 Table 1 报告 ProsQA **98.0%**；括号中的 **97.0%** 是原论文数字。|
| [bmarti44/coconut-curriculum-checkpoints](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints/tree/4f9380d59f570b3dabce13ca7eed6a376e209c35)；`4f9380d59f570b3dabce13ca7eed6a376e209c35` | `coconut/checkpoint_best`、`coconut/checkpoint_49` 都是文件，不是 Transformers 目录；模型卡声称 best 对应 epoch 49 | 模型卡《The Curriculum Is the Mechanism》报告 ProsQA test **97.0%**。代码链接为 [research-pipeline](https://github.com/bmarti44/research-pipeline)，模型卡署名 anonymous/preprint；未独立验证身份或文件间等价性。|

前两项的论文是 [Are Latent Reasoning Models Easily Interpretable? v1](https://arxiv.org/html/2604.04902v1#S3.T1)。论文表头使用 GSM8k-Aug；[作者 README](https://github.com/connordilgren/are-lrms-easily-interpretable/blob/3c0c004e2cec09bdb26861c4fe5e39d2b0278cfe/README.md) 明确说明性能使用 `gsm_original_test.json`（原 GSM8K test），不是用于解释性分析的过滤测试集。论文与模型卡共同建立“作者报告→发布仓库”的关联；没有下载权重后验证这一关联。

作者训练配置已固定到 GitHub `3c0c004e2cec09bdb26861c4fe5e39d2b0278cfe`：

| 配置 | GSM8K Coconut | ProsQA Coconut |
|---|---|---|
| 基础模型 | `openai-community/gpt2` | 同左 |
| 数据 | `gsm_original_train.json` / `gsm_original_valid.json`（README称 GSM8k-Aug） | `prosqa_train.json` / `prosqa_valid.json` |
| latent curriculum | `c_thought=2`, `max_latent_stage=3`, 每阶段3 epochs | `c_thought=1`, `max_latent_stage=6`, 每阶段5 epochs |
| 初始化与 epoch 配置 | CoT `checkpoint_6`；`resume=3`, `num_epochs=47` | 无 CoT 初始化；`resume=0`, `num_epochs=50` |
| batch / 数值精度 | 每设备32，累积1；README假设4张A100 80GB；`bf16=false` | 同左 |
| 其余 | seed0，lr1e-4，weight_decay0.01，reset_optimizer=true，pad_latent_to_max=true | 同左 |

以上是发布配置字段原值；没有把 `resume` 和 `num_epochs` 推算成已经证实的实际训练步数。来源：[GSM8K YAML](https://github.com/connordilgren/are-lrms-easily-interpretable/blob/3c0c004e2cec09bdb26861c4fe5e39d2b0278cfe/args_coconut-cot-no_cot/gpt2_gsm8k_coconut.yaml)、[ProsQA YAML](https://github.com/connordilgren/are-lrms-easily-interpretable/blob/3c0c004e2cec09bdb26861c4fe5e39d2b0278cfe/args_coconut-cot-no_cot/gpt2_prosqa_coconut.yaml)。作者 README 指定原 Coconut 代码 commit `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`，并非原 Coconut 作者发布的权重。

bmarti44 模型卡记录 GPT-2 124M、ProsQA 17,886 训练样本、AdamW lr1e-4 / weight decay0.01、batch128、50 epochs、FP32、seed0、最多6 latent stages；声称 OOD 7-hop/8-hop/DAG/Dense 分别为66.0/67.5/59.2/61.2%。模型卡 loader 含 best→50→最高 epoch 的回退逻辑；若未来获准使用，必须固定具体文件并禁止继承这类回退。这里没有执行 loader。[固定版本模型卡](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints/blob/4f9380d59f570b3dabce13ca7eed6a376e209c35/README.md)

## 其他发布：变体或证据不完整

| 仓库 | 固定 revision | 训练/评测信息与限制 |
|---|---|---|
| [ModalityDance/latent-tts-coconut](https://huggingface.co/ModalityDance/latent-tts-coconut) | `89501ce8cc7cefd020b6e30c475a6772494de0a4` | GPT-2 自定义 Coconut 实现、6 latent tokens，配套 [LatentTTS](https://github.com/ModalityDance/LatentTTS/tree/23e756b6c27bf884aaead146f85b2cf0aecfc34d)。卡片没有给出该权重的数值准确率或完整源模型训练配方；不能从名称推断34.1%。仓库 `training_args/train_coconut.yaml` 用 `latent-data` 和 recall@1，不足以证明它是源 Coconut 的训练配方。|
| [connordilgren/gpt2-prosqa-multimode-coconut](https://huggingface.co/connordilgren/gpt2-prosqa-multimode-coconut) | `a30a6c05aeda5db0d27cc6ebd7ec1c627ea72ca8` | `checkpoint_49`；multi-mode 联合 latent/verbal/direct 训练变体，不能与上面普通 Coconut 合并；模型卡无数值准确率。|
| [ModalityDance/slpo-coconut-gpt2](https://huggingface.co/ModalityDance/slpo-coconut-gpt2) | `3355e8c89256d26638580ef7f097e171a4296516` | 从 latent-tts-coconut 经 stopping gate cold start 与 SLPO/RLOO 强化学习；最大12 latent，建议 gate0.6。卡片声称禁用 dropout 确定性推理下 GSM8K35.63%、GSM-Hard7.66%、MultiArith83.10%；不是原 Coconut baseline。完整优化器/训练步数未由模型卡充分披露。|
| [jihwan1205/svp-v-coconut-gpt2](https://huggingface.co/jihwan1205/svp-v-coconut-gpt2) | `c726251250aaff30ffbba7aa2e4dd9fb8724952e` | 从 latent-tts-coconut 经 SVP-V-GRPO；GSM8K-Aug383,891 eligible prompts，G32，B64筛到32，μ2，clip0.2，KLβ0.02，AdamW6e-5，seed0，15 epochs选epoch9。卡片声称 greedy、max_new_tokens64 的最终答案EM：GSM8K50.3%、GSM-Hard11.3%、MultiArith93.4%、SVAMP43.6%、ASDiv-A71.2%、GSM-Plus27.6%。强化学习衍生模型，不能用这些数字代表原 Coconut。|
| [internlm/SIM_COT-GPT2-Coconut](https://huggingface.co/internlm/SIM_COT-GPT2-Coconut) | `676c8a8aeb4afcef3a4fb98da9d539588d969340` | SIM-CoT 监督训练变体；模型卡准确率是 `xx.x` 占位符，不能作为该 checkpoint 的精度证据。|
| [danielpedrozo/gpt2-coconut-gsm-from-cot7](https://huggingface.co/danielpedrozo/gpt2-coconut-gsm-from-cot7) | `907843b485882a891919dd6daa95c8db2842da2d` | 模型卡是未填写模板；训练配置、数据、准确率均未核实。仓库名称不构成训练事实。|

这些检索结果没有进入 CLR 配置、数据生成或源模型选择，也不改变 CODI-GPT2 单源进入 M1 的审查条件。

## 可追溯证据与局限

`reports/evidence/M0b_coconut/` 保存 HF 官方 API 查询 JSON、各仓库元数据与固定 revision 的 README，以及作者 GitHub 固定 commit 的 README、tree、两个训练 YAML。`manifest.json` 对文本/JSON证据逐文件记录 SHA-256；只证明本地证据完整性，不是权重校验。HF 搜索最多返回100条，且本次只聚焦与任务相关的 GPT-2 发布，不声称穷尽 HF。未访问或选择 LLaMA 替代模型。

HF 插件搜索返回不可用；repo-details 能返回仓库概况但不含所需 README/revision，因此后续明确改用 HF 官方 HTTP API 与固定 revision 的文本端点。此工具限制没有引起模型或实验配置降级。第三方发布的训练来源真实性、checkpoint→论文指标的一一对应、潜步可导出性、协议确定性、泄漏情况与实际准确率均未通过本地实测。本次没有 GPU 实验，不适用三 seed 或准确率 bootstrap；不能把资料盘点标成复现验收通过。
