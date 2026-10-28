# 阻塞记录

尚未开始环境与资源验证，暂无已确认阻塞。后续记录问题、复现条件、影响与处理方案。

## 2026-10-28T09:05:30.252895+00:00 — 20261028T090527928097Z_00a778feec5f_5ca255
GatedRepoError: 401 Client Error. (Request ID: Root=1-6ae1bad9-35b9f373755e25c11ff4b8f4;dd7682b8-d564-4d70-938c-9d53864df630)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/9213176726f574b556790deb65791e0c5aa438b6/config.json.
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T090527928097Z_00a778feec5f_5ca255/traceback.txt`。未回退模型或设置。

## 2026-10-28T09:10:33.925519+00:00 — 20261028T091027316779Z_8c832f80400b_4e20b7
ValueError: Unexpected official dataset format
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T091027316779Z_8c832f80400b_4e20b7/traceback.txt`。未回退模型或设置。

## 2026-10-28T09:16:22.602189+00:00 — 20261028T091612704398Z_d57594d5c6b6_9fe4a3
OutOfMemoryError: CUDA out of memory. Tried to allocate 3.64 GiB. GPU 7 has a total capacity of 47.51 GiB of which 913.88 MiB is free. Process 4011893 has 1.52 GiB memory in use. Process 973519 has 7.41 GiB memory in use. Process 1442458 has 14.71 GiB memory in use. Process 1966113 has 22.95 GiB memory in use. Of the allocated memory 22.27 GiB is allocated by PyTorch, and 219.53 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T091612704398Z_d57594d5c6b6_9fe4a3/traceback.txt`。未回退模型或设置。

## Huginn 下载吞吐（2026-10-28）
原 HF HTTP 下载约每连接 0.5 MB/s，已中断并改用同一官方 revision 的并行 Range 下载，逐文件校验 LFS SHA-256。只改变传输实现；原失败记录与未完成缓存保留。

## 2026-10-28T09:18:27.434327+00:00 — 20261028T091823287597Z_cd55e448d4c6_ccda8e
OSError: Error no file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt.index or flax_model.msgpack found in directory /home/me/models/huggingface/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e.
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T091823287597Z_cd55e448d4c6_ccda8e/traceback.txt`。未回退模型或设置。

## 2026-10-28T09:21:45.119261+00:00 — 20261028T092142272750Z_d1dde0b72a3a_b43552
RuntimeError: GPU free 10.13 GiB < required 18 GiB
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T092142272750Z_d1dde0b72a3a_b43552/traceback.txt`。未回退模型或设置。

## 2026-10-28T09:36:09.276856+00:00 — 20261028T093608522525Z_97e4a63c882a_1c4f24
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/me/anaconda3/envs/latdefer/lib/python3.11/site-packages/PIL/../../.././libLerc.so.4)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T093608522525Z_97e4a63c882a_1c4f24/traceback.txt`。未回退模型或设置。

## 官方对照入口启动错误（已修复，2026-10-28）
`datasets` 在torch之前导入时加载了系统旧libstdc++，Pillow/libLerc报GLIBCXX_3.4.29缺失。调整导入顺序为torch在先，未改模型或依赖版本。初次失败发生于Run创建前，仅有 `reports/evidence/codi_reference_smoke.log`，无完整run元数据；这是本轮记录缺口，不作为实验结果。

## 2026-10-28T09:37:32.316053+00:00 — 20261028T093728211821Z_97e4a63c882a_c42675
NotImplementedError: Using RTX 4000 series doesn't support faster communication broadband via P2P or IB. Please set `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1" or use `accelerate launch` which will do this automatically.
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T093728211821Z_97e4a63c882a_c42675/traceback.txt`。未回退模型或设置。

## 2026-10-28T09:45:39.924329+00:00 — 20261028T094253670535Z_b6b4031958a8_9304d0
ChunkedEncodingError: ('Connection broken: IncompleteRead(6985878 bytes read, 60122986 more expected)', IncompleteRead(6985878 bytes read, 60122986 more expected))
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T094253670535Z_b6b4031958a8_9304d0/traceback.txt`。未回退模型或设置。

## M0 交付时未解决项（2026-10-28）

1. **CODI-LLaMA1B**：官方checkpoint名称和GSM8k-Aug来源已核实，但官方基础模型配置/分词器返回401。需要对 `meta-llama/Llama-3.2-1B-Instruct` 有效的本机访问授权。未获取替代模型、未猜测tokenizer或config。
2. **Coconut权重/训练**：官方GitHub releases及HF facebook/coconut检索为空，README只有recipe。batch32官方规模的成本探针OOM；明确batch8的真实ProsQA吞吐外推约百余GPU小时/seed，超过40小时。batch8探针只估成本，没有训练替代checkpoint。GSM8K-Aug成本需实际数据重测；用ProsQA长度代理的巨大估计不得视为精确预算。
3. **Huginn**：精确官方revision已锁定，接口在小型官方架构单元测试通过。真实运行在GPU7空闲10.13GiB时未满足配置18GiB的保守预检；这不是Huginn实际OOM的证据。之后GPU2曾有空闲，但权重获取仍未完成。96并发的同源Range下载最终 `ChunkedEncodingError/IncompleteRead`，运行 `20261028T094253670535Z_b6b4031958a8_9304d0` 已失败。保留已完成分块供恢复；没有加载不完整权重。实际checkpoint加载、官方lm-eval prompt/stop/strict-flexible提取对齐、复现与噪声曲线均未验收。
4. **源训练数据边界**：CLR的val/cal/test已按problem ID冻结且互斥，尚未独立核验源checkpoint训练数据是否包含这些验证问题。M0没有CLR训练集，也未提供conformal保证；M1须补齐train与来源问题去重。
5. **推理协议差异**：官方CODI batch128左padding结果43.594%，CLR等长batch≤16结果41.698%；保留二者。不能声称来源接口对官方批处理完全等价。噪声面板单次baseline39.0625%、K16 sigma0为37.5%，后者作为曲线内部基线。需要后续v2明确固定padding/position/mask与batch语义，禁止覆盖本轮数据。
6. **统计解释**：三seed为同checkpoint重复推理；greedy CI=0不表示测试总体置信区间为零，也不是独立训练seed。64题val噪声面板的均值下降不证明所有任务/源上噪声无效。
7. **日志/成本缺口**：最早官方对照入口的动态库导入失败发生在Run创建前，仅有外层stderr；batch32成本探针OOM未在finally落盘完整耗时/forward数。相关成本未知，不能写为零。之后正式CODI实验和官方对照完整记录GPU wall小时与forward。
8. **M0状态**：仅CODI-GPT2子结果完成；M0整体未完成、未整体验收。M1前仍需处理以上阻塞；不能用未测源的N/A当作负准确率结果。

已修复的启动问题：官方数据分隔符误读（修正前未生成划分）、datasets/torch导入顺序冲突、RTX的NCCL P2P/IB配置。失败记录保留，不覆盖。

## 2026-10-28T11:37:47.054738+00:00 — 20261028T113746024609Z_6b6596ff9777_336d96
ValueError: Unexpected training corpus format
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T113746024609Z_6b6596ff9777_336d96/traceback.txt`。未回退模型或设置。

## 2026-10-28T11:49:09.632677+00:00 — 20261028T114858268738Z_327fccf26f2d_b9a29c
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:3! (when checking argument for argument other in method wrapper_CUDA__equal)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T114858268738Z_327fccf26f2d_b9a29c/traceback.txt`。未回退模型或设置。

## M0.1 v2（2026-10-28，待审查，非完成标记）

- **生成协议一致率门槛未通过**：官方bf16 batch128参考复放1318/1319（99.924%）；batch=1、fp32生产候选为1257/1319（95.299%），准确率43.745%。两者不是同一个通过项。不能自行把用户要求的≥99%改成仅要求参考复放。已提交具体报告供用户选择是否保留此门槛或明确接受精度协议差异。完整噪声试验受此门槛阻止；M1不启动。
- **batch不变性失败**：真实bf16 batch1/4/16存在不同答案。采用用户预先允许的batch1/fp32候选并记录差异，但该候选不是自动验收。最终完整batch对比见 `reports/M0.1_protocol.md`。
- **残差定位**：官方batch128复放仅剩索引1306（官方7、CLR8）不同；逐参数相等、prefill相等，首次投影后输入差0.25。测试no_grad与inference_mode没有消除此差异，不能把这个假说写成结论。
- **G-σ未完成**：300题面板已冻结，18设置×K16的1题smoke完成（183.706秒，1个seed），尚无300题×3seed全量曲线，无sigma推荐。当前吞吐线性外推约45.9 GPU wall小时；不能当作已发生成本或精确预算。
- **去重假设**：全发布GSM8k-Aug（385620条）比实际经过长度/格式过滤的源训练数据更宽；任意13-gram重合即移除，可能把共享模板也排除。归一化精确匹配0；原val/cal/test分别移除3/4/7题。用户允许的原500池重分配已冻结val300/cal193/test1312，不使用test补充val。
- **已修复格式失败**：首次去重假设训练JSON是记录列表，实际为等长question/cot/answer三列；失败run保留，修正后完成，未换数据。
- **诊断入口失败**：残差追踪初版SyntaxError发生于Run创建之前，日志 `reports/evidence_v2/residual.log`，无完整成本信息。下一次官方TrainingArguments重置到cuda:0而CLR在cuda:3，参数比较报跨设备错误（自动run记录保留）；之后明确恢复配置device，实际追踪成功。未把失败结果当实验。
- **M0b**：Huginn仍使用官方HF恢复下载（max_workers=4），网络read timeout由HF继续续传；只已完成文件通过官方LFS SHA后才可验收，真实权重加载/复现尚未完成。CODI-LLaMA1B等待用户HF授权，不寻找替代。Coconut暂停训练与使用；第三方检索仅报告，见 `reports/M0b_coconut_third_party.md`。

## 2026-10-28T12:01:32.180691+00:00 — 20261028T120131639183Z_f055d9571168_dba8d7
RuntimeError: Source protocol agreement gate has not passed; full noise study blocked
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T120131639183Z_f055d9571168_dba8d7/traceback.txt`。未回退模型或设置。

## 2026-10-28T12:18:16.820375+00:00 — 20261028T121812416441Z_1b90f6844702_7ba16c
RuntimeError: Insufficient free VRAM: 6.81 GiB, required 14
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T121812416441Z_1b90f6844702_7ba16c/traceback.txt`。未回退模型或设置。

## 2026-10-28T12:24:03.082498+00:00 — 20261028T122358879576Z_1b90f6844702_d86203
RuntimeError: Insufficient free VRAM: 7.34 GiB, required 14
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T122358879576Z_1b90f6844702_d86203/traceback.txt`。未回退模型或设置。

## 2026-10-28T12:26:01.349832+00:00 — 20261028T122556758574Z_ea9f53c485ac_bb82a7
NotImplementedError: Using RTX 4000 series doesn't support faster communication broadband via P2P or IB. Please set `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1" or use `accelerate launch` which will do this automatically.
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T122556758574Z_ea9f53c485ac_bb82a7/traceback.txt`。未回退模型或设置。

## 2026-10-28T12:41:47.989067+00:00 — 20261028T124143649075Z_6dd3a5e7057c_d4515c
RuntimeError: Insufficient free VRAM: 6.55 GiB, required 18
完整 traceback: `/home/me/PaperField/CalReader/runs/20261028T124143649075Z_6dd3a5e7057c_d4515c/traceback.txt`。未回退模型或设置。

## 2026-10-28 — 用户审查解除 CODI 完整噪声启动阻塞
用户明确答复：“接受 bf16 复放验收，继续 fp32 完整噪声实验”。据此只解除完整噪声的启动门槛，保持真实fp32官方一致率95.299%的记录；不把原指标改写为通过≥99%，不视为G-σ或M1获准。固定val300、3seed、K16和全部18设置不变。授权写入 `configs/protocol_codi_v2.yaml` 的review字段；因协议文件hash更新，先重跑同设置smoke再启动完整实验。

Huginn最新独立阻塞：官方下载已完成且4/4 LFS SHA通过，真实bf16/r32加载成功；重复输入smoke发生真实CUDA OOM，完整复现与噪声未启动。运行 `20261028T122800941040Z_0be72c22d642_1ce21f` 原始OOM traceback保留；自动记录又遇到默认编码ASCII的次生错误，不能只引用metrics中的编码异常掩盖OOM。详见 `reports/M0b_huginn.md` 与 `reports/M0b_huginn_blockers.json`。后续预检要求稳定18GiB，未降低模型、精度或递归深度。
