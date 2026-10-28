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
