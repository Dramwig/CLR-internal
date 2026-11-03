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

## 2026-10-29 — LLaMA底模访问阻塞已通过指定镜像解决
根据用户指示，从ModelScope `LLM-Research/Llama-3.2-1B-Instruct` revision `d3e551343d4d81508a0d226656b826c217e463cd` 下载9文件（2.48GB），本地SHA和上游Git/LFS指针对象全部一致；离线格式验证通过。没有从HF下载底模，也没有更换模型。详见 `reports/M0b_llama_modelscope_v2.md`。

独立剩余项：CODI作者训练后checkpoint不在本地，ModelScope尚未找到已核实镜像。用户“不用HF下载”的范围正在澄清，未擅自从HF取作者权重，亦未以底模冒充CODI。原v1授权失败记录仅为历史，不再代表底模资源现状。

## 2026-10-29 — 作者checkpoint渠道已获明确授权
用户回复“从什么平台下载都可以只要能下载”。来源范围阻塞解除；现在从作者公开HF仓库下载锁定revision `b2c88ba224b06b12b52ef39b87f794b98a6eb1c8` 的CODI-LLaMA checkpoint，官方大小3,408,265,790字节、LFS SHA-256 `c6d4b4b096f03fafca18409a4a7535ac9539f558ed741b31878fecadb40ed23b`。底模仍是已校验的ModelScope本地版本，未改模型。下载/校验与GPU smoke须实际完成后才宣告相应阻塞解除；GPT2长实验不停止、不改配置。

## 2026-10-29T01:28:32.505068+00:00 — 20261029T012805976560Z_080db608cb52_3cd001
RuntimeError: Error(s) in loading state_dict for CODI:
	Unexpected key(s) in state_dict: "dynamic_cls.1.bias", "dynamic_cls.1.weight", "dynamic_cls.3.bias", "dynamic_cls.3.weight". 
完整 traceback: `/home/me/PaperField/CalReader/runs/20261029T012805976560Z_080db608cb52_3cd001/traceback.txt`。未回退模型或设置。

## CODI-LLaMA v2流水线阻塞
Traceback (most recent call last):
  File "/home/me/PaperField/CalReader/scripts/m0b_codi_llama_job.py", line 22, in <module>
    s=step('source_smoke','scripts/m01_protocol_eval.py',c['source_smoke'])
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/me/PaperField/CalReader/scripts/m0b_codi_llama_job.py", line 13, in step
    with log.open('x',encoding='utf-8') as f:subprocess.run([sys.executable,script,config,*overrides],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/me/anaconda3/envs/latdefer/lib/python3.11/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/me/PaperField/CalReader/.venv/bin/python', 'scripts/m01_protocol_eval.py', 'configs/M0b_codi_llama_smoke_v2.yaml']' returned non-zero exit status 1.

未降级模型、batch或精度。阶段日志见 reports/evidence/M0b_llama_modelscope/repro。

## 2026-10-29 — CODI-LLaMA作者checkpoint下载/同一性阻塞解除
run `20261029T011806667943Z_8b873a7eb8a6_7735de` 完成3.408GB作者权重下载，SHA与官方LFS完全一致。底模ModelScope+作者checkpoint均已就绪，GPU5上的分阶段验证/复现流程已实际启动。当前不能把来源问题解除等同于模型smoke、复现或噪声验收完成；状态见 `reports/evidence/M0b_llama_modelscope/repro_job.json`。

## 2026-10-29 — CODI-LLaMA已定位的checkpoint结构兼容问题
首次真实source smoke失败：官方checkpoint比当前官方模型类多出 `dynamic_cls.1.{weight,bias}` 与 `dynamic_cls.3.{weight,bias}` 四个键，strict加载正确拒绝；失败run及`repro_job.json`保留。官方仓库当前源码无dynamic_cls引用，test.py原本strict=False会忽略它们。

v2新建 `clr/sources/codi_llama_v2.py`：只有这四个精确键可作为不参与forward的buffer保留，所有缺失键或其他额外键均拒绝，最终load_state_dict仍strict=True。不猜测该头激活/功能、不使用它、不修改checkpoint，也不更换模型。官方对照封装亦按同一方法保留这四键；真正的test.py前向与解码不变。此修复须经新smoke和官方逐题对照验证，不把单元测试当模型复现。

## 2026-10-29T01:34:21.948145+00:00 — 20261029T013420939329Z_361949d688ff_79c947
RuntimeError: GPU free 7.72 GiB < required 8 GiB
完整 traceback: `/home/me/PaperField/CalReader/runs/20261029T013420939329Z_361949d688ff_79c947/traceback.txt`。未回退模型或设置。

## CODI-LLaMA v2流水线阻塞
Traceback (most recent call last):
  File "/home/me/PaperField/CalReader/scripts/m0b_codi_llama_job.py", line 23, in <module>
    o=step('official_smoke','scripts/m0b_codi_reference_v2.py',c['official_smoke'])
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/me/PaperField/CalReader/scripts/m0b_codi_llama_job.py", line 13, in step
    with log.open('x',encoding='utf-8') as f:subprocess.run([sys.executable,script,config,*overrides],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/me/anaconda3/envs/latdefer/lib/python3.11/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/me/PaperField/CalReader/.venv/bin/python', 'scripts/m0b_codi_reference_v2.py', 'configs/M0b_codi_llama_official_smoke_v2.yaml']' returned non-zero exit status 1.

未降级模型、batch或精度。阶段日志见 reports/evidence/M0b_llama_modelscope/repro_retry1。

## 2026-10-29 — CODI-LLaMA实际source smoke通过，官方入口显存预检失败
新loader的GPU5 smoke已通过：run `20261029T013327437248Z_080db608cb52_ccd688`，8题、44.655秒、159次forward，真实batch1/4/8在这8题答案相同（不足以宣称全量batch不变）。随后官方smoke进入模型前检测到GPU5仅7.72GiB空闲，低于8GiB阈值，停止并保留失败，未加载模型/产生预测。

CUDA新探针GPU4空闲17.56GiB。新建显式gpu4配置，预检提高到14GiB，重跑匹配smoke；只改变设备调度和更保守的内存门槛，模型/精度/batch/latent全部保持，原GPU5配置和结果保留。新流水线状态 `reports/evidence/M0b_llama_modelscope/repro_gpu4_job.json`。

## 2026-10-29 — CODI-LLaMA完成复现后的未解决项
真实官方test.py全1319题为55.497%（按题10k bootstrap 95% CI 52.843–58.150%），CLR batch16为55.269%；同batch128对照为55.421%。逐题不一致由27题降到14题，仍不能宣称完全等价，数值执行路径残差尚未定位。
与固定论文v3 Table A4 β=20参考51.9%相差+3.597pp；clean test1312仍55.335%，排除7题重叠不能解释全部偏差。底模镜像同一性、官方recipe和完整checkpoint校验均通过；发布checkpoint到论文实验条目的训练manifest/代码版本对应关系仍不明。没有为追平改指标、换划分或挑seed。分析与逐题差异位置见 `reports/M0b_codi_llama_analysis_v2.md`。LLaMA噪声曲线尚未执行，M0b整体未验收；不阻塞既定GPT2单源主线。

## 临时资源避让：物理 GPU 2（记录于 2026-10-29T16:10:48.755456+08:00）

收到用户经另一任务转达的资源限制：至北京时间 2026-10-29 18:06，物理 GPU 2 禁止本项目使用，涵盖设备重映射；已写入 AGENTS.md。核查当前噪声进程容器 PID 216252 / 宿主 PID 3253918，nvidia-smi 对应物理 GPU 3（UUID GPU-ef23fa9f-5ab6-08fc-1fca-aef1b2b787cc）；其环境无 CUDA_VISIBLE_DEVICES 重映射，协议指定 cuda:3。未发现运行中的本项目 Huginn / CODI-LLaMA 评测；未停止或修改其他用户进程。现有 GPT2 任务继续，无协议变更。


## M0.1 完整噪声运行结束
运行 `/home/me/PaperField/CalReader/runs/20261028T131534016226Z_83bfd1aab8dd_d4e55b` 完成；G-σ结果见完整报告，M0.1审查及M1仍未授权。推荐输出：{"selected": {"sigma": 1.5, "from_step": 3, "temperature": 0, "accuracy_drop": 0.04944444444444446, "nondegenerate": 0.3288888888888889}, "tie_break": "nondegenerate descending, drop ascending, sigma ascending, from_step ascending"}

## 2026-10-30 — 当前状态（补充历史阻塞）

- M0.1第1–4项实验/工程交付及收尾核验完成；完整噪声71文件哈希一致，23测试通过、1跳过。旧记录中“等待完整噪声”已解除。G-σ按均值通过，sigma1.5/from_step3下降4.944pp、非退化32.889%；仅距5pp门槛0.056pp，统计稳健性未得到证明。
- bf16复放验收与fp32生产协议差异已获用户明确接受；仍保留fp3295.299%一致率、跨batch差异、单题投影数值dispatch未完全归因的限制，不改写原指标。
- M1未完成/未启动：用户明确选择“我提供M1任务书”，等待具体交付及规模；不能用自行缩小的范围宣称完成。
- M0b未整体完成：Huginn实际smoke OOM、CODI-LLaMA14题同batch残差及论文偏差/噪声曲线仍待处理；Coconut继续暂缓。这些与GPT2单源M1分开记录。
- 磁盘空间不足已临时解除：按用户授权 `reserve.sh release 10` 释放10GiB，可用约12.65GiB，自动填充保持暂停；M1容量仍待任务书后估算。未删除任何实验或权重。
- 原提案路径 `docs/IdeaB_Calibrated_Latent_Reader.md` 不存在；本次通读项目指定的 `docs/proposal.md`，其中没有M1编号任务书。

详见 `reports/M0.1_closure_v2.md`。

## 2026-10-30T01:18:56.175118+00:00 — 20261030T011828343325Z_cb2983bfb3ec_8032be
FileNotFoundError: [Errno 2] No such file or directory: 'data/M1_v3/candidates.json.tmp'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261030T011828343325Z_cb2983bfb3ec_8032be/traceback.txt`。未回退模型或设置。

## 2026-10-30T01:20:46.022550+00:00 — 20261030T012044635539Z_49d2f4edb21f_0553af
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

完整 traceback: `/home/me/PaperField/CalReader/runs/20261030T012044635539Z_49d2f4edb21f_0553af/traceback.txt`。未回退模型或设置。

## 2026-10-30 — M1 任务书已收到；当前约束

M1任务书缺失的旧阻塞已解除。M0.1审查通过；CODI-LLaMA复现按现状获接受、噪声延至M6前；Huginn按用户指令暂停，Coconut继续暂缓。

用户明确批准保留原cohort左padding。同题K路复制原batch1 KV及上下文，不新增padding；manifest保存cohort_id、left_padding、pad_to、协议hash。**已知限制：z_t依赖cohort组成/padding宽度，不需要消除。**

候选准备首次目录不存在错误已修复并保留失败run。GPU4 smoke在CUDA初始化时OOM，实际源forward=0；已显式选择GPU2重跑，未改变模型/精度/数据。GPU2临时避让窗口已过期。

M1全量仍受200题smoke、val300批处理和续算关卡以及60GPU小时/8GiB预算约束。fp16是有损存储；将单独比较原始fp32和fp16往返状态在原KV上的续算，不将磁盘特征冒充无损源状态。尚无完整标签统计，不宣称M1完成。

## 2026-10-30T01:36:27.101482+00:00 — 20261030T012126501342Z_c7c7a651e47e_6eef7a
TimeoutError: M1 200-question smoke exceeded 900 seconds; full run prohibited
完整 traceback: `/home/me/PaperField/CalReader/runs/20261030T012126501342Z_c7c7a651e47e_6eef7a/traceback.txt`。未回退模型或设置。

## M1 smoke 超时与显式修正（2026-10-30）

首轮GPU2 smoke run `20261030T012126501342Z_c7c7a651e47e_6eef7a` 在900秒完成161/200题，failed，23655次forward、0.2500001GPU小时；未计作通过。按正式split分工修正：同200题clean+7个早退，100train做K8轨迹，100val做两种K16单步MC；完整阶段仍做所有t，配置未降精度/模型。修正版run `20261030T013639323603Z_13ad784ef5d6_518bf2`正在运行。预算按阶段实际题数估计，包含失败尝试成本。详细状态见 `reports/M1_data.md` 与 `reports/evidence_M1/full_job.json`。

## M1 smoke阻塞已解除（2026-10-30）

修正版200题smoke在898.671秒完成，run `20261030T013639323603Z_13ad784ef5d6_518bf2`；此前900秒超时保留为失败。当前阻塞仅为正在执行的val300批处理/续算关卡及之后的最终预算。预计完整20000训练题生成38.408GPU小时（25%余量、另计关卡），6.816GiB（2倍余量），不是已消耗或已完成。原test1312的历史cohort会在v3冻结时恢复，284题候选宽度修正，cal/test尚未推理。

## 2026-10-30T02:25:19.458017+00:00 — 20261030T022518234836Z_39c11741d8fb_f9026e
RuntimeError: Insufficient disk for measured budget plus 2GiB reserve
完整 traceback: `/home/me/PaperField/CalReader/runs/20261030T022518234836Z_39c11741d8fb_f9026e/traceback.txt`。未回退模型或设置。

## M1 pipeline — generation_failed

Runs: {"smoke_run": "runs/20261030T013639323603Z_13ad784ef5d6_518bf2", "gate_run": "runs/20261030T015147548907Z_f9aeb43dc9b2_daa0ae", "budget_run": "runs/20261030T022507631329Z_ae2889e005b8_1fe957", "generation_run": "runs/20261030T022518234836Z_39c11741d8fb_f9026e"}. Evidence: `reports/M1_results.md`, `reports/evidence_M1/full_job.json`. No failed gate was bypassed.

## 2026-11-01 — M1 状态核查与恢复

val300验收已通过，run `20261030T015147548907Z_f9aeb43dc9b2_daa0ae`：早退T一致率100%，原fp32与fp16往返状态的零噪声批处理续算均无答案不一致；三seed噪声比较通过。v3已冻结：reader_train20000 / reader_val2000 / val300 / cal500 / test1005，预算39.462GPU小时（含预检）、6.816GiB，没有缩减规模。

此前全量run `20261030T022518234836Z_39c11741d8fb_f9026e` 在磁盘预检退出（0次源forward、0GPU小时），不是持续运行或数据生成完成。2026-11-01复查磁盘配重自动填充已恢复开启（谁恢复未核实），当前free18.628GiB。按用户已有授权release10GiB并持久暂停自动填充，恢复空间约28.6GiB；GPU2实测空闲14.35GiB。

原输出目录只有空目录，使用rmdir移除（非空会拒绝），未删除任何数据。旧失败状态和log复制为 `reports/evidence_M1/full_job_disk_failure_20261030.*`。使用相同已冻结配置显式重启，tmux `clr-m1-full-restart`，日志 `reports/evidence_M1/full_job_restart_20261101.log`。M1仍未完成，成功与否以新run/metrics.json为准。

确认重启已进入真实源模型计算：新run `20261031T170800331382Z_39c11741d8fb_ec36b0`（北京时间2026-11-01），config SHA与旧失败尝试完全一致。检查时已完成4题、417次forward；当前磁盘约29GiB，物理GPU2。此前失败run与所有验收结果均保留。

## 2026-11-01T02:25:31.028174+00:00 — 20261031T170800331382Z_39c11741d8fb_ec36b0
RuntimeError: Disk reserve below1GiB; stopped, no data deletion
完整 traceback: `/home/me/PaperField/CalReader/runs/20261031T170800331382Z_39c11741d8fb_ec36b0/traceback.txt`。未回退模型或设置。

## M1 pipeline — generation_failed

Runs: {"smoke_run": "runs/20261030T013639323603Z_13ad784ef5d6_518bf2", "gate_run": "runs/20261030T015147548907Z_f9aeb43dc9b2_daa0ae", "budget_run": "runs/20261030T022507631329Z_ae2889e005b8_1fe957", "generation_run": "runs/20261031T170800331382Z_39c11741d8fb_ec36b0"}. Evidence: `reports/M1_results.md`, `reports/evidence_M1/full_job.json`. No failed gate was bypassed.


## 2026-11-01 — 全量中断后状态核验

全量 run `20261031T170800331382Z_39c11741d8fb_ec36b0` 于北京时间11月1日10:25因磁盘余量低于1GiB保护退出；当前无生成进程。计算进度 reader_train 10147/20000，耗时9.291732 GPU小时，1105526次源forward。持久化manifest含10112题，早退日志含10140题；内存中的末尾不能视为已落盘。现有文件保留，其他split尚未完成。当前磁盘恢复约27.9GiB、配重自动填充仍暂停；运行期间空间骤降原因尚未核实。原生成器禁止覆盖已有输出且不支持恢复，需要验证分片与行为日志边界后实现显式续跑，不能直接原命令重跑或宣称M1完成。


## 2026-11-01 — M1 v2 显式断点恢复

用户要求继续且不因磁盘余量停止。新增 `scripts/m1_generate_v2.py`、`clr/data/resume.py` 和 `configs/M1_data_v2.yaml`：在新目录 `data/dumps/codi_gpt2_v2` 复用完整分片，原结果不覆盖。核验连续分片、张量索引、冻结ID前缀、生产cohort上下文、7步clean、每题8条完整轨迹及早退日志后才续跑。未落盘尾部重新按原种子计算；重复计算成本保留。

取消低余量主动退出，改为低于20GiB时调用配重release补至约25GiB；已提前释放30GiB并保持自动填充暂停。实际I/O失败仍如实报错；60GPU小时（包含前次9.291732小时和预检）及8GiB产物预算保持。恢复逻辑及原policy/data测试6项通过，首次测试发现可变列表共享已修复。

已知统计限制：原中断运行没有持久保存fp32→fp16误差聚合，不能从fp16文件反推；原分片写入时已执行误差阈值检查，恢复后误差汇总仅涵盖新生成状态，报告必须标明，不冒充全量误差统计。尚未完成M1，不训练reader/probe。


2026-11-01 北京时间21:54启动v2续跑，run `20261101T135435455123Z_ed305481698c_e92e9a`，物理GPU2；命令 `PYTHONPATH=. PYTHONUTF8=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_finish_job_v2.py configs/M1_data_v2.yaml device=cuda:2`。恢复核验316个文件/469806行/10112题通过，审计见 `reports/evidence_M1/recovery_audit_v2.json`。自动完成链将继续生成剩余split，计算标签统计和图，更新报告及冻结清单。当前仍在运行，不是完成结果。


## 2026-11-02 — M1续跑进度与空间管理修正

当前v2运行仍在执行。reader_train20000、reader_val2000、val300、cal500已完成；检查时test492/1005，总23292/23805题。续跑耗时11.702GPU小时，含此前中断生成累计20.994GPU小时（另计预检），尚未进入最终报告生成。

发现reserve自动释放参数为小数，CLI只接受整数，历次调用返回2；生成未因该错误停止。已修正为向上取整，并新增回归测试，两项恢复测试通过。当前在途进程仍使用内存中的旧函数，未声称热更新；已显式release30GiB成功，余量约52GiB，自动填充保持暂停，以保障本次剩余生成。研究协议、结果和种子未改变。


## 2026-11-02 — 服务器重启后 v3 恢复

用户确认服务器重启并要求续跑。原v2进程已消失，metrics残留running是重启前快照，不代表仍在执行。重启前test798/1005、v2成本12.800449GPU小时/1606828次forward；原失败生成9.291732GPU小时也计入预算。四个已完成split复用，不重跑。test已持久化768题/5376状态行、9216条MC记录和44条K64记录，通过冻结前缀、cohort、完整步骤和行为日志核验；第769题起重算，共剩237题。

新代码与配置v3、新输出 `data/dumps/codi_gpt2_v3`，原v1/v2不覆盖。已完成文件用硬链接复用，生成器不修改复用文件；本次记录SHA和祖先run成本。重启恢复与policy/data测试8项通过。物理GPU2当前空闲，显式继续使用；release20GiB成功且自动填充保持暂停。自动配重调用已使用整数参数。

命令：`PYTHONPATH=. PYTHONUTF8=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_finish_job_v3.py configs/M1_data_v3.yaml device=cuda:2`。tmux `clr-m1-reboot-v3`，状态 `reports/evidence_M1/full_job_v3.json`。保持原协议、划分、seed及全部实验设置；中断test段的量化误差聚合未持久化，恢复后的test误差统计仅覆盖新生成部分，限制与reader_train一致，不能称为全量统计。

## M1 pipeline — generation_completed

Runs: {"smoke_run": "runs/20261030T013639323603Z_13ad784ef5d6_518bf2", "gate_run": "runs/20261030T015147548907Z_f9aeb43dc9b2_daa0ae", "budget_run": "runs/20261030T022507631329Z_ae2889e005b8_1fe957", "generation_run": "runs/20261102T031137055963Z_24b03138a255_a2c9f2"}. Evidence: `reports/M1_results.md`, `reports/evidence_M1/full_job_v3.json`. No failed gate was bypassed.


## 2026-11-03 — 完成状态与文档同步

M1 v3数据及报告均已完成，待用户审查。共23,805题，产物约3.03GiB，生成累计22.615373GPU小时（另计预检），770个数据文件及6个冻结文件SHA复核通过。当前接续索引为 `docs/STATUS.md`，`AGENTS.md` 已同步，后续状态变化必须同时持久化。保留reader_train/test中断部分量化误差聚合缺失、cohort依赖等限制；无新训练任务启动，原冻结结果未改动。

## 2026-11-03T04:20:46.595165+00:00 — 20261103T042018075557Z_ac120dca4c9d_1a691c
ValueError: Insufficient distinct observed distractors
完整 traceback: `/home/me/PaperField/CalReader/runs/20261103T042018075557Z_ac120dca4c9d_1a691c/traceback.txt`。未回退模型或设置。

## 2026-11-03T04:23:23.487758+00:00 — 20261103T042241754493Z_7e9a24cdc0bd_be4495
ValueError: Out of range float values are not JSON compliant: inf
完整 traceback: `/home/me/PaperField/CalReader/runs/20261103T042241754493Z_7e9a24cdc0bd_be4495/traceback.txt`。未回退模型或设置。


## 2026-11-03 — M1审查通过，M2–M3启动

用户授权合并里程碑，当前不训练CLR。输入隔离源题目文本，状态只用模型实际输出。Q1规则经用户明确：clean仅轨迹起点独立Bernoulli标签及clean终点，traj起点之后用自身观测及终点；不监督更早clean前缀。reader_val报告clean/traj分别指标和各t样本覆盖。cal_A/cal_B250/250已固定。

QA首轮smoke因少量题不足4种答案失败，未覆盖；固定训练集数值池补足后run `20261103T042122100178Z_172c4808908a_81b8b5`通过，输出 `data/M23_v1a/smoke`。全量QA任务tmux `clr-m23-questions`。Q4成对交换正值作为对方负值，保证保留问题上数值边际精确一致；未匹配/解析失败单独统计。全量不得使用失败smoke。

源调用范围冲突按具体要求解释为B5a/C/D/E例外，其他仅读M1；不新增轨迹。GPU当前均忙，显式使用cuda3训练轻量probe、cuda1量化复算，负载与实际OOM如实记录；没有替换模型或降精度。D候选许可和谱系正在核实；尚无完整基线结论或M2–M3完成声明。

## 2026-11-03T04:24:37.710723+00:00 — 20261103T042428840243Z_bccfc4a75de4_c06200
AttributeError: 'dict' object has no attribute 'extend'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261103T042428840243Z_bccfc4a75de4_c06200/traceback.txt`。未回退模型或设置。

## 2026-11-03T04:28:26.722826+00:00 — 20261103T042816573221Z_29ba5c5a548e_585cc7
ValueError: At least three seeds required
完整 traceback: `/home/me/PaperField/CalReader/runs/20261103T042816573221Z_29ba5c5a548e_585cc7/traceback.txt`。未回退模型或设置。

## 2026-11-03T04:30:01.789103+00:00 — 20261103T043001120742Z_f85b0e85ce68_67c15c
RuntimeError: GPU free 2.98 GiB < required 3 GiB
完整 traceback: `/home/me/PaperField/CalReader/runs/20261103T043001120742Z_f85b0e85ce68_67c15c/traceback.txt`。未回退模型或设置。


M23吞吐修正：原layer smoke与首次基线全量在未完成时显式中止，run 20261103T042746003860Z_a5cdd8ef1442_572a81 / 20261103T042831267322Z_43c2e73af758_02fee8 保留，非通过。重复模板/数值编码加入纯函数缓存；layer smoke仅测试0/12层，全量仍13层。未更改模型/标签/训练超参。中止前GPU时间上界分别3.5/2.7分钟，源forward=0。

M23全量训练第二次在预处理显式中止（run20261103T043232575827Z_1cc0d3967c95_84dbc6，无normalization/checkpoint、未测试）。实测单shard numpy复制约0.5–0.9秒，torch转换0.0055秒；改用torch向量化CPU bank/总体标准差，不改变来源状态/标签/超参。py-spy0.4.1已安装用于定位，但ptrace被系统拒绝，未提升权限；用独立只读微基准定位。

M23结构审查修正：中止基线run20261103T044101470059Z_19e9238e5441_14ac1a（先前已保存尝试保留不纳入最终）。B0/B1/B3改为对z+模板特征+选项数值特征联合两层MLP，Q3共享逐选项打分；避免旧加法结构无法表达Q4状态×数值交互。新增交互测试，7项测试通过。此为实现修正，未按test结果选择模型或seed；超参网格/样本量保持。


## M2–M3已知限制（2026-11-03；与运行错误分开）

- B5b为可选项，按任务许可推迟到M6。B5a采用原始z注入CODI输入embedding的Patchscope式代理，表示空间不匹配可能导致负面结果，不声称严格复现外部Patchscope论文。
- Q4当前只生成clean状态问题；仅从可解析方程中间量构造。失败题不出Q4，其计数见M2报告。正负数值通过跨题配对匹配边际，因此样本并非完全独立。
- Q3干扰项使用同题行为及训练/当前split答案池，存在可被B0利用的先验及跨题依赖。conformal报告经验覆盖率；不能无条件宣称层内可交换性成立。
- Q1 reader_val保留轨迹Bernoulli样本，test使用K16均值；NLL/Brier/ECE按Bernoulli定义一致，MAE的标签抽样精度不同。clean/traj指标分开，不补齐缺失clean起点标签。
- 域分类器也可能识别cohort/padding、题长、难度等差异，不能单独证明记忆化。层分析面板只有源训练分布，不作test最优层选择。
- Orca-Math的GSM8K原始test派生谱系与单值gold尚未核实；MAWPS精确数据文件许可/原始下载未核实，均未加入候选生成。入选SVAMP/ASDiv的13-gram去重不能排除语义改写或基础模型预训练接触。
- 部分早期失败/主动中断run没有最终GPU小时或前向计数；成本台账明确缺失，已记录总和仅为下界，不能写成精确总成本。早期未提交脚本仅有事后源码存档；QA经提交代码完整重建逐字节一致，最终基线使用已提交v2。

B5a v2预检首次入口在创建run前触发libLerc/GLIBCXX_3.4.29动态库错误，尚未加载源模型或前向。原因是新增入口先导入pyarrow，修正为与现有已运行入口相同的torch-before-pyarrow导入顺序；无依赖/模型回退。首次日志reports/M23_patchscope_batch_gate.log保留，重试使用独立日志。

B5a v2 smoke首次配置覆盖被严格schema拒绝：batch_gate_run未预先声明（KeyError，加载模型前）。已把已通过gate的run路径写入v2配置，保留原失败日志；无源前向。

2026-11-03 B5a Q3 reader_val已完成（source_forward_calls累计69253）；batch64 smoke完成但耗时78.75秒，原batch8为55.72秒，未见吞吐收益（不同共享GPU，不能归因为batch本身），因此不采用v2替换完整run。保留gate及smoke结果、额外成本；最终协议仍fp32/batch8。


## M23测试集访问协议偏离（审计发现，2026-11-03）

早期smoke使用了固定test的10题子集并计算指标；被中止的早期完整基线尝试也已产生部分test预测。这不满足“test仅最终评估”的严格要求。未使用这些指标选择超参、模型结构、seed或划分；Q4修正源于代码交互结构检查，batch优化只用reader_val数值门槛与运行耗时。保留所有记录，不删除曝光历史，不更换测试集掩盖此问题。最终结果须带此限制交审查；后续milestone smoke一律使用val代理最终评估路径。


M2–M3问题库、B0–B5a、层/域诊断、官方500题、两候选未见池与量化抽查均完成，待用户审查。B5b推迟M6，未训练CLR。报告：reports/M2_questions.md、M3_baselines.md、M3_shift.md、M3_unseen_pools.md；冻结索引 data/M23_v1a/freeze.json。


## M2–M3最终核验（2026-11-03）

实现、全部必需实验、四份报告及图已完成，待用户审查；未训练CLR，B5b按任务许可推迟M6。Q1按用户确认的起点/后续观测状态Bernoulli规则执行。
冻结清单 `data/M23_v1a/freeze.json` 的635个文件SHA全部一致；核验记录 `data/M23_v1a/verification.json`。51个基线checkpoint、216份基线预测、16份B5a预测齐全。测试40 passed；1个已暂停Huginn模块因可选lm_eval缺失跳过。最终报告run `runs/20261103T055223163924Z_faee26776ad3_cea0d0`。
已记录成本下界2.6075 GPU小时、202641次源前向，历史中断缺失成本未记成零。D两池含smoke合计0.417348 GPU小时，低于5小时。
必须随结果保留的审查限制：早期smoke触及10题test，旧中止run保存过部分test预测，未用于调参但不符合严格test-only-final约定；不得声称该偏离已获接受。Q4边际匹配在原split保证，cal_A/B半集没有独立重配。conformal依赖假设、来源未核实候选池及成本缺口见BLOCKERS；不更换划分/seed掩盖问题。
B3 test Q3准确率58.59% ± 0.19pp，Q4 68.26% ± 1.09pp（三seed95% t区间、clean/seen/t等权）；B5a Q3约24.55%、Q4约50.37%，如实保留负面结果。M3_baselines.md末尾已逐项回答三项审查问题。

## 2026-11-03 M4资源现状

配重已耗尽（0块），本次申请40GiB未释放新增空间，当前约15.6GiB可用；不因此停止。所有GPU共享繁忙，最大空闲显存约6GiB。将按val smoke实测配置与预算，禁止隐式降级；详见docs/M4_TASK.md。

## 2026-11-03 M4未见池重叠与方程解析

按任务要求冻结的各池对半划分中，172条unseen_eval与unseen_train有13-gram重叠（SVAMP164、ASDiv8），精确重复0；不擅自重划分，最终另报825条无重叠子集诊断。方程可解析SVAMP990/993、ASDiv744/1000，其余题仅跳过Q4、保留行为标签；见data/M4_v0/equation_failures.json与unseen_train_eval_overlap.json。
