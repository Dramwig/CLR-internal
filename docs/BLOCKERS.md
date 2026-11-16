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

## M4 控制器状态识别修复（2026-11-03）
控制器首次把已存在的 `metrics.json: running` 误判成失败；实际k4训练一直在运行，未重启/修改模型。已改为等待终态，保留 `reports/M4_training_suite.log`，修复后日志另存v2。无test访问。

## 2026-11-03T06:54:44.111516+00:00 — 20261103T065436633657Z_e4d19c76ca4d_787eb3
ValueError: Out of range float values are not JSON compliant: inf
完整 traceback: `/home/me/PaperField/CalReader/runs/20261103T065436633657Z_e4d19c76ca4d_787eb3/traceback.txt`。未回退模型或设置。

## 2026-11-03T07:03:05.340402+00:00 — 20261103T070304641918Z_4a6694f5b53e_31f5fc
RuntimeError: Insufficient GPU memory; no fallback
完整 traceback: `/home/me/PaperField/CalReader/runs/20261103T070304641918Z_4a6694f5b53e_31f5fc/traceback.txt`。未回退模型或设置。

## M4 D已确认数据限制（2026-11-03）
公式解析失败259/1993（train139、eval120），逐题跳过Q4；另有2题无法找到保持数值边际的负例，跳过Q4且不丢弃其他类型。
172个eval题与新train有13-gram交集、精确0；原要求ID二分保持冻结，另报825无重叠eval题，不声称题族完全独立。证据：`data/M4_v0/qa/summary.json`、`data_audit.json`、`unseen_train_eval_overlap.json`。

## M4 k8显存预检查门槛（2026-11-03）
首次k8在加载模型前因可用显存小于原4GiB门槛退出；CUDA上下文建立后的余量低于nvidia-smi启动前快照。未发生模型OOM，未改模型/batch/精度/数据。
根据已测2.42GiB峰值，将新run门槛显式设3.5GiB；使用新case `pilot_k8_preflight_v2` 保留失败run。若仍不足直接失败，不自动换卡。训练主协议未降级。

## M4 配置语法修复（2026-11-03）
新增pilot映射误用整数YAML键，被共享严格配置加载器拒绝（训练前，无GPU计算）；改为字符串键并对全部M4配置预加载验证。保留 `reports/M4_pilot_k8_v2.log`；研究参数不变。

## M4 正式训练前预算修订
原4epoch计划有欠训练且仅两次验证的限制；完整val实测更快，改用新配置v1的10epoch上限/每2epoch验证/patience2，预算v2含裕量约76GPUh≤80。模型、数据、指标、seed与few-shot设置不变；原pilot/预算保留。此为显式协议修订，非test驱动选择，详见 `reports/M4_budget_v2.md`。

## 2026-11-03T07:39:08.116396+00:00 — 20261103T073907340169Z_79952377610a_41e040
RuntimeError: Insufficient GPU memory; no fallback
完整 traceback: `/home/me/PaperField/CalReader/runs/20261103T073907340169Z_79952377610a_41e040/traceback.txt`。未回退模型或设置。

## M4 GPU1预检查恢复（2026-11-03）
main seed11在GPU1加载模型前因3.5GiB门槛退出，未训练；seed22/33未中断。GPU1相同模型/batch/精度val smoke通过：`runs/20261103T074503297668Z_47e3387fa005_ea103e`，峰值2.41778GiB、置换差0。
新配置 `M4_train_v2.yaml` 明确将预检查余量设3.0GiB，研究设置保持10epoch等不变；失败run保留。新控制器用独立manifest接续，避免终止已有训练，完成后恢复规范索引。没有降低模型或batch。

## M4 v0 收尾
必做实验完成，G1待审查；未见题族重叠、Q4缺失及有限训练预算的影响未被消除，详见 `reports/M4_REVIEW.md` 的(d)/(e)，不以工程完成替代研究结论。

## M4.1 合成诊断对照未达门槛
原合成任务使用2000训练实例、768维独立噪声std=0.08，B3虽可拟合训练集，但Q3独立验证未达预设98%，Q4也出现泛化差距。不能据该任务CLR失败单独判架构失效。
保留原配置/run；新增显式低噪声对照std=0.001、batch256、最多1000epoch，所有候选使用同一生成协议，先核实B3可解性。未修改真实数据/指标/test。

## 2026-11-05T02:33:37.009975+00:00 — 20261105T023326708562Z_1af7d4fa8662_4d1285
OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 5 has a total capacity of 47.51 GiB of which 61.38 MiB is free. Process 17303 has 1.52 GiB memory in use. Process 1969816 has 17.32 GiB memory in use. Process 1995467 has 20.27 GiB memory in use. Process 2006931 has 8.32 GiB memory in use. Of the allocated memory 7.34 GiB is allocated by PyTorch, and 516.83 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T023326708562Z_1af7d4fa8662_4d1285/traceback.txt`。未回退模型或设置。

## 2026-11-05 — M5a初始工程预检
val-only batch128首轮因共享GPU显存突增OOM，run `20261105T023326708562Z_1af7d4fa8662_4d1285`，模型/N/b/K/精度未回退；traceback保留。
batch16首轮exec返回143，终止原因未观测，末尾成本缺失；按0.080GPUh保守预留，不能将running快照当完成。
显式LoRA合并推理smoke数值差max=0.000175、16轨迹答案一致；全量采用前仍须完整预算外推及批处理核对。
成本证据：`data/M5a_v3/cost_ledger_initial.json`；新版本与旧失败run分别保留，不覆盖冻结M1/M23/M4。

## 2026-11-05T02:44:38.652345+00:00 — 20261105T024438093439Z_8694173da1ce_466a25
UnboundLocalError: cannot access local variable 'torch' where it is not associated with a value
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T024438093439Z_8694173da1ce_466a25/traceback.txt`。未回退模型或设置。

## 2026-11-05T02:52:12.661532+00:00 — 20261105T025211583852Z_8694173da1ce_9dcb5c
RuntimeError: GPU free 8.55 GiB < required 20.0 GiB
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T025211583852Z_8694173da1ce_9dcb5c/traceback.txt`。未回退模型或设置。

## 2026-11-05T02:53:48.080836+00:00 — 20261105T025345383130Z_b2b4dc01fa85_04f91c
AttributeError: 'Lite' object has no attribute 'source_token'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T025345383130Z_b2b4dc01fa85_04f91c/traceback.txt`。未回退模型或设置。

## 2026-11-05T02:53:49.251502+00:00 — 20261105T025345947641Z_ae4c70b9aa47_e65717
OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 2 has a total capacity of 47.51 GiB of which 2.81 MiB is free. Process 1250670 has 10.92 GiB memory in use. Process 1951826 has 11.08 GiB memory in use. Process 2048953 has 494.00 MiB memory in use. Process 2051667 has 2.05 GiB memory in use. Process 2109596 has 18.88 GiB memory in use. Process 2124607 has 4.06 GiB memory in use. Of the allocated memory 3.30 GiB is allocated by PyTorch, and 269.53 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T025345947641Z_ae4c70b9aa47_e65717/traceback.txt`。未回退模型或设置。

## M4.1 诊断接续错误
CalledProcessError: Command '['.venv/bin/python', 'scripts/m41_learnability.py', 'configs/M41_gate_v1_overfit_Q4.yaml']' returned non-zero exit status 1.；不自动进入C，检查reports/M41_gate_controller.log。

## 2026-11-05T02:53:58.112899+00:00 — 20261105T025357012986Z_8694173da1ce_bc9f46
RuntimeError: GPU free 19.48 GiB < required 20.0 GiB
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T025357012986Z_8694173da1ce_bc9f46/traceback.txt`。未回退模型或设置。

## M4.1 B门槛显式恢复
首次v1遇共享GPU2实际OOM，lite遇诊断记录器假设source_token存在的AttributeError（首次优化前）。保留两个失败run；修复日志器为可选参数，并记录新gate sigmoid值。
新配置M41_gate_controller_retry_v2将v1明确放GPU1、lite仍GPU5，模型/batch/精度不变；后续三seed固定GPU0/1/5，避开当前波动明显的GPU2。原错误不删除，计入35GPUh。

## M4.1 初诊代码快照限制
初诊运行使用共享Run记录了config/git/env/结果，但新M41脚本当时尚未加入git，启动时working-tree.patch不包含未跟踪文件；不能把后续提交冒充启动快照。
新增实现已提交066ba79，B门槛重试及C均从已跟踪代码运行。初诊来源/当前重现代码与早期合成生成器变化保留在M4.1详细报告；原M4模型与数据仍由冻结SHA约束。

## 2026-11-05T03:02:07.956143+00:00 — 20261105T030206990105Z_e2d0f808c121_78903a
RuntimeError: GPU free 8.88 GiB < required 14.0 GiB
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T030206990105Z_e2d0f808c121_78903a/traceback.txt`。未回退模型或设置。

## 2026-11-05T03:07:10.132769+00:00 — 20261105T030532326046Z_183ecaa7d8c3_f0579b
OutOfMemoryError: CUDA out of memory. Tried to allocate 18.00 MiB. GPU 0 has a total capacity of 47.51 GiB of which 11.88 MiB is free. Process 3594061 has 17.06 GiB memory in use. Process 1251372 has 10.90 GiB memory in use. Process 1353931 has 7.69 GiB memory in use. Process 1749002 has 1.40 GiB memory in use. Process 2196729 has 9.89 GiB memory in use. Of the allocated memory 9.08 GiB is allocated by PyTorch, and 308.60 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T030532326046Z_183ecaa7d8c3_f0579b/traceback.txt`。未回退模型或设置。

## 2026-11-05T03:08:01.071130+00:00 — 20261105T030800205767Z_737802093d4b_c03853
RuntimeError: GPU free 13.73 GiB < required 14.0 GiB
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T030800205767Z_737802093d4b_c03853/traceback.txt`。未回退模型或设置。

## M4.1 报告合成夹具导入顺序修复
独立报告单测先导入pandas，加载了系统旧libstdc++，随后Pillow报GLIBCXX_3.4.29缺失。测试及统计模块明确先导入torch，与现有项目运行时约定一致；未换模型/环境，未访问真实test数据。

## 2026-11-05T03:16:36.397279+00:00 — 20261105T031624056867Z_f9c337740d00_f35c59
OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 MiB. GPU 0 has a total capacity of 47.51 GiB of which 56.06 MiB is free. Process 3594061 has 17.06 GiB memory in use. Process 1251372 has 10.90 GiB memory in use. Process 1353931 has 7.69 GiB memory in use. Process 1749002 has 1.40 GiB memory in use. Process 2265456 has 9.85 GiB memory in use. Of the allocated memory 8.85 GiB is allocated by PyTorch, and 502.34 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T031624056867Z_f9c337740d00_f35c59/traceback.txt`。未回退模型或设置。

## M4.1 C seed11显存恢复
正式batch256在GPU0发生OOM，原run `20261105T030532326046Z_183ecaa7d8c3_f0579b`保留；22/33不中断。明确仅对11启用activation checkpointing，模型/batch/精度/曝光不变，重算成本计入35GPUh。
前向逐元素一致、梯度1e-6容差单测通过；先做256×192 token压力smoke，再新run重试。旧controller会在22/33工作完成后因11旧失败退出；新恢复器等待其退出后读取现有receipt接续，不杀运行中的子进程。

## M4.1 显存压力边界核实
activation checkpointing下人为补到192 token的压力测试仍OOM，失败保留；这不是生产数据长度。随后仅对train/reader_val全量序列化长度审计（2,019,896训练行），最大实际长度48。
新压力测试明确使用batch256×48（全部实际训练输入的上界），生产max_length=192及样本/模型/精度不变，不截断数据。审计data/M41_v1/length_audit.json；未访问test。

## 2026-11-05 — M5a完整网格预算冲突
5题代表性smoke通过，但原网格估算15.5GPUh（含裕量/已有成本）超过用户10GPUh上限，完整val300未启动。
已准备两条可执行路径：原网格16GPUh；或10GPUh下仅S3 oracle做N16/b4全步（其余原设置不变，估算6.88GPUh）。均待用户预算/范围决定。
本次Q1 dummy特征错误已修复并回归核对；合并推理数值差、批量iid seed协议、失败/中断成本限制详见[审查](../reports/M5a_REVIEW.md)。
证据与接续：`data/M5a_preflight_v0/`、`docs/M5a_LIVE.json`；两项研究问题及最终统计缺失，不作收益结论。

## 2026-11-05T06:10:13.700829+00:00 — 20261105T061013264903Z_bc637722f6bc_7437ab
ValueError: Input changed since search startup: docs/M5a_TASK.md
完整 traceback: `/home/me/PaperField/CalReader/runs/20261105T061013264903Z_bc637722f6bc_7437ab/traceback.txt`。未回退模型或设置。

## 2026-11-05T11:31:57.603133+00:00 — M5a投票覆盖补长
全量300题/44400结果完成，greedy冻结核对全部一致；vote_289_11的4096条轨迹成本52937<oracle目标53493，故原gate=false。
明确将该独立S1池上限扩至8192，沿相同batch128随机地址继续，单独run保留原证据，不改搜索设置/结果。
实时GPU6空闲9997MiB，补长仅pad102，显式min_free9GiB、原fp32/batch128；预算现5.81GPUh<10。
恢复证据与结果见data/M5a_v13/final_gate_diagnosis.json及configs/M5a_extend_votes_v13.yaml。

## 2026-11-05T11:34:44.657867+00:00 — M5a v13完成待审查
M5a继续授权后仅S3 oracle限N16/b4全步，其他网格/题/seed/K不缩小；10GPUh内完成。
统计入口首次误用当前任务文档hash核对旧smoke，已改核对其启动时源码快照；保留失败日志，无新增GPU成本。
离散投票存在不足一条轨迹余额，允许预算曲线可欠花且不外推缺失点；oracle MC/终点gold特权不构成严格数学上界。
比例使用固定N16/b4三者实际费用最大值的共同预算；全计费时probe未花满，分母CI含零不解释比例。
本次新增及预检限制见reports/M5a_search.md；旧预检失败/成本缺口原样保留，未作为零成本。

## M4.1 自动接续暂停
CalledProcessError: Command '['.venv/bin/python', 'scripts/m41_train_suite.py', 'configs/M41_train.yaml']' returned non-zero exit status 1.；详细日志reports/M41_pipeline.log。未静默更改预算或跳过关卡。

## M4.1 收尾
工程完成不等于G1通过；根因边界、初版合成控制失败、曝光/预算协议与剩余限制见[审查报告](../reports/M4.1_REVIEW.md)。

## 2026-11-06T01:17:24.648282+00:00 — 20261106T011723751024Z_489485e78f12_fc2fec
KeyError: 'root'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T011723751024Z_489485e78f12_fc2fec/traceback.txt`。未回退模型或设置。

## 2026-11-06T01:27:58.711671+00:00 — 20261106T012748428869Z_3ae3ced99718_ce42fb
AttributeError: 'list' object has no attribute 'transpose'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T012748428869Z_3ae3ced99718_ce42fb/traceback.txt`。未回退模型或设置。

## 2026-11-06T01:28:44.399567+00:00 — 20261106T012836873883Z_3ae3ced99718_4baa53
AttributeError: 'Branch' object has no attribute 'z'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T012836873883Z_3ae3ced99718_4baa53/traceback.txt`。未回退模型或设置。

## 2026-11-06T01:29:30.970652+00:00 — 20261106T012920375285Z_3ae3ced99718_af1855
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T012920375285Z_3ae3ced99718_af1855/traceback.txt`。未回退模型或设置。

## 2026-11-06T01:36:31.169923+00:00 — 20261106T013622023628Z_dba13f08b164_945646
RuntimeError: Missing value-model FLOP profile
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T013622023628Z_dba13f08b164_945646/traceback.txt`。未回退模型或设置。

## 2026-11-06T01:38:37.734746+00:00 — 20261106T013716476785Z_a7f761ff5492_d6f6e8
OutOfMemoryError: CUDA out of memory. Tried to allocate 84.00 MiB. GPU 0 has a total capacity of 47.51 GiB of which 73.75 MiB is free. Process 17376 has 1.52 GiB memory in use. Process 3576943 has 15.54 GiB memory in use. Process 3586263 has 7.55 GiB memory in use. Process 3778541 has 7.69 GiB memory in use. Process 3835359 has 4.63 GiB memory in use. Process 3894383 has 9.94 GiB memory in use. Of the allocated memory 8.96 GiB is allocated by PyTorch, and 561.74 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T013716476785Z_a7f761ff5492_d6f6e8/traceback.txt`。未回退模型或设置。

## 2026-11-06 — M5b A实现复核与历史test暴露
早期A冻结输入干预混合raw均值与normalized当前状态，run `20261106T011905193452Z_6be830dc977b_a69c5e` 已读取test；全量结果排除，不作为调参/结论。正式按原任务实施从头重训、concat标准化前缀均值/当前及lite-single；设置在正式训练前锁定，不能宣称本里程碑test从未暴露。B3状态缓存OOM保留，显式改CPU缓存，模型/batch/精度/曝光不变；详见docs/M5b_TASK.md与data/M5b_v1/invalid_development_runs.json。

## 2026-11-06T02:01:06.463319+00:00 — 20261106T020105753343Z_cecea963673f_9f4aa8
FileNotFoundError: [Errno 2] No such file or directory: 'data/M5b_analysis_smoke/analysis'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T020105753343Z_cecea963673f_9f4aa8/traceback.txt`。未回退模型或设置。

## 2026-11-06T02:06:01.147061+00:00 — 20261106T020550829072Z_c8c264c55f6d_e4852e
FileNotFoundError: [Errno 2] No such file or directory: 'data/M5b_eval_smoke_v1/evaluated_B3-prefix_11_smoke.json.tmp'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T020550829072Z_c8c264c55f6d_e4852e/traceback.txt`。未回退模型或设置。

## 2026-11-06T02:06:22.472011+00:00 — 20261106T020609343310Z_68b855ca28b4_7431f9
FileNotFoundError: [Errno 2] No such file or directory: 'data/M5b_eval_smoke_v1/evaluated_CLR-lite-single_11_smoke.json.tmp'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T020609343310Z_68b855ca28b4_7431f9/traceback.txt`。未回退模型或设置。

## 2026-11-06T05:13:28.693528+00:00 — M5b收尾边界
实验/统计完成；搜索收益关卡未通过，组内排序改善未转化为显著搜索收益。
同覆盖率集合比较是使用评估标签的描述性诊断；计算匹配是平均前向预算上限，非严格等FLOPs/延迟。
A早期无效test暴露仍构成隔离偏离；其处置与全部局限见[唯一审查](../reports/M5b_REVIEW.md)，不重复历史失败。

## 2026-11-06T05:16:33.248867+00:00 — M5b收尾边界
实验/统计完成；搜索收益关卡未通过，组内排序改善未转化为显著搜索收益。
同覆盖率集合比较是使用评估标签的描述性诊断；计算匹配是平均前向预算上限，非严格等FLOPs/延迟。
A早期无效test暴露仍构成隔离偏离；其处置与全部局限见[唯一审查](../reports/M5b_REVIEW.md)，不重复历史失败。

## 2026-11-06T06:38:10.054908+00:00 — 20261106T063809562300Z_4807a4b10761_3ee448
FileNotFoundError: [Errno 2] No such file or directory: 'data/M6_v1/runs/20261106T063809562300Z_4807a4b10761_3ee448.json.tmp'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T063809562300Z_4807a4b10761_3ee448/traceback.txt`。未回退模型或设置。

## M6 可选Huginn与启动状态（2026-11-06）
实时8卡均被占用，无整张空闲≥24GB GPU；按任务许可暂缓Huginn，不阻塞CODI双源。
首次smoke在模型加载前因台账父目录缺失退出（GPU0h/源前向0）；新增mkdir后独立重启，旧run保留。
证据：reports/M6_source_smoke.log；当前reports/M6_source_smoke_v2.log；后续新增错误由recorded逐项记录。

## 2026-11-06T06:39:18.778439+00:00 — 20261106T063842269771Z_4807a4b10761_a8f3d7
ValueError: The `past_key_values` should be either a `Cache` object or `None`.
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T063842269771Z_4807a4b10761_a8f3d7/traceback.txt`。未回退模型或设置。

## 2026-11-06T07:09:57.303989+00:00 — 20261106T070035210710Z_f14e9fb1123a_81badd
RecursionError: maximum recursion depth exceeded in __instancecheck__
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T070035210710Z_f14e9fb1123a_81badd/traceback.txt`。未回退模型或设置。

## 2026-11-06T07:29:37.210051+00:00 — 20261106T072936670814Z_71708f35daaa_8a3278
ValueError: Missing RQ4 domain calibration data: source_gpt2_to_llama
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T072936670814Z_71708f35daaa_8a3278/traceback.txt`。未回退模型或设置。

## M6 RQ1图表入口环境（2026-11-06）
首次CPU报告在导入Pillow时缺GLIBCXX_3.4.29，尚未创建run、GPU0h；证据reports/M6_rq1_report.log。
仅提前导入torch仍失败；显式LD_LIBRARY_PATH=/home/me/anaconda3/envs/latdefer/lib后成功，记录于控制器runtime_env及启动环境。未换模型/数据。

## M6控制器日志命名（2026-11-06）
控制器首次rq1_report复用了独立运行同名log，覆盖了首次CPU导入失败日志；首条原日志不可恢复，同错误第二次日志reports/M6_rq1_report_v2.log保留。
无GPU/预测结果丢失；后续碰到既存log自动使用_controller序号，失败重启用_attempt序号，保留全部后续记录。

## M6 A配置记录偏离（2026-11-06）
A启动YAML继承未使用的M5b网格/v1字段，实际按M6固定小sigma/N8/b4/6步/lite+B3在启动源码实现；未加载/训练v1。协议正确但未全部YAML显式化，属新增配置偏离。
原config/启动代码/逐题输出不改；事后审计data/M6_v1/rq1/configuration_audit.json明确非启动快照。未来入口改读configs/M6_rq1_explicit_protocol_v2.yaml，本次未补造旧元数据或新增GPU实验。

## 2026-11-06T08:18:28.869442+00:00 — 20261106T064505704384Z_46346b5277a6_2acd02
RuntimeError: No LLaMA sigma passes M0.1 drop/nondegeneracy gates
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T064505704384Z_46346b5277a6_2acd02/traceback.txt`。未回退模型或设置。

## 2026-11-06T08:18:35.761680+00:00 — 20261106T075705261265Z_0a05b2e7b9b2_2e99df
RuntimeError: Required prior M6 run failed: runs/20261106T064505704384Z_46346b5277a6_2acd02
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T075705261265Z_0a05b2e7b9b2_2e99df/traceback.txt`。未回退模型或设置。

## 2026-11-06T08:25:12.380211+00:00 — 20261106T082511435547Z_62ca2bc6d9da_4da347
RuntimeError: GPU free 1.58 GiB < required 14 GiB
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T082511435547Z_62ca2bc6d9da_4da347/traceback.txt`。未回退模型或设置。

## 2026-11-06 — M7 新增协议与资源限制
sender direct 33.8%（固定val80）/cot512 50.0%（val8 smoke），两者样本量不同；用户指定主协议统一程序化 commitment。
部署时承诺来自模型，程序化与真实部署有差距；natural_commitment 仅作附加分析、不进主指标/调参/阈值。
历史M3保留seeded消息+adapter+receiver logits，未保存raw relay KV；历史复放与新raw比对分开报告。
完整val估算目标规模1层约179GiB；按15GiB授权减为1层/基础episode464+58+29+29+116，目标规模未达到。
cal_B只有29独立组，α=.01阈值必为∞，不以经验FPR或问题重复数代替独立校准样本量。
反事实只干预语义变量e，可能改变多条支持记录的团队字段；不声称单原始事件编辑。详见docs/M7_TASK.md、data/M7_v1/plan.json。

## 2026-11-06 — M7 评估smoke启动兼容性
评估入口在创建run前因Pillow/libLerc找不到GLIBCXX_3.4.29退出；日志reports/M7_evidence/calibration_smoke.log。
与Idea A相同需Conda libstdc++，新增scripts/m7_env.sh固定LD_LIBRARY_PATH，并明确评估导入顺序；未换模型/精度/数据。
首次失败未执行模型/CUDA调用，只有启动错误日志；重试另存日志和run。不把缺失启动run冒充已记录。

## 2026-11-06 — M7 错配KV位置修正（test前）
原错配receiver单条重编码供体，审计却读原/CF双条批次保存的供体；580个开发基础组中118个供体有1–4个左pad，post-RoPE物理位置因而不同。
保留旧mismatch.jsonl；v2按供体原/CF批次重算、选取第一条并保留完整mask/位置，逐张量和metadata核对保存值后重算receiver。
正式校准/检测改为显式读取mismatch_v2.jsonl，不回退旧版；reader主QA训练只用clean/CF，无须重训。
修正smoke用reader_val2组，之后重算全部train/val/cal错配输出；test尚未生成，正式test直接生成v2。
证据：configs/M7/mismatch_*_v2.yaml、reports/M7_evidence/mismatch_*_v2.log、data/M7_v1/mismatch_v2_complete.json。

## M6 B1门槛未达，按用户预登记降级使用
全部8点无drop<5pp且nondeg≥30%的设置；按已授权规则选sigma=2.0，drop=0.019895833333333335、nondeg=0.14777777777777779。
LLaMA Q1标“标签退化程度高”，源内π分别定义；不把两源Q1视为相同V。证据data/M6_v1/noise_selection_v2.json；原4点失败保持不改。

## M6 Q4前缀子集与CPU审计入口（2026-11-06）
任务要求截ID前缀并复用GPT2原配对；子集不保证原完整split的正负数值边际精确相等，不重新配对；完整data_audit将报告TVD/计数。
独立CPU审计首次缺PYTHONPATH，创建run前失败，GPU0h；日志reports/M6_audit_marginals_smoke_v2.log保留。
按控制器环境重启通过：runs/20261106T122304750730Z_ae564d11b4f6_2cfc7c，仅train/val代理，无reader test；审计补充不改变标签或训练。

## M6 迁移标签预算与跨步支持（2026-11-06）
C3的N只计目标训练QA标签；固定reader_val用于选模，cal_A250题用于所有模型校准，均不计入N，不能将N称为全部监督预算。
跨步cal t≤2、eval t≥4的支持集分离，不满足真实重要性加权的重叠前提；estimated/clipped weighted仅经验诊断，须并列集合大小，不能以全集覆盖宣称有效保证。
方法/标签/划分不改；详细报告显式说明，最终统计仍未完成。

## M6 — Q4 ID前缀的数值边际不平衡
完整审计：reader_train6817题TVD=0.1298147604，reader_val1000题TVD=0.1981981982；cal/test均为0（seen模板、正负数值分布）。
ID前缀不构成原配对闭集；保持用户指定ID前缀与GPT2原配对，不重新配对。不能把训练/reader_val写为数值边际精确平衡；两模型同面板，限制须随Q4结果报告。
证据：runs/20261106T185147038791Z_b8b05e845c1b_9144da/audit.json。

## M6 — 可选D数据的队列优先级修正
原控制器缺少C2/C4前的LLaMA unseen准入判断；在C1进行中修复为C3→D→C2→C4，并继续预留所有已承诺推断。
仅向旧控制器PID发送SIGINT，使线程池等待C1既有队列完成后交接；不向GPU子进程发送信号，不改训练设置。新控制器复用原配置、已完成run与checkpoint。
证据：data/M6_v1/controller_handover_20261107.json、configs/M6_priority_resume.yaml、reports/M6_tests_priority_v2.log（18 passed）；接续tmux clr-m6-priority。

## 2026-11-06T19:45:45.644148+00:00 — 20261106T082813762598Z_ad8b9322f77d_4cb371
RuntimeError: M6 child failed: production_noise_v2; reports/M6_production_noise_v2.log
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T082813762598Z_ad8b9322f77d_4cb371/traceback.txt`。未回退模型或设置。

## M6 — 可选 Huginn 资源复核与预算
接续时GPU0出现整张空闲；初始“无空闲卡”快照不代表全程资源状态。
历史2题计时外推完整复现约26.0GPUh（不确定性高），已超过B剩余2.10GPUh，且尚不含要求的数据生成；不准入，不追加GPU实验。
资源快照/成本依据data/M6_v1/huginn_admission.json；历史复现未完成，不以CODI结果代替。

## M6 — 最终推断调度交接
串行最终推断预计延长等待，改为三GPU独立队列；回归测试验证索引完整及shift/report等待屏障。
CPU控制器SIGINT等待现有C3全量队列结束后交接；GPU作业不被中断，旧父控制器退出失败按原样保留。无模型/训练/校准设置变化，成本继续累计。
证据data/M6_v1/controller_handover_parallel_eval.json、reports/M6_tests_parallel_eval_v6.log（20 passed）；接续tmux clr-m6-parallel。

## 2026-11-06T22:15:37.228609+00:00 — 20261106T192534243350Z_180e2864dfe1_80d0d0
RuntimeError: Resumed M6 controller failed; reports/M6_priority_pipeline.log
完整 traceback: `/home/me/PaperField/CalReader/runs/20261106T192534243350Z_180e2864dfe1_80d0d0/traceback.txt`。未回退模型或设置。

## M6 — C2/C4最终预算省略（test前锁定）
C2估计训练7.96GPUh+推断预留6.26GPUh，连同已用15.8GPUh超CD27，未执行。
C4仅N100两源/三seed完成；完整族实测0.0210训练秒/QA、0.00811推断秒/QA后，N300/1k/3k/full训练估计1.84/2.72/5.24/171GPUh，含新增族推断预留12.8GPUh，均未准入。
96个checkpoint及省略范围已在最终reader test前锁定；不能声称有完整多源结果或N≥300的B5b比较。
证据data/M6_v1/training_lock.json、pipeline_noise_v2.json的cost_admissions；无基于test的准入选择。

## 2026-11-07T05:31:28.358961+00:00 — 20261107T053104602592Z_8b16775e11a4_66c9ad
RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
完整 traceback: `/home/me/PaperField/CalReader/runs/20261107T053104602592Z_8b16775e11a4_66c9ad/traceback.txt`。未回退模型或设置。

## 2026-11-07 — M7.1 层扫描探针优化修订
初版scan训练损失在高维标准化输入上明显振荡（lr=.01），暂停未启动源数据的控制器，保留全部扫描run。
仅据训练内loss预先设v2 lr=.0003/1000steps；选层规则/800–200组划分不变，尚无C1或test反馈。
证据configs/M71/scan_v2.yaml、reports/M71_evidence/layer_scan.log；两版成本均计入30GPUh。

## 2026-11-07 — M7.1 source扩展入口缩进修复
扩展raw controls后GPU2资源smoke入口在Python解析阶段IndentationError，尚未创建run或加载CUDA模型。
修复缩进并全入口py_compile通过；原日志reports/M71_evidence/source_smoke_gpu2.log保留，重试另存。

## 2026-11-07 — M7.1 controls smoke配置入口
首次启动在load_config拒绝YAML未声明的新覆盖键include_controls，未创建run或调用模型。
显式新增configs/M71/controls_smoke.yaml承载两个控制开关；旧日志controls_smoke.log保留，重试另存v2。

## 2026-11-07T06:07:21.706837+00:00 — 20261107T060721353754Z_d709b25b9104_d97209
RuntimeError: Source GPU free 11GiB below configured requirement
完整 traceback: `/home/me/PaperField/CalReader/runs/20261107T060721353754Z_d709b25b9104_d97209/traceback.txt`。未回退模型或设置。

## 2026-11-07 — M7.1 CPU统计布局入口记录
首次人工12组统计布局诊断通过，但以临时Python入口执行，启动源码未纳入run快照；不冒充完整可复现的启动记录，排除正式指标。
新增scripts/m71_statistics_smoke.py与configs/M71/qa_analysis_smoke.yaml后按正式YAML入口重跑；仅人工val代理，无源模型调用、无test访问、0GPUh。
保留两版日志reports/M71_evidence/statistics_val_proxy_smoke{,_v2}.log及对应run；不改任何科学协议或冻结结果。

## 2026-11-07 — M7.1 natural收尾控制器入口修复
调度器v2 YAML的seed映射用了整数键，被严格配置loader拒绝；尚无模型调用或test读取。改用新config natural_finish_v3.yaml的字符串键，原配置/错误日志保留；模型、子集、批次、阈值未改。
证据：reports/M71_evidence/natural_finish_v2.log；新控制器同源码、新配置。

## 2026-11-07T12:43:28.672711+00:00 — 20261107T120616525909Z_944261a2f7c0_691d01
FileExistsError: data/M71_v1/pool/source/test/natural_commitment.jsonl
完整 traceback: `/home/me/PaperField/CalReader/runs/20261107T120616525909Z_944261a2f7c0_691d01/traceback.txt`。未回退模型或设置。

## 2026-11-07T12:58:37.045186+00:00 — 20261107T121722207176Z_7dd6f8e381b6_5e441d
ValueError: M7.1 run still open: runs/20261107T121722207176Z_7dd6f8e381b6_5e441d
完整 traceback: `/home/me/PaperField/CalReader/runs/20261107T121722207176Z_7dd6f8e381b6_5e441d/traceback.txt`。未回退模型或设置。

## 2026-11-07 — M7.1完成待审查
投影准确率只过点门槛，A4损失区间较宽；程序化承诺部署差距与历史raw缺证据保留；text仅test300，不能当作全test2000。
A2/A3近随机；A1 held-out从99.5%降至88.4%，natural未检出诚实推理错误；监督参照未学会，不能充当oracle上界。
收尾路径闭合bug用新版本修复，原统计/失败run保留；三seed CF=97.9/99.9/81.6%，组CI不包含重训seed方差。
唯一审查reports/M7.1_REVIEW.md；完整证据reports/M7.1_details.md、data/M71_v1/analysis_v3/。

## 2026-11-07T20:23:44.608873+00:00 — 20261107T202336020393Z_a73a1f74e407_c7c887
ValueError: Nonintegral public arithmetic
完整 traceback: `/home/me/PaperField/CalReader/runs/20261107T202336020393Z_a73a1f74e407_c7c887/traceback.txt`。未回退模型或设置。

## 2026-11-07T20:23:44.997746+00:00 — 20261107T202335153436Z_cdecbfbad0b7_423222
ValueError: Nonintegral public arithmetic
完整 traceback: `/home/me/PaperField/CalReader/runs/20261107T202335153436Z_cdecbfbad0b7_423222/traceback.txt`。未回退模型或设置。

## 2026-11-07T20:23:45.646847+00:00 — 20261107T202335304820Z_cc0a5e74207b_6c5e79
ValueError: Nonintegral public arithmetic
完整 traceback: `/home/me/PaperField/CalReader/runs/20261107T202335304820Z_cc0a5e74207b_6c5e79/traceback.txt`。未回退模型或设置。
# 2026-11-08 M8a 新增限制（执行中）
两篇文献初始代理请求Connection refused，禁用失效代理后直连已成功取回原文；scope为latent-thought KV，本项目为full-prefill KV，文献方法只能标适配基线，不声称原系统逐项复现。E2真实数域零空间存在FP32矩阵乘舍入残差；额外丢弃改变fp16特征的记录更新以保证实际reader输入逐位一致，单独报告FP32残差。详情最终reports/M8a_details.md。


## 2026-11-08 M8a 完成限制
A1训练负例未覆盖e+2，seed33置信度及校准门限异常有cal_B证据；E2只保证实际fp16 reader输入逐位一致，FP32舍入残差明示。文献为full-prefill适配，未复现原latent-thought拓扑/PCA/RePS。有限PGD/4面板EOT/早停不构成鲁棒保证；50/200主曲线相同主要反映早停。clean receiver准确率仅30.8%，原本目标命中26.8%；主指标排除原本目标，另报clean-correct154组条件失效。正文、逐组CI及运行证据见reports/M8a_REVIEW.md。

## 2026-11-08T06:31:21.881940+00:00 — 20261108T063044359115Z_ec97354b8d4f_b3b200
KeyError: 'sha256'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261108T063044359115Z_ec97354b8d4f_b3b200/traceback.txt`。未回退模型或设置。

## 2026-11-08T06:33:50.630971+00:00 — 20261108T063242344584Z_85a04fb5da6c_d02266
FileNotFoundError: [Errno 2] No such file or directory: 'docs/M8b_LIVE.json.tmp' -> 'docs/M8b_LIVE.json'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261108T063242344584Z_85a04fb5da6c_d02266/traceback.txt`。未回退模型或设置。

## 2026-11-08 — M8b 开发与协议限制
LIVE并发临时文件冲突已加文件锁，freeze字段兼容已修复；原失败记录/成本保留。R2 seed33一次进程消失无终态，原记录不补造；额外保守计入10.1分钟上界，同设置在GPU5重启。
R2当前固定随机f、128维、冻结主体的新适配器在val A1/A4接近随机；warm小样本诊断亦未充分学好，不作为有效鲁棒性提升。所有负面结果保留。
随机f现场/缓存改为同一fp16充分统计；反传inplace问题在val前修复并加raw-KV梯度测试。完整数据实占17.5GiB（初次文字估算误算13.5GiB），未发生I/O错误。
详情docs/M8b_TASK.md、reports/M8b_evidence/，最终test尚未打开。

## 2026-11-08T13:29:23.529105+00:00 — 20261108T132913426339Z_6d673f74ff63_62766a
ValueError: Source replay differs: R3 {'exact_fraction': 0.9999999403953552, 'max_absolute': 0.0}
完整 traceback: `/home/me/PaperField/CalReader/runs/20261108T132913426339Z_6d673f74ff63_62766a/traceback.txt`。未回退模型或设置。

## 2026-11-08T16:08:42.752020+00:00 — 20261108T160830165990Z_f922e56d8f22_40af4f
ValueError: Source replay differs: R2 {'exact_fraction': 0.9984580592105263, 'max_absolute': 0.00390625}
完整 traceback: `/home/me/PaperField/CalReader/runs/20261108T160830165990Z_f922e56d8f22_40af4f/traceback.txt`。未回退模型或设置。

## 2026-11-08T16:10:23.504553+00:00 — 20261108T161023220310Z_186a526b8a56_0bae0c
RuntimeError: GPU memory precheck
完整 traceback: `/home/me/PaperField/CalReader/runs/20261108T161023220310Z_186a526b8a56_0bae0c/traceback.txt`。未回退模型或设置。

## 2026-11-09 M8b test前预算调整
完整val外推加20%余量与加载预留，主500+额外λ500预计27.4GPUh>25。
主500/三seed/四设计/E1–E3/四ε/50与200全部保留；附加λ=.1/10敏感性改固定102组，ID在test前按无标签规则冻结，不能把其CI当500组。
主表λ1及三λ102组附加结果分列；配置、公式和冻结ID见docs/M8b_TASK.md、data/M8b_v1/budget_lock.json。

## 2026-11-09 M8b 完成限制
R2/R4语义读取未达近完美；新适配器的训练负例修复、初始化与f维度均可能影响设计比较，负面结果完整保留。
R1均值NLL规则不满足独立单问veto指数定理；有限PGD/EOT与成功早停只给经验失效，不能证明最坏情况鲁棒性。
E1/E2针对原f盲区；R4需先消费消息，不能在消费前拒收。共享GPU时延、CPU/GPU舍入修复及孤儿run成本缺证据见reports/M8b_REVIEW.md。

## 2026-11-09T01:10:20.857960+00:00 — 20261109T011013351421Z_bdabd784be93_7aa729
ValueError: Public answer references do not match candidates
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T011013351421Z_bdabd784be93_7aa729/traceback.txt`。未回退模型或设置。

## 2026-11-09T01:12:42.162775+00:00 — 20261109T011216792176Z_bdabd784be93_7581c5
RuntimeError: M8c direct-lookup admission gate failed
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T011216792176Z_bdabd784be93_7581c5/traceback.txt`。未回退模型或设置。

## 2026-11-09T01:30:22.596550+00:00 — 20261109T013022218907Z_acc23fc19a4c_e65e05
RuntimeError: Adapter GPU memory precheck failed
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T013022218907Z_acc23fc19a4c_e65e05/traceback.txt`。未回退模型或设置。

## 2026-11-09T01:31:05.935930+00:00 — 20261109T013105546699Z_f883ca61198a_a46111
RuntimeError: Adapter GPU memory precheck failed
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T013105546699Z_f883ca61198a_a46111/traceback.txt`。未回退模型或设置。

## 2026-11-09T01:31:31.268355+00:00 — 20261109T013129855015Z_04f475f20e4f_edb344
RuntimeError: M8b total GPU budget exceeded
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T013129855015Z_04f475f20e4f_edb344/traceback.txt`。未回退模型或设置。

## 2026-11-09T02:19:20.385759+00:00 — 20261109T021920039300Z_6928b71c0a29_bf9d8c
RuntimeError: Adapter GPU memory precheck failed
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T021920039300Z_6928b71c0a29_bf9d8c/traceback.txt`。未回退模型或设置。

## 2026-11-09T03:06:49.666885+00:00 — 20261109T030642082724Z_e02c2e28a946_42e3f9
OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 5 has a total capacity of 47.51 GiB of which 18.75 MiB is free. Process 17303 has 1.52 GiB memory in use. Process 537856 has 704.00 MiB memory in use. Process 538953 has 22.79 GiB memory in use. Process 1118439 has 15.98 GiB memory in use. Process 1319700 has 6.49 GiB memory in use. Of the allocated memory 6.05 GiB is allocated by PyTorch, and 1.70 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T030642082724Z_e02c2e28a946_42e3f9/traceback.txt`。未回退模型或设置。

## 2026-11-09 M8c开发记录
full日志依赖门槛失败，按授权short通过；R4低曝光仅诊断，正式三seed各≥4.224M问题。
共享GPU5加载OOM未换模型/精度；一个误启重复val诊断已中止，真实记录及保守成本补差保留。
M8b LIVE误更新已修复；上游实际33897文件SHA均一致，详data/M8c_v1/upstream_prelock.json。

## 2026-11-09T04:13:01.741567+00:00 — 20261109T041301368563Z_8e83cd288ddb_e0d696
KeyError: 'max_context'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T041301368563Z_8e83cd288ddb_e0d696/traceback.txt`。未回退模型或设置。

## 2026-11-09T05:43:33.721104+00:00 — 20261109T041333353840Z_482eec68a407_a050a8
RuntimeError: M8c wall-time limit
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T041333353840Z_482eec68a407_a050a8/traceback.txt`。未回退模型或设置。

## 2026-11-09 M8c 新test协议预算未准入
用户要求主test n≥200；当前新三ε矩阵最大可行估计124组，点估计181组，保守做满200组累计20.2GPUh>15。
按用户规则停止GPU任务，等待追加预算决定；test未打开，旧60组方案从未锁定或执行，次要100组尚未执行。
val主矩阵90分钟触发超时（19/20组完整），旧λ敏感性中止按原心跳加终止上界补差计费；所有部分记录保留。
吞吐来自19完整val组，末组截断可能低估长尾；有限PGD曲线非单调不能解释为认证鲁棒性。详细reports/M8c_REVIEW.md、data/M8c_v1/budget_feasibility.json。

## 2026-11-09T14:10:44.837006+00:00 — 20261109T061044089104Z_8ab574f90d00_cd6884
RuntimeError: M8c wall-time limit
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T061044089104Z_8ab574f90d00_cd6884/traceback.txt`。未回退模型或设置。

## 2026-11-09T14:10:48.060581+00:00 — 20261109T061036143169Z_8412e9a482af_508978
RuntimeError: Primary child failed: ['final_test_seed22']
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T061036143169Z_8412e9a482af_508978/traceback.txt`。未回退模型或设置。

## 2026-11-09T17:29:21.979064+00:00 — 20261109T144316687997Z_9adf6a431b6a_5bec2b
RuntimeError: M8c GPU budget shutdown reserve reached
完整 traceback: `/home/me/PaperField/CalReader/runs/20261109T144316687997Z_9adf6a431b6a_5bec2b/traceback.txt`。未回退模型或设置。

## 2026-11-09 M8c 交付限制
21GPUh预算内实际λ网格、最小扰动样本规模与E1/E2未执行范围均见reports/M8c_REVIEW.md及budget_lock.json，未把部分网格冒充全量。
二分为有限PGD搜索，不是认证最小值；未成功/原已目标/截尾及独立报告检测另列，条件比值不能代表所有episode。
依赖门槛仅验证群体依赖；R4在消费消息后检测，conformal clean阈值不保证自适应攻击下的检测率。

## 2026-11-10T05:31:02.025010+00:00 — 20261110T053055184672Z_fa94d5b3eb4d_29a6e7
KeyError: 'ci95'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T053055184672Z_fa94d5b3eb4d_29a6e7/traceback.txt`。未回退模型或设置。

## 2026-11-10T05:46:30.642386+00:00 — 20261110T054401981250Z_42c3172908df_f047ab
KeyError: 'options'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T054401981250Z_42c3172908df_f047ab/traceback.txt`。未回退模型或设置。

## 2026-11-10 M9交付限制
LLaMA B0/prefix/single-state对照缺失，论文标NA，按任务不补跑；联合/单源比较有训练规模及优化混杂。
CPU导出旧格式缺元数据/区间字段错误保留失败run；用冻结QA恢复选项，逐项复放原conformal值，无新推断。
首次制图环境代理安装失败，直连完成；系统缺LaTeX引擎，未执行TeX编译。早期成本缺口标下界，不给项目完整总计。
静态协议v1保留但不用于论文，v2读取实际冻结噪声选择；控制器调度交接保留历史running记录与终止说明。
结论边界与不能主张清单见reports/FINAL_SUMMARY.md；以上未靠新增实验补齐。

## 2026-11-10T09:48:25.362637+00:00 — 20261110T094822526172Z_4647f82145ec_60f395
ValueError: Out of range float values are not JSON compliant: inf
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T094822526172Z_4647f82145ec_60f395/traceback.txt`。未回退模型或设置。

## 2026-11-10T09:58:10.376627+00:00 — 20261110T095720277292Z_6a754f2457ab_0cb2f7
OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 3 has a total capacity of 47.51 GiB of which 18.69 MiB is free. Process 1565510 has 9.98 GiB memory in use. Process 3956603 has 17.74 GiB memory in use. Process 272553 has 10.91 GiB memory in use. Process 293745 has 8.46 GiB memory in use. Of the allocated memory 7.77 GiB is allocated by PyTorch, and 219.11 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T095720277292Z_6a754f2457ab_0cb2f7/traceback.txt`。未回退模型或设置。

## 2026-11-10T10:00:19.943383+00:00 — 20261110T095931513341Z_1f9c52542969_68953b
OutOfMemoryError: CUDA out of memory. Tried to allocate 52.00 MiB. GPU 4 has a total capacity of 47.51 GiB of which 37.56 MiB is free. Process 1421060 has 30.73 GiB memory in use. Process 296103 has 1.03 GiB memory in use. Process 308848 has 15.69 GiB memory in use. Of the allocated memory 13.71 GiB is allocated by PyTorch, and 1.48 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T095931513341Z_1f9c52542969_68953b/traceback.txt`。未回退模型或设置。

## 2026-11-10T10:00:20.731520+00:00 — 20261110T095742657265Z_42ae1eb7dc8a_d0e886
OutOfMemoryError: CUDA out of memory. Tried to allocate 262.00 MiB. GPU 4 has a total capacity of 47.51 GiB of which 169.56 MiB is free. Process 1421060 has 30.73 GiB memory in use. Process 296103 has 924.00 MiB memory in use. Process 308848 has 15.69 GiB memory in use. Of the allocated memory 395.05 MiB is allocated by PyTorch, and 16.95 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T095742657265Z_42ae1eb7dc8a_d0e886/traceback.txt`。未回退模型或设置。

## 2026-11-10 M10 LLaMA补充吞吐面板比例偏离
LLaMA reader_val实际1000组；run 20261110T094237091635Z_bb9fab37a1bf_56ff0b的100组补充吞吐面板占10%，违反≤5% smoke要求，保留原记录。
此前正式训练引用的20组smoke合规（2%），本次偏离未读test、不用于选模型；不能将100组面板写成合规smoke。
新增自动5%组比例检查，补跑configs/M10/llama_CLR-lite-single_throughput_corrected50_smoke.yaml（50/1000）。失败/补跑成本均计入P1。

## 2026-11-10 M10 B5c吞吐与P1资源竞争失败
B5c batch64/checkpoint smoke先因GPU3新增占用OOM；换GPU4时与本任务P1 prefix22重叠，造成两项OOM，属调度失误。
失败run/traceback均保留并计入各自子预算；prefix22以同一科学配置重跑，不择seed/换数据。
新增M10逐GPU进程锁，禁止本任务两个新run同时占同卡；恢复P1后，B5c只在独立GPU2重测。
证据reports/M10_evidence/B5c_gpt2_checkpoint64*_smoke.log、llama_B3-prefix_22_full.log；无test访问。
复核另发现B5c把冻结主体强制eval使HF激活checkpoint未实际生效；已改为dropout=0的train模式（基座权重仍冻结），在独立GPU2新版本重新测量。

## 2026-11-10 M10 P2-4 B5c全量预算准入未通过
r16 LoRA/2层映射/Qwen3-1.7B的val-only smoke成功；实测batch16、真实checkpoint64、checkpoint512。
最快实测下仅GPT2 Q3/Q4同一全训练集一遍×3seed已超过19GPUh；LLaMA/审计/校准/test另需成本，超过独立12GPUh上限。
按用户预算取舍规则不启动全量、不擅自缩标签或seed；B5c最终比较缺失，论文不得声称CLR优于B5c。
脚本scripts/m10_b5c_budget.py；配置configs/M10/B5c_budget_admission.yaml；精确估算data/M10_v1/B5c_budget_admission.json。未读test。

## 2026-11-10T10:26:23.354307+00:00 — 20261110T102622070636Z_2c2c921ceb77_dcf9c7
TypeError: Qwen3ForCausalLM.__init__() got an unexpected keyword argument 'dtype'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T102622070636Z_2c2c921ceb77_dcf9c7/traceback.txt`。未回退模型或设置。

## 2026-11-10T10:28:43.867756+00:00 — 20261110T102822333305Z_2c2c921ceb77_196890
TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T102822333305Z_2c2c921ceb77_196890/traceback.txt`。未回退模型或设置。

## 2026-11-10 M10 HiddenBench协议与源smoke
65案例仅63独立公开描述组，按13/20/4/20/6组冻结train/val/cal_A/cal_B/test；1%阈值只能为+∞，test仅6组，结果限探索性。
原结论不可逆推出自由文本事实；commitment统一显式声明被审计事实及原结论，只做mismatch，不能外推counterfactual。
复用M71真实KV特征与projection；通用文本选项显式增序列上限至384（局部位置<256），不截断、不套固定team编码。
源smoke首次误用主环境dtype加载失败，通信环境重跑后汇总漏return失败；均保留计费，retry3通过，val1组=5%。
训练smoke按QA标签抽≤5%；独立组粒度为1/13，有限场景无法达到组数≤5%，明确披露，val仍1/20。
证据data/M10_v1/hiddenbench/protocol.json、source_smoke.json及reports/M10_evidence/hiddenbench_*；无test推断。

## 2026-11-10T10:53:40.287891+00:00 — 20261110T105339605898Z_d1d410df3241_ab97a9
FileNotFoundError: [Errno 2] No such file or directory: 'data/M10_v1/hiddenbench/protocol_public_v2.json'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T105339605898Z_d1d410df3241_ab97a9/traceback.txt`。未回退模型或设置。

## 2026-11-10 M10 HiddenBench公开评分契约澄清
原结论及“审计事实”声明不足以推出其他候选事实不存在；test前新建protocol_public_v2.json，明确“候选面板中恰有一个事实”。
所有类别统一声明；程序验证A1/A4实际模型输入、标签和期望标签逐项不变，KV/receiver提示不变，旧protocol.json保留。
JSON字段首次按分号切分遇到事实内标点失败；改为JSON解码边界并加转义/分号单测；无test模型调用。
新公开契约val-only评估smoke通过；证据public_protocol_verification.json、hiddenbench_public_v2_eval_smoke_retry2.log；11项单测通过。

## 2026-11-10T13:30:08.820723+00:00 — 20261110T132949706064Z_c1aeb1a7d2a2_3a1082
TypeError: latdef.utils.run.Run.write_metrics() got multiple values for keyword argument 'test_opened'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T132949706064Z_c1aeb1a7d2a2_3a1082/traceback.txt`。未回退模型或设置。

## 2026-11-10T13:30:12.477711+00:00 — 20261110T132951415825Z_8bf099707dc1_60e3c4
TypeError: latdef.utils.run.Run.write_metrics() got multiple values for keyword argument 'test_opened'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T132951415825Z_8bf099707dc1_60e3c4/traceback.txt`。未回退模型或设置。

## 2026-11-10T13:31:38.681210+00:00 — 20261110T132952908184Z_bcc4a73b5a70_40810e
TypeError: latdef.utils.run.Run.write_metrics() got multiple values for keyword argument 'test_opened'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T132952908184Z_bcc4a73b5a70_40810e/traceback.txt`。未回退模型或设置。

## 2026-11-10T13:37:18.516843+00:00 — 20261110T132951294202Z_8e9a03f8d101_7fc8e4
TypeError: latdef.utils.run.Run.write_metrics() got multiple values for keyword argument 'test_opened'
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T132951294202Z_8e9a03f8d101_7fc8e4/traceback.txt`。未回退模型或设置。

## 2026-11-10T15:25:25.640644+00:00 — 20261110T152525316247Z_ed7c1848e24f_fcb832
SameFileError: PosixPath('/home/me/PaperField/CalReader/runs/20261110T152525316247Z_ed7c1848e24f_fcb832/config.json') and PosixPath('/home/me/PaperField/CalReader/runs/20261110T152525316247Z_ed7c1848e24f_fcb832/config.json') are the same file
完整 traceback: `/home/me/PaperField/CalReader/runs/20261110T152525316247Z_ed7c1848e24f_fcb832/traceback.txt`。未回退模型或设置。

## M10b 新增工程失败与曝光说明
首次source_snapshot自复制失败保留run/traceback，修复后同设置smoke通过；一次未source环境import失败，无GPU调用。
前者原GPU计时缺失，独立账本保守计3.6秒；不作为模型结果。M10原B5c循环epochs1/min_epochs8未使用，M10b明确一遍完整标签。
曝光少于八遍CLR/B6，不能单独归因架构/语言化设计；LLaMA、Q1/Q2、审计均不在授权范围。详情docs/M10b_TASK.md。

## M10b 上游核验发现既有非指标日志差异
136269项核验仅reports/M10_evidence/tests_final.log与M10冻结SHA不符；日志mtime23:03:33，M10 freeze23:02:05/audit23:02:22，M10b首run23:25:25。
这是M10b启动前的pytest stdout改写；原字节未找到，不重造、不修改旧freeze，当前字节与SHA/时间保存在data/M10b_v1/preexisting_log_drift.json。
其余模型/数据/指标136268项一致；科学实验继续，最终报告将此列为未通过的字节完整性例外，不声称上游全部一致。

## 2026-11-11T04:02:43.312671+00:00 — 20261111T040027345050Z_36af86beac40_113c40
OutOfMemoryError: CUDA out of memory. Tried to allocate 420.00 MiB. GPU 3 has a total capacity of 47.51 GiB of which 337.56 MiB is free. Process 1565510 has 10.48 GiB memory in use. Process 1830350 has 11.74 GiB memory in use. Process 548240 has 8.77 GiB memory in use. Process 548373 has 16.17 GiB memory in use. Of the allocated memory 7.58 GiB is allocated by PyTorch, and 731.89 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261111T040027345050Z_36af86beac40_113c40/traceback.txt`。未回退模型或设置。

## 2026-11-11 M10b seed33资源中断与恢复
seed11/22完整完成；33于12:02共享GPU3显存OOM（0.0377GPUh保留），原控制器退出，预算watch只防硬上限未自动恢复；17:00状态检查发现。
所有卡空闲<9GiB，原设置峰值预留19.4GiB；新增CPU队列连续3次≥22GiB后同设置重跑33，仅device可变，不重训11/22；等待0GPUh。
当前12.5/22GPUh，按最慢完整seed+2GPUh预留预计21.4；ETA为恢复后8–9h加资源等待。详细接续docs/M10b_EXECUTION_CONTINUATION.md。

## 2026-11-11T11:27:10.260703+00:00 — 20261111T112434784211Z_beec019ccbd0_c9844b
OutOfMemoryError: CUDA out of memory. Tried to allocate 280.00 MiB. GPU 1 has a total capacity of 47.51 GiB of which 176.00 MiB is free. Process 3076674 has 8.76 GiB memory in use. Process 3133646 has 8.41 GiB memory in use. Process 3698541 has 20.25 GiB memory in use. Process 3678761 has 9.89 GiB memory in use. Of the allocated memory 8.83 GiB is allocated by PyTorch, and 568.33 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261111T112434784211Z_beec019ccbd0_c9844b/traceback.txt`。未回退模型或设置。

## 2026-11-11 M10b无显存阈值实际OOM
依用户取消显存阈值等待/退出，以GPU1、min_free_GiB=0和完整原batch512尝试；首batch实际OOM，申请280MiB时仅余176MiB。
run20261111T112434784211Z_beec019ccbd0_c9844b失败0.0431GPUh保留计费，累计12.54/22；11/22已完成权重不动，33未完成、未test。
原等待队列已停；当前停止依据是真实OOM，不再使用显存阈值，预算硬上限保留。证据data/M10b_v1/controller_resume_v3.log。

## 2026-11-12T01:18:18.742059+00:00 — 20261112T011813077212Z_d14fbe16e0be_6d04d0
ValueError: FLOP instrumentation failed equivalence check
完整 traceback: `/home/me/PaperField/CalReader/runs/20261112T011813077212Z_d14fbe16e0be_6d04d0/traceback.txt`。未回退模型或设置。

## 2026-11-12T01:19:03.209141+00:00 — 20261112T011857895959Z_d14fbe16e0be_ca0f95
ValueError: FLOP instrumentation failed equivalence check: CLR-lite Q3 b1 FLOPs=0.0, error=0.0, argmax=True
完整 traceback: `/home/me/PaperField/CalReader/runs/20261112T011857895959Z_d14fbe16e0be_ca0f95/traceback.txt`。未回退模型或设置。

## M10c 开销口径与完整性例外
B3独立训练账本缺失，NA；GPU共享，未测独占/端到端预处理延迟。
两次smoke FLOP计数为0，no_grad修复且完整计费；trace与精确口径见reports/M10c_details.md。
启动前已有M10b完成stdout/LIVE哈希漂移，不改旧freeze；科学哈希一致，见data/M10c_v1/upstream_audit.json。
M10 pytest日志既有例外已按授权加FINAL_SUMMARY；独立paper复现通过，未编译TeX。

## 2026-11-12 — M10d 环境接续
CPU汇总首次因系统libstdc++缺GLIBCXX_3.4.29退出；显式使用latdefer/lib后同协议重跑成功（reports/M10d_summary*.log）。
内置TeX编译器CryptUnprotectData失败；改用项目内独立Tectonic，未改论文科学内容。首次编译器下载因默认代理拒连改为直连。
最终编译与范围验证见data/M10d_v1/manuscript_validation.json；模型/阈值无回退。

## 2026-11-12 — M10d 编译接续
独立Tectonic已完成8页编译；默认TU编码使Times字体回退，采用独立build wrapper指定T1以匹配原pdfLaTeX字体，正文源不改。
日志reports/M10d_compile*.log；完整编译状态/页数/显示核验见data/M10d_v1/manuscript_validation.json。

## 2026-11-12T07:25:34.045065+00:00 — 20261112T072248432462Z_189260e52f8b_40c2ab
AssertionError: ({'source': 'llama', 'model': 'B2', 'split': 'reader_val', 'qtype': 'Q4', 'kind': 'clean', 'metric': 'accuracy', 'mean': 0.4994279994279994, 'lo': 0.4920152295152295, 'hi': 0.5067925067925069}, 2.3833357166824776e-05)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261112T072248432462Z_189260e52f8b_40c2ab/traceback.txt`。未回退模型或设置。

## M10e — 方法适配与论文交接
HSRM按任务用Bernoulli BCE替代原文组内排序；CLUE末层prefix delta替代原文全层显式思维边界，完整轨迹质心不随t重训。
无模型/训练精度/批大小降级；CPU导出精度审计失败后按既有表1 float64口径双重重放修复，原日志/预测/温度保留。
论文改稿/8页编译按用户后续指令交给另一个对话，不能标作已验收。证据reports/M10e_details.md、M10e_PAPER_HANDOFF.md。

## 2026-11-12 — M11-qual 导出边界与修复
M8c未保存扰动张量和同一攻击的另一设计审计分数；按授权仅导出已有分数，缺字段显式null，未重跑PGD。
Q4补足沿用跨题中间量负例族、按供体哈希固定；不宣称扩展面板仍有原题库精确数值边际匹配。
附加LLaMA linear_probe无冻结conformal阈值，标null；任务必需Q3方法的阈值/集合完整。
M5b冻结值为排序logits，原值保留，另用冻结同权重Q1温度计算概率；未拟合新温度。
两次val smoke因reader_val无MC文件、旧conformal未按方法/seed过滤失败，已修复；未访问test样本，费用全部保留。
原trace：runs/20261112T120657331440Z_7860e4ef7195_cfca45、runs/20261112T120930019017Z_7860e4ef7195_38ca87；详见reports/M11_qual_REVIEW.md。

## M12 — 数值调用形状核验
初始严格训练等价测试发现二分类头3D/原2D调用有微小舍入差；正式训练前恢复原2D形状，四类型输出与梯度逐bit测试通过。
原/修正版smoke run均保留，修正版全关卡重跑且所有成本计入；未访问test。证据reports/M12_evidence/unit.log、unit_v2.log及各run启动源码快照。

## 2026-11-12T19:22:57.294160+00:00 — 20261112T192135363582Z_7d7cf0bfdc8a_60af67
OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 3 has a total capacity of 47.51 GiB of which 88.75 MiB is free. Process 1565510 has 12.16 GiB memory in use. Process 3229549 has 7.69 GiB memory in use. Process 450607 has 8.04 GiB memory in use. Process 581864 has 15.99 GiB memory in use. Process 583702 has 3.24 GiB memory in use. Of the allocated memory 2.58 GiB is allocated by PyTorch, and 155.34 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
完整 traceback: `/home/me/PaperField/CalReader/runs/20261112T192135363582Z_7d7cf0bfdc8a_60af67/traceback.txt`。未回退模型或设置。

## M12 — A4 seed11共享显存OOM及恢复
GPU3在首epoch一个优化步后OOM，失败run保留并计0.0225GPUh，未触test。
同模型/batch1024/bf16/完整曝光重新训练至GPU0；A4 seed22/33不中断。
恢复证据data/M12_ablation/recovery.json；原控制器等待另两seed后退出，接续复用完成回执。

## 2026-11-12T22:14:02.887621+00:00 — 20261112T194218454921Z_9ee3281182c3_923a43
RuntimeError: Recovered pipeline failed with exit code 1
完整 traceback: `/home/me/PaperField/CalReader/runs/20261112T194218454921Z_9ee3281182c3_923a43/traceback.txt`。未回退模型或设置。

## M12 — 置换脚本依赖导入恢复
最终15份评估已完成；置换入口在recorded/GPU初始化前因Pillow/libLerc需要GLIBCXX_3.4.29退出。
显式LD_PRELOAD现有latdefer libstdc++.so.6后依赖导入预检通过；不改科学代码/配置/数据/精度。
从置换阶段接续，不重跑训练/已有评估；证据data/M12_ablation/postprocess_environment_recovery.json。

## M12 — CPU bootstrap等价提速
原高级索引合成计时41.1秒，计数矩阵0.398秒；10000次RNG(11)抽样完全相同、合成点估计/CI误差≤1e-14，9测试通过。
显式中止旧CPU统计后重新读取原预测汇总，无新GPU/推断/校准；旧日志/源码快照保留。
证据data/M12_ablation/statistics_runtime_recovery.json；旧冻结实现未改。

## M12 — 解释范围
A1保留原逻辑位置及规范化排序；零置换敏感性不能证明选项隔离，raw诊断另列。
复用设计有不同训练数据/曝光；其差异不作严格单组件因果结论；bootstrap条件于原校准集。
新增未解决运行阻塞：无。
详细reports/M12_details.md；旧科学输入SHA核验见data/M12_ablation/upstream_audit.json。
