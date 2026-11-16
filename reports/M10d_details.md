# M10d evidence

用户任务见docs/M10d_TASK.md。旧版本只读，新版本data/M10d_v1。

B6：data/M10_v1/paper_inputs/main.csv中gpt2/B6/test/Q1/nll=0.574010961648221、Q2/auroc=0.852715892250876。冻结构建scripts/m10_export_gsm.py→scripts/m9_export.py：clean、seen、t等权、三seed。查表无新推断。
B4：M6/M9/M10冻结清单及trained索引未找到B4 checkpoint；upstream_inputs.json记录检索。

M3复用Probe B4（纯线性z→标量，未使用模板/选项；保留未使用参数与AdamW行为），仅状态维度从768扩到2048。数据M6 reader_train6817题/252399条独立Bernoulli Q1，reader_val1000题/37142条。归一化仅训练状态。两LR[.001,.0003]×8epoch，WD=.0001、batch4096，无提前停止；每epoch完整reader_val NLL选优，三seed皆选LR.001/epoch8。Q2不额外训练。
smoke20训练题/20val题（0.293%/2.00%，733/758实例），4.82s，训练、val推断及Q1/Q2分层统计通过；test未读。3项协议单测通过。

训练run /home/me/PaperField/CalReader/runs/20261112T024518010110Z_2c8d73cccc27_1dba42；全部3checkpoint哈希统一写training_lock.json后，final run /home/me/PaperField/CalReader/runs/20261112T024717093641Z_b3ebcf90e887_187be8才加载cal与test；每seed的Q1/Q2各自cal_A温度，未调整cal_B/conformal阈值。保存reader_val/test预测及逐seed/t/clean/traj统计。
最终test三seed均值及95%按题bootstrap10000 CI见four_cells.json；不包含重新训练seed的不确定性。reader_val Q1 clean/traj与Q2见B4_summary.csv。独立原始逐t统计与论文汇总差<1e-7。负面结果：B4的LLaMA Q2 AUROC0.806低于B3-prefix0.852，原样保留。

新增RTX4090 GPU5 resident wall 0.0289412008567 GPUh，含smoke/训练/校准/最终推断；无源前向。CPU汇总首次GLIBCXX导入失败（无GPU调用），显式LD_LIBRARY_PATH使用既有latdefer/lib后同设置成功。code_post_run_snapshot为事后保全，未冒充启动快照；各run保留启动config/git/env。

论文manuscripts/CVPR27-CLR/sec/5_results.tex只改四个TODO及列最优加粗；反向精确替换后与manuscript_before原始字节一致。number_audit.md只追加。原PDF为8页，新PDF状态见manuscript_validation.json。
相关旧数据/预测/校准文件4895项SHA验证一致；这里只核实相关科学输入，不宣称消除了M10c已记录的旧日志漂移。M6/M9/M10 freeze原字节未改。
