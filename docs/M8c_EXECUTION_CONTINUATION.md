# M8c 单 run 时间上限接续
原21GPUh final_lock.json保持只读。此补充只处理原seed22在8小时单run上限失败时的缺行接续；若原run完整则不执行。
科学设置与原config逐项比较：同200ID/层/权重/校准/三个ε/λ1/PGD200/EOT16/早停。继承原独立部署随机key，已保存行不重算，部分组只补缺失(trial key)；新run与GPU成本全部保留。
新的执行代码/config在再次访问test前以execution_supplement.json锁定；不依据任何test问答或检测率改设置。汇总run是CPU派生，不冒充GPU运行；校验原与接续行无重复、全200×6行后写原主矩阵descriptor。
此接续属于用户必须做满200组的执行修复。若剩余主矩阵耗时预测导致累计超过21GPUh，停止并报告；不裁n或ε。

原seed22于189完整组后8h上限停止。接续资源门槛使用8GiB（与原val门槛相同），共享卡实测free11.3GiB；原seed22单run驻留约9.5GiB。此为显存预检设置调整，模型/精度/batch/科学矩阵不变，OOM仍直接报错。
