Sesame PPO 训练曲线与方法对比（2026-11-03）

优先阅读 report.pdf（8 页）或 report.html（图表已内嵌）。
figures/：5 张中文图，每张 PNG + 矢量 PDF。
tables/：4 份 UTF-8 BOM CSV，包含组统计、逐种子指标与训练里程碑。
media/：10 秒三组对照视频与缩略图，视频不替代统计。
data/：原始统计 JSON 副本、协议、PPO 配置与视频验证。
report_manifest.json：来源与全部产物 SHA-256。

重建仅处理已保存元数据，不启动训练或仿真：
  python3 /home/ywwang/sesame/scripts/plot_study_report.py
  python3 /home/ywwang/sesame/scripts/build_study_report.py

统计单位为每组 3 个独立训练种子，误差条为样本标准差。
结果是本地参考仿真中的范围内泛化，不是官方评分或实机结论。
