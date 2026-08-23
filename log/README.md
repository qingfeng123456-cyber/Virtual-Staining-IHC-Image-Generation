# 实验日志目录

每次通过 CLI 运行训练、验证、推理或数据审计时，项目会在本目录创建一个
轻量日志包。训练 run 的典型位置是：

~~~text
log/<run-id>/train_log_bundle.zip
~~~

日志包不包含 checkpoint、预测图或官方测试图像；它只保存环境、有效配置、
模型模块参数量、逐 epoch 性能、验证指标和错误 traceback，便于从 AutoDL
下载并交给 Codex 分析。

详细使用步骤见 [AutoDL 保姆级指南](../docs/AUTODL_GUIDE.md)。
