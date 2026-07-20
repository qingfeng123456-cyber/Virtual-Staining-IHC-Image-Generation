# 数据目录

开发阶段的匿名样例数据已经删除。当前初赛官方数据位于
dataset/official/，实际结构如下：

~~~text
dataset/official/
├─ train/
│  ├─ DAPI/
│  ├─ HLA-DR/
│  ├─ CD45RO/
│  ├─ Vimentin/
│  └─ CD68/
└─ test/
   └─ DAPI/
~~~

数据没有 val 文件夹，程序按完整 ROI 从 train 中固定生成本地 train/val。
详细命名规则、审计、训练、本地评分、test 推理和提交命令见项目根目录 README.md。
真实数据、checkpoint、预测图和提交包默认不会进入 Git。
