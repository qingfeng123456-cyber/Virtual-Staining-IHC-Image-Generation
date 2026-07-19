# 数据目录

此目录已经清空，不再包含开发阶段使用的匿名样例数据。请把赛事官方数据解压到
`dataset/official/`，推荐结构如下：

```text
dataset/official/
├─ train/<organ>/{DAPI,HLA-DR,CD45RO,Vimentin,CD68}/
├─ val/<organ>/{DAPI,HLA-DR,CD45RO,Vimentin,CD68}/
└─ test/<organ>/DAPI/
```

详细命名规则、环境变量、审计、训练、推理和提交命令见项目根目录的 `README.md`。
真实数据、checkpoint、预测图和提交包默认不会进入 Git。
