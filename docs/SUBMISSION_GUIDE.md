# 提交指南

## 目录

单目标 CD68 的正确目录类似：

```text
results/
└─ test/
   └─ CD68/
      ├─ 00000_fake.jpg
      └─ 00001_fake.jpg
```

输出 stem 与测试 DAPI stem 一一对应，后缀只出现一次。当前数字 stem 合法，validator 不要求文件名以 ROI 开头。除非官方明确要求四目标，否则只生成 `submit_targets` 中的目录。

## 生成

先运行 predict，再运行 make-submission。writer 只执行 clamp、uint8 round 和验证集确定的模型平均；不 blur、不 resize、不颜色校正。JPEG 质量 100、4:4:4 subsampling，并且只编码一次。

## 校验

`validate-submission` 检查层级、target、数量、缺失、多余、重复、`_fake`、扩展名、256×256、mode、解码、dtype、范围、全黑/全白异常和 stem 对应。报告写入 `artifacts/submission_report.json` 与 `artifacts/submission_files.csv`。

ZIP 中成员必须从 `results/` 开始，例如 `results/test/CD68/00000_fake.jpg`。不能出现 `project/results/...` 或最外层 `submission_CD68/results/...`。

## Smoke 与正式提交

`smoke_test_manifest.csv` 来自 held-out validation 的 DAPI-only 引用，只验证软件链路。smoke ZIP 位于 `outputs/<run>/smoke_submission`，不得上传。正式 `test_manifest.csv` 为空时，predict、make-submission 和 validator 必须明确失败。

