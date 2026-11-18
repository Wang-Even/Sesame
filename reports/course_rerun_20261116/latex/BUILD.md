# LaTeX 报告编译说明

主文件为 `report.tex`，可直接编辑。正文包含 13 张结果/配置表和 9 张图，数学公式使用 LaTeX 数学环境。实验报告日期为 2026-11-16，LaTeX 导出日期为 2026-11-18。

插图按 `paper_style/report_overrides.json` 替换为论文样式的高分辨率 PNG，保留 LaTeX 包内的原图路径。图注与派生的 `source_report.md` 保持一致；原始 `report.md` 不变。

## 绘图样式与平滑说明

Continuous post-startup curves without markers or raw overlays. Every run is cropped at its first valid update log and time is rebased to that timestamp. EMA half-life remains 131072 real steps, calculated before cropping. Baseline training curves show only the three-seed mean; final-test figures retain sample SD. Baseline training comparisons use environment steps only. Raw data and all test/table values remain unchanged; cost tables include startup.

导出器只复制图像与数据，不计算或修改实测值，也不执行平滑。`content_manifest.json` 记录覆盖配置、实际文件来源及 SHA256；附带文件保留原相对目录。

## Overleaf

上传整个源码 ZIP，主文件选择 `report.tex`，编译器选择 **XeLaTeX**。正文使用 `ctexart` 和 TeX Live 自带的 Fandol 中文字体，不依赖服务器字体路径。保留全部子目录即可找到图片和附带文件。

## 本地 TeX Live / MiKTeX

在本目录执行两次：

```bash
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

也可使用 `latexmk -xelatex report.tex`。需安装中文 LaTeX 支持、Fandol 字体及常用数学、表格、插图和超链接宏包。

项目提供 Tectonic 0.17.0 编译入口。从 Sesame 项目根目录可执行：

```bash
.runtime/latex_toolchain/bin/tectonic-local --untrusted --outdir reports/course_rerun_20261116/latex reports/course_rerun_20261116/latex/report.tex
```

## 项目内更新

`python3 scripts/export_course_report_latex.py --source reports/course_rerun_20261116 --output reports/course_rerun_20261116/latex --paper-style` 会根据原 Markdown 报告重新生成本目录的 `report.tex`，因此会覆盖对生成稿的直接编辑。仅重新编译已有 `.tex` 时无需运行该脚本。修改 LaTeX 后应重新编译 PDF。

`content_manifest.json` 保存转换时的原表格、图片与哈希；CSV 为原报告的实测数据。正文中的 `runs/`、`datasets/`、`scripts/` 等实验路径均相对 Sesame 项目根目录解释。`source_report.md` 是图注已替换的派生稿，不是编译入口。省略 `--paper-style` 会恢复默认的原报告图像与图注。
