# CVPR 2027 LaTeX 项目

使用官方 CVPR 2027 author kit，主论文保持匿名审稿模式 `\usepackage[review]{cvpr}`。

- 官方仓库：https://github.com/cvpr-org/author-kit
- 固定版本：`a2369342011145be8c85e5ad21a9b61f9fa53473`
- 获取日期：2026-11-12
- `cvpr.sty`、`ieeenat_fullname.bst` 保持官方原文件。
- `template/cvpr2027/` 保留完整的官方示例，供查阅。

## 目录结构

```text
main.tex                  主论文入口：标题、章节引用、参考文献
preamble.tex              共用宏包、颜色、命令、定理及图片占位命令
rebuttal.tex              独立编译的匿名作者回复骨架，使用官方 rebuttal 模式
cvpr.sty                  CVPR 版式文件
ieeenat_fullname.bst      BibTeX 参考文献格式文件
refs.bib                  论文参考文献数据库
sec/
  0_abstract.tex
  1_intro.tex
  2_related.tex
  3_method.tex
  4_setup.tex
  5_results.tex
  6_analysis.tex
  7_cost.tex
  8_limitations.tex
  9_conclusion.tex
  X_suppl.tex              补充材料骨架，默认不加载
fig/
  architecture.tex
  transfer.tex
  search.tex
  [fig_arch.pdf]           待提供
  [F1_transfer.pdf]        待提供
  [F2_search.pdf]          待提供
template/cvpr2027/         完整官方原始示例
outputs/                  所有生成的 PDF（旧版本保存在 archive/）
build/                    编译中间文件和日志
.latexmkrc                默认输出目录配置
```

正文内容、数字、参考文献条目和红色 TODO 保留原样；原 `figures/` 路径统一为 `fig/`。
`preamble.tex` 按本论文需要组织，`rebuttal.tex` 按官方文件结构整理为待填写骨架。

## ieeenat_fullname.bst 的作用

它是 **BibTeX 样式文件**，规定参考文献条目的作者姓名、排序、标题、会议/期刊和标点等排版规则。
`refs.bib` 保存文献数据；`ieeenat_fullname.bst` 决定这些数据如何显示。

根目录 `main.tex` 通过以下两行使用它们：

```latex
\bibliographystyle{ieeenat_fullname}
\bibliography{refs}
```

将 `.bst` 放在根目录与官方模板一致，便于本地和 Overleaf 编译时直接找到。

## 编译

在项目根目录运行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error rebuttal.tex
```

`.latexmkrc` 自动将所有生成的 PDF 写入 `outputs/`，将中间文件和日志写入 `build/`。
主论文输出：`outputs/main.pdf`；作者回复输出：`outputs/rebuttal.pdf`。
历史预览保存在 `outputs/draft_preview_approx_template.pdf`，旧版编译保存在 `outputs/archive/`。
`fig/` 中的 PDF 属于输入图片素材，不属于编译输出。

如果编译工具不加载 `.latexmkrc`，请显式添加 `-outdir=outputs -auxdir=build -emulate-aux-dir`。

Overleaf 可上传整个目录，选择 `main.tex` 或 `rebuttal.tex` 为主文件，并使用 pdfLaTeX。
独立查看官方示例时，以 `template/cvpr2027/main.tex` 为主文件编译。
