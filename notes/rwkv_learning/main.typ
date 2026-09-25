// 通用论文空模板（Typst）
// 将下方元信息和各节中的占位文本替换为论文内容即可。

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.2cm),
  numbering: "1",
)
#set text(
  font: "Noto Serif CJK SC",
  size: 11pt,
  lang: "zh",
)
#set par(
  justify: true,
  leading: 0.7em,
  first-line-indent: 2em,
)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => block(above: 1.4em, below: 0.6em)[
  #text(font: "Noto Sans CJK SC", weight: "bold", size: 15pt)[#it]
]
#show heading.where(level: 2): it => block(above: 1em, below: 0.4em)[
  #text(font: "Noto Sans CJK SC", weight: "bold", size: 12pt)[#it]
]
#show figure.caption: set text(size: 9.5pt)
#show link: underline

// ===== 论文元信息 =====
#let paper-title = "论文标题"
#let paper-author = "作者姓名"
#let paper-affiliation = "单位名称"
#let paper-email = "author@example.com"
#let paper-date = "2026 年  月  日"

// ===== 标题页 =====
#align(center)[
  #v(1.2cm)
  #text(font: "Noto Sans CJK SC", size: 22pt, weight: "bold")[#paper-title]
  #v(0.8cm)
  #text(size: 12pt)[#paper-author]
  #v(0.2cm)
  #text(size: 10pt, fill: rgb("4b5563"))[
    #paper-affiliation \
    #paper-email
  ]
  #v(0.35cm)
  #text(size: 10pt, fill: rgb("4b5563"))[#paper-date]
]

#v(0.8cm)

#block(
  width: 100%,
  inset: (x: 0.4cm, y: 0.25cm),
)[
  #text(font: "Noto Sans CJK SC", weight: "bold")[摘要]
  #h(1em)
  在此填写摘要。摘要应简要说明研究问题、方法、主要结果与结论。
]

#block(
  width: 100%,
  inset: (x: 0.4cm, y: 0.25cm),
)[
  #text(font: "Noto Sans CJK SC", weight: "bold")[关键词]
  #h(1em)
  关键词一；关键词二；关键词三
]

#pagebreak()

// ===== 正文 =====
= 引言

在此填写引言。

== 研究背景

在此填写研究背景与问题定义。

== 研究贡献

在此列出本文的主要贡献。

= 相关工作

在此填写相关工作与文献综述。

= 方法

在此介绍研究方法、模型、数据或实验设置。

== 问题定义

在此填写问题定义和符号约定。

== 方法细节

在此填写方法细节。

= 实验

在此填写实验设置、评价指标和实验结果。

#figure(
  table(
    columns: 3,
    [*方法*], [*指标一*], [*指标二*],
    [方法 A], [--], [--],
    [方法 B], [--], [--],
  ),
  caption: [实验结果表。],
) <tab:results>

= 讨论

在此讨论实验结果、局限性和可能的改进方向。

= 结论

在此总结本文工作，并说明未来工作方向。

// ===== 参考文献 =====
// 将 references.bib 放在本目录后，取消下一行注释即可：
// #bibliography("references.bib")

// ===== 附录（按需保留） =====
// #pagebreak()
// = 附录
// 在此填写附加证明、实验细节或其他材料。
