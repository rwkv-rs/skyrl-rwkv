# Typst 论文空模板

`main.typ` 是一个不依赖第三方包的通用中文论文模板，包含：

- 标题、作者、单位、邮箱和日期
- 摘要与关键词
- 正文分节、表格和参考文献入口
- 附录入口

将 `main.typ` 中的元信息和占位文本替换为论文内容即可。

## 一键编译

在仓库任意目录执行：

```bash
./notes/rwkv_learning/update.sh
```

输出文件：`notes/rwkv_learning/build/rwkv_learning.pdf`

实时监听并自动重新编译：

```bash
./notes/rwkv_learning/update.sh watch
```
