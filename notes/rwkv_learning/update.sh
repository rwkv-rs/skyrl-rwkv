#!/usr/bin/env bash
# 一键编译 RWKV 学习笔记；在项目目录或任意工作目录执行都可以。
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$PROJECT_DIR/main.typ"
OUTPUT_DIR="$PROJECT_DIR/build"
OUTPUT="$OUTPUT_DIR/rwkv_learning.pdf"

if ! command -v typst >/dev/null 2>&1; then
  printf '错误：未找到 typst，请先安装：https://typst.app/open-source/\n' >&2
  exit 127
fi

mkdir -p "$OUTPUT_DIR"

case "${1:-build}" in
  build)
    typst compile --root "$PROJECT_DIR" "$INPUT" "$OUTPUT"
    printf '已更新：%s\n' "$OUTPUT"
    ;;
  watch|-w|--watch)
    exec typst watch --root "$PROJECT_DIR" "$INPUT" "$OUTPUT"
    ;;
  *)
    printf '用法：%s [build|watch]\n' "$0" >&2
    exit 2
    ;;
esac
