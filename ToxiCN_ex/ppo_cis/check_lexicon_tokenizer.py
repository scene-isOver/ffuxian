"""验证 ToxiCN 词典文件与指定 tokenizer 的词表是否一致。

背景：
    词典 JSON 的结构是 ``{"中文词": [token_id, ...]}``，而 ``ToxicLexicon``
    只使用 value（id 序列）在 ``input_ids`` 里做子序列匹配，完全不看 key。
    因此一旦 tokenizer 的词表与生成词典时的词表不同，行号就会整体或局部
    错位，tke_bert 的知识特征会失效（匹配不到）或串味（匹配到别的词）。

本脚本做两层检查：
    1. 词表层：不同 tokenizer 的 vocab 字典是否逐项相同；
    2. 编码层：对每个词典词编码，是否得到与词典完全相同的 id 序列。

用法（在 ToxiCN_ex 目录下）：
    python -m ppo_cis.check_lexicon_tokenizer
    python -m ppo_cis.check_lexicon_tokenizer --tokenizers bert-base-chinese,chinese-roberta-wwm-ext
    python -m ppo_cis.check_lexicon_tokenizer --tokenizers D:/models/bert-base-chinese,D:/models/chinese-roberta-wwm-ext

独立脚本：只读数据与模型目录，不修改任何现有文件，不需要时可整体删除。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

from .config import LEXICON_DIR, resolve_model_name
from .data import LEXICON_FILES


# ===================== 词典加载 =====================
# 五个词典文件合并成一张表；同时记录每个词来自哪个文件，便于按文件统计。
def load_lexicon(lexicon_dir: Path) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    entries: Dict[str, List[int]] = {}
    origin: Dict[str, str] = {}
    for filename in LEXICON_FILES:
        path = Path(lexicon_dir) / filename
        if not path.exists():
            print(f"  [skip] {path} 不存在")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for word, ids in data.items():
            entries[word] = [int(t) for t in ids]
            origin[word] = filename
        print(f"  {filename:<14s} {len(data):>6d} 条")
    return entries, origin


# ===================== 单个 tokenizer 的检查 =====================
# 逐词编码并与词典 value 比对；add_special_tokens=False 才能去掉 [CLS]/[SEP]。
def check_tokenizer(name: str, path: str, entries: Dict[str, List[int]],
                    origin: Dict[str, str]):
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:                      # 目录缺文件 / 无分词器
        print(f"[{name}] 加载失败：{exc}")
        return None, {}, []

    print(f"\n[{name}] {path}")
    print(f"  vocab size = {len(tokenizer.get_vocab())}")

    per_file: Dict[str, int] = {f: 0 for f in LEXICON_FILES}
    examples = []
    for word, ids in entries.items():
        got = tokenizer(word, add_special_tokens=False).input_ids
        if got == ids:
            per_file[origin[word]] += 1
        elif len(examples) < 8:
            examples.append((word, ids, got, origin[word]))
    return tokenizer, per_file, examples


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(
        description="检查 ToxiCN 词典的 token id 与 tokenizer 词表是否一致")
    parser.add_argument("--lexicon_dir", default=str(LEXICON_DIR))
    parser.add_argument("--tokenizers", default="bert-base-chinese,chinese-roberta-wwm-ext",
                        help="逗号分隔的模型名或本地目录；第一个作为基准（用于 id 反查）")
    args = parser.parse_args()

    names = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
    print("词典目录：", args.lexicon_dir)
    entries, origin = load_lexicon(Path(args.lexicon_dir))
    print(f"合计 {len(entries)} 条词条\n")

    # 解析每个 tokenizer 的真实目录，失败的先剔除，避免中途中断。
    resolved = []
    for name in names:
        try:
            resolved.append((name, resolve_model_name(name)))
        except FileNotFoundError as exc:
            print(f"[{name}] 跳过：{exc}\n")

    if not resolved:
        print("没有可用的 tokenizer，退出。")
        raise SystemExit(1)

    vocabularies: Dict[str, dict] = {}
    results = []
    for name, path in resolved:
        tokenizer, per_file, examples = check_tokenizer(name, path, entries, origin)
        if tokenizer is None:
            continue
        vocabularies[name] = tokenizer.get_vocab()

        matched = sum(per_file.values())
        total = len(entries)
        print(f"  总匹配 {matched}/{total} ({matched / total:.1%})")
        print("  按文件：")
        for filename in LEXICON_FILES:
            n_file = sum(1 for w, f in origin.items() if f == filename)
            if n_file == 0:
                continue
            print(f"    {filename:<14s} {per_file[filename]:>5d}/{n_file:<5d}"
                  f" {per_file[filename] / n_file:.1%}")
        if examples:
            print("  不匹配示例（词典 id -> tokenizer 编码）：")
            for word, ids, got, filename in examples:
                print(f"    [{filename}] {word!r}: {ids} -> {got}")

        results.append((name, matched, total))

    # 词表逐项比对：以第一个 tokenizer 为基准。
    if len(vocabularies) >= 2:
        base_name = next(iter(vocabularies))
        base_vocab = vocabularies[base_name]
        print(f"\n词表逐项比对（基准 {base_name}，size={len(base_vocab)}）：")
        for name, vocab in vocabularies.items():
            if name == base_name:
                continue
            same = vocab == base_vocab
            diff = [tok for tok, idx in base_vocab.items()
                    if vocab.get(tok) != idx]
            print(f"  {name}: identical={same}  vocab_size={len(vocab)}"
                  f"  差异词数={len(diff)}")
            if diff[:5]:
                print(f"    示例：{diff[:5]}")

    # 反向确认：词典里出现的 id 在基准词表中对应什么字。
    if vocabularies:
        base_name = next(iter(vocabularies))
        inverse = {idx: tok for tok, idx in vocabularies[base_name].items()}
        probe = [4868, 5307, 5770, 1796, 1769, 4567]
        print(f"\n{base_name} 的 id 反查：")
        for idx in probe:
            print(f"  {idx} -> {inverse.get(idx, '<不存在>')}")

    # 结论：所有 tokenizer 都 100% 匹配才算通过。
    print("\n" + "=" * 52)
    all_ok = all(matched == total for _, matched, total in results)
    if all_ok:
        print("结论：词典与所有 tokenizer 完全匹配，tke_bert 的知识特征安全。")
        raise SystemExit(0)
    print("结论：存在不匹配（见上方示例）。若该 tokenizer 要用于 tke_bert，"
          "需要按它的词表重新生成词典。")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
