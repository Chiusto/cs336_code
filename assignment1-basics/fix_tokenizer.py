import pathlib
p = pathlib.Path(r"C:\Users\UBU\Desktop\cs336_code\assignment1-basics\cs336_basics\tokenizer\tokenizer.py")
content = pathlib.Path(r"C:\Users\UBU\Desktop\cs336_code\assignment1-basics\cs336_basics\tokenizer\tokenizer.py.bak").read_text(encoding="utf-8")
# Add rank optimization after token_to_id block
old_init = "        # 处理特殊token"
new_init = """        # 处理特殊token"""
content = content.replace(
    "        self.token_to_id = {token:idx for idx,token in vocab.items()}",
    "        self.token_to_id = {token:idx for idx,token in vocab.items()}\n        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}\n        self.merges_set = set(self.merges)"
)
# Replace encode logic to use rank
content = content.replace(
    "                            best_pair = min(valid, key=self.merges.index)",
    "                            best_pair = min(valid, key=self.merges_rank.get)"
)
content = content.replace(
    "                best_pair = min(valid, key=self.merges.index)",
    "                best_pair = min(valid, key=self.merges_rank.get)"
)
# Also replace membership check for speed (optional)
content = content.replace(
    "                            valid = [p for p in pairs if p in self.merges]",
    "                            valid = [p for p in pairs if p in self.merges_set]"
)
content = content.replace(
    "                valid = [p for p in pairs if p in self.merges]",
    "                valid = [p for p in pairs if p in self.merges_set]"
)
p.write_text(content, encoding="utf-8")
print("patched")
print(p.read_text(encoding="utf-8")[:500])
