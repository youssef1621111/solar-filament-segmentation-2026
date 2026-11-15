import json

with open("external_reports/all_competition_kernels.json", "r", encoding="utf-8") as f:
    kernels = json.load(f)

for k in kernels:
    title = str(k.get("title", ""))
    author = str(k.get("author", ""))
    if any(c in title.lower() for c in ["0.", "lb", "score", "sota", "baseline", "0_"]):
        print(f"{author:30s} | {title}")
