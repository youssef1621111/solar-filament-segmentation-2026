import json
import glob
import os

notebooks = [
    "external_reports/prvsiyan/solar-filaments-pq-metric-and-pipeline-explained.ipynb",
    "external_reports/muhammadshoaibaltaf/magfilo-v1-evaluation-aligned-solar-filament-s.ipynb",
    "external_reports/topabaem/filament-v02-submission-final-001.ipynb"
]

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        continue
    print("=" * 60)
    print("FILE:", nb_path)
    print("=" * 60)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for i, c in enumerate(nb["cells"]):
        text = "".join(c.get("source", []))
        if c["cell_type"] == "markdown":
            first_line = text.strip().split("\n")[0] if text.strip() else ""
            print(f"Cell {i} MD: {first_line}")
        elif c["cell_type"] == "code":
            if any(k in text.lower() for k in ["model", "submission", "score", "lb", "train", "post", "ensemble"]):
                first_lines = "\n".join(text.strip().split("\n")[:4])
                print(f"Cell {i} CODE:\n{first_lines}\n---")
