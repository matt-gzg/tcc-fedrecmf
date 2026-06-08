import re
import csv

log_file = "training_centralized.log"
csv_file = "results_centralized.csv"

pattern = re.compile(
    r"RESULT \| h=(\d+) \| reg=([\d.]+) \| lr=([\d.]+) \| iter=(\d+) \| iter_b=(\d+) \| HR20=([\d.]+)"
)

rows = []
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            rows.append({
                "hidden_dim": int(match.group(1)),
                "reg": float(match.group(2)),
                "lr": float(match.group(3)),
                "iter": int(match.group(4)),
                "iter_b": int(match.group(5)),
                "hr20": float(match.group(6))
            })

rows.sort(key=lambda x: x["hr20"], reverse=True)

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["hidden_dim", "reg", "lr", "iter", "iter_b", "hr20"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"{len(rows)} resultados exportados para {csv_file}")