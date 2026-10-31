"""Protocol 6: seeded sample of 80 from stratum-R stage-1 inclusions."""
import csv, random, os
H = os.path.dirname(os.path.abspath(__file__))
c = {r["rec_id"]: r for r in csv.DictReader(open(os.path.join(H, "candidates.csv")))}
inc = []
for b in (1, 2, 3):
    inc += [r["rec_id"] for r in csv.DictReader(open(os.path.join(H, f"stage1/stage1_decisions_batch{b}.csv"))) if r["decision"] == "include"]
R = sorted(i for i in inc if c[i]["stratum"] == "R"); Cn = sorted(i for i in inc if c[i]["stratum"] == "C")
samp = sorted(random.Random(20261028).sample(R, 80)) if len(R) > 80 else R
with open(os.path.join(H, "fulltext_list.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rec_id", "stratum", "arxiv_id", "oa_pdf", "title"])
    for i in Cn + samp:
        r = c[i]; oa = r["oa_pdf"]
        if i == "Hc7a552ee": oa = "https://www.nature.com/articles/s41586-024-08328-6.pdf"
        w.writerow([i, r["stratum"], r["arxiv_id"], oa, r["title"]])
print("C", len(Cn), "R eligible", len(R), "R sampled", len(samp))
