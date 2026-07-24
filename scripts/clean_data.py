"""Add headers to the raw WDBC data.csv and sanity-check it, producing data_clean.csv."""
import csv
import sys

FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
STATS = ["mean", "se", "worst"]

COLUMNS = ["id", "diagnosis"] + [
    f"{feat}_{stat}" for stat in STATS for feat in FEATURES
]


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def check(rows):
    errors = []
    if any(len(r) != len(COLUMNS) for r in rows):
        errors.append("inconsistent column count")

    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")

    feat_rows = [tuple(r[1:]) for r in rows]
    if len(feat_rows) != len(set(feat_rows)):
        errors.append("duplicate feature rows")

    for r in rows:
        if r[1] not in ("M", "B"):
            errors.append(f"unexpected diagnosis value: {r[1]!r} (id={r[0]})")
        for v in r[2:]:
            try:
                float(v)
            except ValueError:
                errors.append(f"non-numeric feature value: {v!r} (id={r[0]})")
        if any(v == "" for v in r):
            errors.append(f"empty cell (id={r[0]})")

    return errors


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data_clean.csv"

    rows = load_rows(src)
    errors = check(rows)
    if errors:
        print(f"Found {len(errors)} issue(s):")
        for e in errors[:20]:
            print(" -", e)
        sys.exit(1)

    diag_counts = {"M": 0, "B": 0}
    for r in rows:
        diag_counts[r[1]] += 1

    with open(dst, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    print(f"OK: {len(rows)} rows, {len(COLUMNS)} columns, no missing/duplicate/invalid values.")
    print(f"Class balance: M={diag_counts['M']} ({diag_counts['M']/len(rows):.1%})"
          f"  B={diag_counts['B']} ({diag_counts['B']/len(rows):.1%})")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
