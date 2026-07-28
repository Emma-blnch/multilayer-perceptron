"""Split data_clean.csv into a training set and a validation set, stratified
by diagnosis so both sets keep the same class balance."""
import argparse
import csv
import random
import sys
from pathlib import Path


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)  # transforme chaque ligne du fichier en liste de valeurs (ex: "842302", "M", "17.99", ...])
        header = next(reader)  # stocke à part la première ligne (le header)
        return header, list(reader)  # retourne un tuple (header, rows)


def split(rows, ratio, seed):
    rng = random.Random(seed)
    by_class = {}
    for row in rows:  # regroupe les lignes par "classe" (M ou B)
        by_class.setdefault(row[1], []).append(row)  # si clé existe pas dans dict (row[1], []), la crée avec liste vide

    train, valid = [], []
    for label, class_rows in by_class.items():  # pour chaque classe séparément
        class_rows = class_rows[:]  # copie la liste pour ne pas modif original
        rng.shuffle(class_rows)  # mélange aléatoirement ordre des lignes de classe
        cut = round(len(class_rows) * ratio)  # calcule indice de coupure (pos dans liste mélangée ou on coupe entre train et valid)
        train += class_rows[:cut]  # les "cut" premières lignes vont dans train
        valid += class_rows[cut:]  # le reste dans valid

    rng.shuffle(train)  # remélange pour éviter d'avoir tous les M ou B a la suite
    rng.shuffle(valid)
    return train, valid


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)  # reprend auto la doc du haut du fichier pour --help
    parser.add_argument("csv", nargs="?", default="data/data_clean.csv", help="cleaned dataset to split")
    parser.add_argument("--train-out", default="data/data_train.csv", help="output path for the training set")
    parser.add_argument("--valid-out", default="data/data_valid.csv", help="output path for the validation set")
    parser.add_argument("--ratio", type=float, default=0.8, help="fraction of rows kept for training (default 0.8)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for a repeatable split")
    args = parser.parse_args()

    src = Path(args.csv)
    if not src.exists():
        sys.exit(f"error: {src} not found (run src/clean_data.py first)")
    if not 0 < args.ratio < 1:
        sys.exit("error: --ratio must be between 0 and 1")

    header, rows = load_rows(src)  # charger
    train, valid = split(rows, args.ratio, args.seed)  # séparer

    write_csv(args.train_out, header, train)  # écrire fichier de sortie
    write_csv(args.valid_out, header, valid)  # écrire fichier de sortie

    # fonction locale pour recompter M/B pour affichage dans terminal
    def counts(subset):
        c = {"M": 0, "B": 0}
        for r in subset:
            c[r[1]] += 1
        return c

    print(f"Train: {len(train)} rows {counts(train)} -> {args.train_out}")
    print(f"Valid: {len(valid)} rows {counts(valid)} -> {args.valid_out}")


if __name__ == "__main__":
    main()
