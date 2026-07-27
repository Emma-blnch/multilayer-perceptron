#!/bin/sh
# Remove the generated split/normalization files (data_train.csv, data_valid.csv,
# normalization.json), so they can be regenerated from scratch.

FILES="data/data_train.csv data/data_valid.csv data/normalization.json"

for f in $FILES; do
    if [ -f "$f" ]; then
        rm "$f"
        echo "removed $f"
    else
        echo "skipped $f (not found)"
    fi
done
