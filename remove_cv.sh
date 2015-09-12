#!/bin/bash
files=($(ls cv/))

i=1
for f in "${files[@]}"; do
    if ((i > 10)); then
        rm cv/$f
    fi
    ((i=i+1))
done
