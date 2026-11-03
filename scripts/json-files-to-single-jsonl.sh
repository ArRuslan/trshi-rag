#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments." >&2
    echo "Usage: $0 <input-dir> <output-file>" >&2
    exit 1
fi

jq -c '.[]' "$1"/*.json >"$2"
