#!/bin/bash
for filename in modules/test_cases/*/input; do
    cat "$filename" | python main.py
done