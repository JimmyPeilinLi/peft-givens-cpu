#!/usr/bin/env bash
set -e
pytest -q tests
echo "✔️  all pytest cases passed"
