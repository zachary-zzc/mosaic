#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test for HaluMem pipeline: build and evaluate on test.jsonl (1-2 users).
"""
import os
import sys
from experiments.halumem._utils import setup_mosaic_path, load_jsonl, DATASET_HALUMEM_DIR

setup_mosaic_path()

def main():
    test_path = os.path.join(DATASET_HALUMEM_DIR, "test.jsonl")
    if not os.path.isfile(test_path):
        print(f"[error] test.jsonl not found: {test_path}")
        sys.exit(1)
    users = load_jsonl(test_path)
    print(f"Loaded {len(users)} users from test.jsonl")
    # TODO: Call graph build + eval for first user
    user = users[0]
    print(f"User UUID: {user.get('uuid')}")
    print(f"Sessions: {len(user.get('sessions', []))}")
    print("[placeholder] Build graph and run evaluation here.")
    print("Quick test complete.")

if __name__ == "__main__":
    main()
