#!/usr/bin/env python3

import sys


def print_usage():
    print("Usage: gen-spec.py <thread count>", file=sys.stderr)


if len(sys.argv) != 2:
    print_usage()
    exit(1)

try:
    thread_count = int(sys.argv[1])
except ValueError:
    print_usage()
    exit(1)

if thread_count < 1:
    print_usage()
    exit(1)

print("LTLSPEC NAME safety := G count(" + ", ".join(f"state[{i}] = Critical" for i in range(thread_count)) + ") <= 1;")
print()

for i in range(thread_count):
    print(f"LTLSPEC NAME liveness#{i} := G (state[{i}] = Entering -> F state[{i}] = Critical);")
