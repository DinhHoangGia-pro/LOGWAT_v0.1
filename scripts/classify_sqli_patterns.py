#!/usr/bin/env python3
import re, os
from collections import defaultdict, Counter

IN_FILE = os.path.join(os.getcwd(), 'data', 'sqli_unique_samples.txt')
OUT_FILE = os.path.join(os.getcwd(), 'data', 'sqli_pattern_groups.txt')
if not os.path.exists(IN_FILE):
    raise SystemExit('Input file not found: '+IN_FILE)

patterns = {
    'UNION-based': re.compile(r'\bunion\b', re.I),
    'Boolean-based': re.compile(r"(or\s+\d+\s*=\s*\d+|and\s+\d+\s*=\s*\d+|\'\s*=\s*\'|=\s*'\s*'|\bor\s+1\s*=\s*1\b|\band\s+1\s*=\s*1\b)", re.I),
    'Time-based/blind': re.compile(r'sleep\s*\(|benchmark\s*\(', re.I),
    'Error-based': re.compile(r'extractvalue\s*\(|updatexml\s*\(|sqlerror|sql syntax|mysql_fetch|warning:|error in your sql', re.I)
}

groups = defaultdict(list)
full = []
with open(IN_FILE, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 1)
        if len(parts) == 2:
            uid, content = parts
        else:
            uid = parts[0]
            content = ''
        s = content.lower()
        assigned = None
        for name, pat in patterns.items():
            if pat.search(s):
                assigned = name
                break
        if assigned is None:
            assigned = 'Stacked/other'
        groups[assigned].append((uid, content))
        full.append((uid, assigned, content))

# counts
counts = {k: len(v) for k,v in groups.items()}
# determine largest group
largest_group = max(groups.items(), key=lambda kv: len(kv[1]))

# print counts
print('Pattern classification counts:')
for k in sorted(counts.keys(), key=lambda x: -counts[x]):
    print(f" - {k}: {counts[k]}")

# print 10 samples from largest group
lg_name, lg_list = largest_group
print('\nLargest group:', lg_name, 'count=', len(lg_list))
print('First 10 samples:')
for uid, content in lg_list[:10]:
    print(uid + '\t' + content[:400])

# write full classification
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('source_uid\tgroup\tcontent\n')
    for uid, group, content in full:
        # escape newlines in content
        content2 = content.replace('\n',' ').replace('\r',' ')
        f.write(f"{uid}\t{group}\t{content2}\n")

print('\nWrote full classification to', OUT_FILE)
