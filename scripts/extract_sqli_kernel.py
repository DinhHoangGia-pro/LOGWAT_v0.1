#!/usr/bin/env python3
import re, os
from collections import defaultdict, Counter

IN_FILE = os.path.join(os.getcwd(), 'data', 'sqli_unique_samples.txt')
OUT_FILE = os.path.join(os.getcwd(), 'data', 'sqli_kernel_groups.txt')
if not os.path.exists(IN_FILE):
    raise SystemExit('Input file not found: '+IN_FILE)

# regex to capture core starting at first quote or semicolon indicating injection
core_pat = re.compile(r"[\'\"]\s*;.*?(?=&|$)", re.I | re.S)
# fallback: capture after first semicolon
fallback_pat = re.compile(r";.*?(?=&|$)", re.I | re.S)

# replacement map for common identifiers to placeholders
identifiers = [r'usuarios', r'datos', r'nombre', r'precio', r'dni', r'email', r'ntc', r'id', r'login', r'password', r'b1', r'b2']
ident_re = re.compile(r"\b(" + r"|".join(identifiers) + r")\b", re.I)

kernels = defaultdict(list)
entry_count = 0
with open(IN_FILE, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 1)
        uid = parts[0]
        content = parts[1] if len(parts) > 1 else ''
        entry_count += 1
        m = core_pat.search(content)
        if not m:
            m = fallback_pat.search(content)
        core = m.group(0) if m else ''
        # normalize
        core_n = core.lower().strip()
        # remove leading quotes and semicolons
        core_n = re.sub(r"^[\'\"]\s*;", '', core_n)
        core_n = core_n.strip()
        # replace identifiers
        core_n = ident_re.sub('<ID>', core_n)
        # replace quoted strings content
        core_n = re.sub(r"'[^']*'", "'<STR>'", core_n)
        core_n = re.sub(r'\"[^\"]*\"', '"<STR>"', core_n)
        # replace numbers
        core_n = re.sub(r"\b\d+\b", '<NUM>', core_n)
        # collapse whitespace
        core_n = re.sub(r"\s+", ' ', core_n).strip()
        if core_n == '':
            core_n = '<NO_CORE_FOUND>'
        kernels[core_n].append((uid, content))

# produce counts
kernel_counts = [(k, len(v)) for k,v in kernels.items()]
kernel_counts.sort(key=lambda x: -x[1])

# write output
with open(OUT_FILE, 'w', encoding='utf-8') as out:
    out.write(f"Total input rows: {entry_count}\n")
    out.write(f"Unique kernels: {len(kernel_counts)}\n\n")
    for k,cnt in kernel_counts:
        out.write(f"COUNT={cnt}\tKERNEL={k}\n")
        # list UIDs (up to 20) for inspection
        uids = [uid for uid,_ in kernels[k]]
        out.write('UIDs: ' + ','.join(uids[:20]) + ('\n' if len(uids)<=20 else '...\n'))
    out.write('\n')

# print summary to console
print(f"Processed {entry_count} entries")
print(f"Unique kernels: {len(kernel_counts)}")
print('Top kernels:')
for k,cnt in kernel_counts[:10]:
    print(f" - {cnt} occurrences -> {k}")
print('\nWrote kernel groups to', OUT_FILE)
