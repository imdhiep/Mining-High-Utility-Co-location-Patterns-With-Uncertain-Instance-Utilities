import sys, json
for fn in sys.argv[1:]:
    for l in open(fn):
        if not l[:3].isalpha() or l.startswith('layout') or l.startswith('cal'): continue
        a = l.split(' ', 4)
        try: r = json.loads(a[4])
        except Exception: continue
        print(a[0], a[2], a[3], r['nT'], r['zerr_top'], 'pt', r['point'][1:], 'lb', r['lb1.645'], 'bf05', r['bfdr0.05'], 'bf10', r['bfdr0.1'], 'eq', r['eq0.5'])
