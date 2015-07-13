import sys
inf = sys.argv[1]
outf = sys.argv[2]
seen = set()
with open(inf, 'rb') as f:
  with open(outf, 'wb') as g:
    while True:
      a, b = f.read(8), f.read(8)
      if not a:
        break
      if b in seen:
        continue
      seen.add(b)
      g.write(a + b)
