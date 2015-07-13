import hashlib
import random
import sys
n = int(sys.argv[1])
l = int(sys.argv[2])
alph = sys.argv[3]
def md5(s):
  return hashlib.md5(s).hexdigest()
for _ in range(n):
  print(md5(''.join(random.choice(alph) for _ in range(l))))
