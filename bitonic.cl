#line 2 "bitonic.cl"
void check(__global T *ary, uint a, uint b);
void check(__global T *ary, uint a, uint b) {
  T x = ary[a], y = ary[b];
  if (less(y, x)) {
    ary[a] = y;
    ary[b] = x;
  }
}
__kernel void bitonic_cross(
    __global T *ary,
    uint size,
    uint offset,
    uint i
)
{
  uint item = get_global_id(0) + offset;
  if (item >= size / 2) return;
  uint l = item/i*i*2, r = l + 2*i - 1, idx = i - 1 - item % i;
  uint a = l + idx, b = r - idx;
  if (b >= size) return;
  check(ary, a, b);
}

__kernel void bitonic_inc(
    __global T *ary,
    uint size,
    uint offset,
    uint j
)
{
  uint item = get_global_id(0) + offset;
  if (item >= size / 2) return;
  uint l = item/j*j*2, idx = item % j;
  uint a = l + idx, b = l + j + idx;
  if (b >= size) return;
  check(ary, a, b);
}
