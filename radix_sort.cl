#line 2 "radix_sort.cl"
__kernel void write_bits(
    const __global T *ary,
    uint size,
    int b,
    __global uint *bits,
    uint offset
)
{
  uint idx = offset + get_global_id(0);
  bits[idx] = idx < size ? (1-GET_BIT(ary[idx], b)) : 0;
}

__kernel void partition(
    const __global T *in,
    __global T *out,
    uint size,
    const __global uint *count,
    uint offset
)
{
  uint lidx = get_local_id(0), gidx = offset + get_global_id(0);
  uint lo = gidx - lidx, hi = lo + LOCAL_SIZE;
  uint total_zeroes = count[size];
  uint zeroes_start = count[lo];
  uint zeroes_group = count[hi] - zeroes_start;
  __local T part_zeroes[LOCAL_SIZE];
  __local T part_ones[LOCAL_SIZE];
  int bit = 1 - count[gidx + 1] + count[gidx];
  uint zeroes_prefix = count[gidx] - zeroes_start;
  if (gidx < size) {
    if (bit) {
      part_ones[lidx - zeroes_prefix] = in[gidx];
    } else {
      part_zeroes[zeroes_prefix] = in[gidx];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidx < zeroes_group) {
    gidx = zeroes_start + lidx;
    if (gidx < size)
      out[gidx] = part_zeroes[lidx];
  } else {
    lidx -= zeroes_group;
    gidx = total_zeroes + lo - zeroes_start + lidx;
    if (gidx < size)
      out[gidx] = part_ones[lidx];
  }
}

__kernel void Scan_Naive(
    const __global uint* inArray,
    __global uint* outArray,
    uint N, uint offset)
{
  uint gidx = get_global_id(0);
  if (gidx < offset)
    outArray[gidx] = inArray[gidx];
  else
    outArray[gidx] = inArray[gidx] + inArray[gidx - offset];
}
