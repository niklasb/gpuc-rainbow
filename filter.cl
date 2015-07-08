#line 2 "filter.cl"
__kernel void set_flags(
    const __global T* ary,
    __global uint* flags,
    uint N)
{
  uint i = get_global_id(0);
  if (i > N)
    return;
  flags[i] = i < N ? predicate(ary, i) : 0;
}

__kernel void compact(
    const __global T* in,
    __global T* out,
    __global uint* flags,
    uint N)
{
  uint i = get_global_id(0);
  if (i >= N)
    return;
  uint bit = flags[i+1] - flags[i];
  if (bit)
    out[flags[i]] = in[i];
}
