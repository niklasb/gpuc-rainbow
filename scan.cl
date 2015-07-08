#line 2 "scan.cl"
__kernel void scan_naive(
    const __global T* inArray,
    __global T* outArray, uint N, uint offset)
{
  uint i = get_global_id(0);
  if (i >= N)
    return;
  if (i < offset)
    outArray[i] = inArray[i];
  else
    outArray[i] = combine(inArray[i], inArray[i - offset]);
}

__kernel void shift(
    const __global T* inArray,
    __global T* outArray,
    uint N)
{
  uint i = get_global_id(0);
  outArray[i] = i > 0 && i < N ? inArray[i-1] : IDENTITY;
}

#define UNROLL
#define NUM_BANKS      32
#define NUM_BANKS_LOG    5
#define SIMD_GROUP_SIZE    32

// Bank conflicts
#define AVOID_BANK_CONFLICTS 1
#if AVOID_BANK_CONFLICTS
  #define OFFSET(x) ((x) + ((x) / NUM_BANKS))
#else
  #define OFFSET(x) (x)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void scan(
    __global uint* array, __global uint* higherLevelArray,
    __local uint* localBlock)
{
  uint blockSize = get_local_size(0);
  uint gidx = get_global_id(0) + blockSize * get_group_id(0);
  uint lidx = get_local_id(0);
  localBlock[OFFSET(lidx)] = array[gidx + lidx] + array[gidx + lidx + 1];
  // up sweep
  uint stride;
  for (stride = 2; stride <= blockSize; stride <<= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lidx % stride == stride - 1) {
      localBlock[OFFSET(lidx)] += localBlock[OFFSET(lidx - (stride >> 1))];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidx == 0) {
    higherLevelArray[get_group_id(0)] = localBlock[OFFSET(blockSize - 1)];
    localBlock[OFFSET(blockSize - 1)] = 0;
  }
  // down sweep
  for (stride = blockSize; stride >= 2; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lidx % stride == stride - 1) {
      uint left = localBlock[OFFSET(lidx - (stride >> 1))];
      localBlock[OFFSET(lidx - (stride >> 1))] = localBlock[OFFSET(lidx)];
      localBlock[OFFSET(lidx)] += left;
    }
  }
  array[gidx + lidx + 1] = localBlock[OFFSET(lidx)] + array[gidx + lidx];
  array[gidx + lidx] = localBlock[OFFSET(lidx)];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// removed localBlock because we just use a read with an offset on the CPU side
__kernel void scan_add(__global uint* higherLevelArray, __global uint* array)
{
  array[get_global_id(0)] += higherLevelArray[get_group_id(0) / 2];
}
