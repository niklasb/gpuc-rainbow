#line 2 "kernels.cl"

int build_string(__constant uint* alphabet, ulong n, uint* buf);
void hash_from_index(__constant uint* alphabet, ulong idx, uint* hash);
ulong reduce(const uint* hash, ulong round);
ulong construct_chain_from_hash(
    __constant uint* alphabet,
    uint* start_hash,
    int start_iteration,
    int end_iteration
);
ulong construct_chain_from_value(
    __constant uint* alphabet,
    ulong start_value,
    int start_iteration,
    int end_iteration,
    uint* hash
);
ulong rt_lookup(
    const __global ulong2 *rt, ulong lo, ulong hi, ulong endpoint
);

int build_string(__constant uint* alphabet, ulong n, uint* buf)
{
  ulong offset = 0;
  ulong num = 1;
  int len = 0;
  while (offset + num <= n) {
    offset += num;
    num *= ALPHA_SIZE;
    len++;
  }
  n -= offset;
  /*int len = 0;*/
  /*if (n) {*/
    /*len = ceil(log((float)n) / log((float)ALPHA_SIZE));*/
    /*n -= pow((float)ALPHA_SIZE, len - 1);*/
  /*}*/
  uint x = 0;
  int i;
  for (i = 0; i < len; ++i) {
    x|=GETCHAR(alphabet, n % ALPHA_SIZE)<<((i&3)<<3);
    if ((i&3)==3) {
      buf[i>>2] = x;
      x = 0;
    }
    /*PUTCHAR(buf, i, GETCHAR(alphabet, n % ALPHA_SIZE));*/
    n /= ALPHA_SIZE;
  }
  buf[i>>2] = x;
  return len;
}

void hash_from_index(__constant uint* alphabet, ulong idx, uint* hash) {
  uint buf[16];
  for (int i = 0; i < 16; ++i)
    buf[i] = 0;
  int len = build_string(alphabet, idx, buf);
  compute_hash(buf, len, hash);
}

ulong reduce(const uint* hash, ulong round) {
  ulong x = hash[0] | ((ulong)hash[1]<<32);
  x ^= round | (TABLE_INDEX<<32);
  return x % NUM_STRINGS;
}

ulong construct_chain_from_hash(
    __constant uint* alphabet,
    uint* hash,
    int start_iteration,
    int end_iteration)
{
  ulong x;
  for (int i = start_iteration; i < end_iteration; ++i) {
    x = reduce(hash, i);
    hash_from_index(alphabet, x, hash);
  }
  return x;
}

ulong construct_chain_from_value(
    __constant uint* alphabet,
    ulong start_value,
    int start_iteration,
    int end_iteration,
    uint *hash)
{
  hash_from_index(alphabet, start_value, hash);
  if (start_iteration == end_iteration)
    return start_value;
  return construct_chain_from_hash(
      alphabet, hash, start_iteration, end_iteration);
}

__kernel void hash_and_reduce(
    __constant uint* alphabet,
    __global ulong *inout,
    ulong round
    /*,__global uint *dbg*/
    )
{
  uint idx = get_global_id(0);
  ulong start = inout[idx];

  uint hash[HASH_SIZE];
  hash_from_index(alphabet, start, hash);
  inout[idx] = reduce(hash, round);
}

__kernel void generate_chains(
    ulong offset,
    ulong hi,
    __constant uint* alphabet,
    __global ulong2 *out,
    ulong out_offset
    /*,__global ulong *dbg*/
    )
{
  /*ulong start = offset + get_global_id(0);*/
  /*if (start>=hi)return;*/
  /*ulong end = construct_chain_from_value(&params, start, 0, chain_len);*/
  /*out[get_global_id(0)] = (ulong2){end, start};*/
  ulong lo = offset + get_global_id(0) * BLOCK_SIZE;
  uint hash[HASH_SIZE];
  for (ulong start = lo; start < min(hi, lo + BLOCK_SIZE); ++start) {
    /*params.dbg = dbg + start - offset;*/
    ulong end = construct_chain_from_value(
        alphabet, start, 0, CHAIN_LEN, hash);
    /*ulong end = 2;*/
    out[start - offset + out_offset] = (ulong2){end, start};
  }
}

__kernel void compute_endpoints(
    ulong offset,
    ulong hi,
    __constant uint* alphabet,
    const __global uint *queries,
    int num_queries,
    __global ulong4 *out
    /*,__global ulong *dbg*/
    )
{
  ulong id = offset + get_global_id(0);
  if (id >= hi)
    return;
  // TODO can we get rid of the modulus?
  int start_iteration = id / num_queries;
  int query_idx = id % num_queries;
  uint hash[HASH_SIZE];
  for (int i = 0; i < HASH_SIZE; ++i)
    hash[i] = queries[query_idx * HASH_SIZE + i];
  ulong end = construct_chain_from_hash(
      alphabet, hash, start_iteration, CHAIN_LEN);
  // TODO  waste less space
  out[id] = (ulong4){end, start_iteration, query_idx, 0};
}

#define NOT_FOUND (ulong)(-1)

ulong rt_lookup(const __global ulong2 *rt, ulong lo, ulong hi_, ulong endpoint) {
  ulong hi = hi_;
  while (lo < hi) {
    ulong mid = (lo + hi) / 2;
    if (rt[mid].x >= endpoint)
      hi = mid;
    else
      lo = mid + 1;
  }
  return (lo < hi_ && rt[lo].x == endpoint) ? rt[lo].y : NOT_FOUND;
}

__kernel void fill_ulong(
    __global ulong *buf,
    ulong size,
    ulong a,
    ulong b
    )
{
  ulong idx = get_global_id(0);
  if (idx < size)
    buf[idx] = a + b * idx;
}

__kernel void lookup_endpoints(
    ulong offset,
    ulong hi,
    __constant uint* alphabet,
    const __global uint *queries,
    const __global ulong4 *lookup,
    __global ulong *results,
    const __global ulong2 *rt,
    ulong rt_lo, ulong rt_hi
    //,__global ulong *dbg
    )
{
  ulong id = offset + get_global_id(0);
  if (id >= hi)
    return;

  ulong endpoint = lookup[id].x;
  int start_iteration = lookup[id].y;
  int query_idx = lookup[id].z;

  // we assume perfect rainbow table here!
  ulong start = rt_lookup(rt, rt_lo, rt_hi, endpoint);
  if (start != NOT_FOUND) {
    uint hash[HASH_SIZE];
    ulong candidate = construct_chain_from_value(
        alphabet, start, 0, start_iteration, hash);
    uint diff = 0;
    for (int i = 0; i < HASH_SIZE; ++i)
      diff |= hash[i] ^ queries[query_idx * HASH_SIZE + i];
    if (!diff)
      results[query_idx] = candidate;
  }
}
