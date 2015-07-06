struct RTParams {
    __constant uint* alphabet;
    int alphabet_size;
    ulong num_strings;
    int chain_len;
    int table_index;
    /*__global ulong *dbg;*/
};
#define PARAMS const struct RTParams*

int build_string(PARAMS params, ulong n, uint* buf);
void hash_from_index(PARAMS params, ulong idx, uint* hash);
ulong reduce(PARAMS params, const uint* hash, ulong round);
ulong construct_chain_from_hash(
    PARAMS params,
    uint* start_hash,
    int start_iteration,
    int end_iteration);
ulong construct_chain_from_value(
    PARAMS params,
    ulong start_value,
    int start_iteration,
    int end_iteration,
    uint* hash);
ulong rt_lookup(const __global ulong2 *rt, ulong lo, ulong hi, ulong endpoint);

int build_string(PARAMS params, ulong n, uint* buf)
{
  ulong base = params->alphabet_size;
  ulong offset = 0;
  ulong num = 1;
  int len = 0;
  while (offset + num <= n) {
    offset += num;
    num *= base;
    len++;
  }
  n -= offset;
  for (int i = 0; i < len; ++i) {
    PUTCHAR(buf, len - i - 1, GETCHAR(params->alphabet, n % base));
    n /= base;
  }
  return len;
}

void hash_from_index(PARAMS params, ulong idx, uint* hash) {
  uint buf[16]; // TODO why does everything break with buf[4]?
  int len = build_string(params, idx, buf);
  compute_hash(buf, len, hash);
  return;
}

ulong reduce(PARAMS params, const uint* hash, ulong round) {
  ulong x = 0;
  for (int i = 0; i < 4*HASH_SIZE; ++i)
    x = (x * 0x100 + GETCHAR(hash, i)) % params->num_strings;
  return (x + round + params->table_index) % params->num_strings;
}

ulong construct_chain_from_hash(
    PARAMS params,
    uint* hash,
    int start_iteration,
    int end_iteration)
{
  ulong x;
  for (int i = start_iteration; i < end_iteration; ++i) {
    x = reduce(params, hash, i);
    hash_from_index(params, x, hash);
  }
  return x;
}

ulong construct_chain_from_value(
    PARAMS params,
    ulong start_value,
    int start_iteration,
    int end_iteration,
    uint *hash)
{
  hash_from_index(params, start_value, hash);
  if (start_iteration == end_iteration)
    return start_value;
  return construct_chain_from_hash(
      params, hash, start_iteration, end_iteration);
}

__kernel void generate_chains(
    ulong offset,
    ulong hi,
    ulong num_strings,
    int chain_len,
    int table_index,
    __constant uint* alphabet,
    int alphabet_size,
    __global ulong2 *out,
    int block_size
    /*,__global ulong *dbg*/
    )
{
  struct RTParams params;
  params.num_strings = num_strings;
  params.chain_len = chain_len;
  params.table_index = table_index;
  params.alphabet = alphabet;
  params.alphabet_size = alphabet_size;

  /*ulong start = offset + get_global_id(0);*/
  /*if (start>=hi)return;*/
  /*ulong end = construct_chain_from_value(&params, start, 0, chain_len);*/
  /*out[get_global_id(0)] = (ulong2){end, start};*/
  ulong lo = offset + get_global_id(0) * block_size;
  uint hash[HASH_SIZE];
  for (ulong start = lo; start < min(hi, lo + block_size); ++start) {
    /*params.dbg = dbg + start - offset;*/
    ulong end = construct_chain_from_value(&params, start, 0, chain_len, hash);
    /*ulong end = 2;*/
    out[start - offset] = (ulong2){end, start};
  }
}

__kernel void compute_endpoints(
    ulong offset,
    ulong hi,
    ulong num_strings,
    int chain_len,
    int table_index,
    __constant uint* alphabet,
    int alphabet_size,
    const __global uint *queries,
    int num_queries,
    __global ulong4 *out
    /*,__global ulong *dbg*/
    )
{
  ulong id = offset + get_global_id(0);
  if (id >= hi)
    return;

  struct RTParams params;
  params.num_strings = num_strings;
  params.chain_len = chain_len;
  params.table_index = table_index;
  params.alphabet = alphabet;
  params.alphabet_size = alphabet_size;

  int start_iteration = id / num_queries;
  int query_idx = id % num_queries;
  uint hash[HASH_SIZE];
  for (int i = 0; i < HASH_SIZE; ++i)
    hash[i] = queries[query_idx * HASH_SIZE + i];
  ulong end = construct_chain_from_hash(&params, hash, start_iteration, chain_len);
  out[id] = (ulong4){end, start_iteration, query_idx, 0};
}

#define NOT_FOUND (ulong)(-1)

ulong rt_lookup(const __global ulong2 *rt, ulong lo, ulong hi_, ulong endpoint) {
  ulong hi = hi_;
  while (lo < hi) {
    ulong mid = (lo + hi) / 2;
    if (rt[mid][0] >= endpoint)
      hi = mid;
    else
      lo = mid + 1;
  }
  return (lo < hi_ && rt[lo][0] == endpoint) ? rt[lo][1] : NOT_FOUND;
}

__kernel void fill_ulong(
    __global ulong *buf,
    ulong size,
    ulong val
    )
{
  ulong idx = get_global_id(0);
  if (idx < size)
    buf[idx] = val;
}

__kernel void lookup_endpoints(
    ulong offset,
    ulong hi,
    ulong num_strings,
    int chain_len,
    int table_index,
    __constant uint* alphabet,
    int alphabet_size,
    const __global uint *queries,
    const __global ulong4 *lookup,
    __global ulong *results,
    const __global ulong2 *rt,
    ulong rt_lo, ulong rt_hi
    //,__global ulong *dbg
    )
{
  struct RTParams params;
  params.num_strings = num_strings;
  params.chain_len = chain_len;
  params.table_index = table_index;
  params.alphabet = alphabet;
  params.alphabet_size = alphabet_size;

  ulong id = offset + get_global_id(0);
  if (id >= hi)
    return;

  ulong endpoint = lookup[id][0];
  int start_iteration = lookup[id][1];
  int query_idx = lookup[id][2];


  // we assume perfect rainbow table here!
  ulong start = rt_lookup(rt, rt_lo, rt_hi, endpoint);
  if (start != NOT_FOUND) {
    uint hash[HASH_SIZE];
    ulong candidate = construct_chain_from_value(
        &params, start, 0, start_iteration, hash);
    uint diff = 0;
    for (int i = 0; i < HASH_SIZE; ++i)
      diff |= hash[i] ^ queries[query_idx * HASH_SIZE + i];
    if (!diff)
      results[query_idx] = candidate;
  }
}
