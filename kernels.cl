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
  uint buf[4];
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

// total: queries * t
__kernel void compute_endpoints(
    const __global ulong *queries,
    __global ulong3 *out,
    ulong offset,
    ulong hi,
    ulong num_strings,
    int chain_len,
    int table_index,
    __constant uint* alphabet,
    int alphabet_size
    /*,__global ulong *dbg*/
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

  int start_position = id / chain_len;
  int query_idx = id % chain_len;

  ulong start = queries[query_idx];
  uint hash[HASH_SIZE];
  ulong end = construct_chain_from_value(&params, start, 0, chain_len, hash);
  out[id - offset] = (ulong3){end, query_idx, start_position};
}
