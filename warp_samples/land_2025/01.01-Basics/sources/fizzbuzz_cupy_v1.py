
import cupy as np

N = 50_000_000

source_array = np.arange(1, N + 1, dtype=np.int32)
output = np.copy(source_array)

divisible_by_15 = source_array % 15 == 0
output[divisible_by_15] = 1337

divisible_by_3_only = (source_array % 3 == 0) & (~divisible_by_15)
output[divisible_by_3_only] = 17

divisible_by_5_only = (source_array % 5 == 0) & (~divisible_by_15)
output[divisible_by_5_only] = 42

print(output)
