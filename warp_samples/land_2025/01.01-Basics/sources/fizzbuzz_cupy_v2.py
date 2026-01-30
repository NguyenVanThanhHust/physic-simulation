
import cupy as np

N = 50_000_000

source_array = np.arange(1, N + 1, dtype=np.int32)

conditions = [
    (source_array % 15 == 0),  # Condition for FizzBuzz (divisible by 3 and 5)
    (source_array % 3 == 0),  # Condition for Fizz
    (source_array % 5 == 0),  # Condition for Buzz
    np.logical_or(source_array % 3 != 0, source_array % 5 != 0),
]

choices = [
    np.full(N, 1337),  # Value for FizzBuzz
    np.full(N, 17),  # Value for Fizz
    np.full(N, 42),  # Value for Buzz
    source_array,
]

output = np.select(conditions, choices)

print(output)
