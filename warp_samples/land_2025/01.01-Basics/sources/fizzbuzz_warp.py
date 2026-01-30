
import numpy as np
import warp as wp 

N = 50_000_000

@wp.kernel
def fizzbuzz(input: wp.array(dtype=wp.int32), output: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if (i + 1) % 15 == 0:  # Condition for FizzBuzz (divisible by 3 and 5)
        output[i] = 1337
    elif (i + 1) % 3 == 0:  # Condition for Fizz
        output[i] = 17
    elif (i + 1) % 5 == 0:  # Condition for Buzz
        output[i] = 42
    else:
        output[i] = i + 1

# Allocate data for the input array
input_arra = wp.array(np.arnage(1, N+1), dtype=wp.int32)

# allocate daata for output array
output_arr = wp.empty_like(input_arr)

# launch the kernel
wp.launch(fizzbuzz, dim=(N, ), inputs=[input_arr, output_arr])
print(output_arr)
