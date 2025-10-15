def calc_flops(problem_sizes):
    # print('GFlops')
    data = []
    for problem_size in problem_sizes:
        M, N, K = problem_size[0], problem_size[1], problem_size[2]
        total_flops = 2*K*M*N
        # print(problem_size, ':', total_flops * 10**-9) # Convert to GFlops
        data.append((problem_size, total_flops))
    return data

def flops_to_time(compute_data):
    print('Flops time (ms)')
    compute_time = []
    for problem_size, total_flops in compute_data:
        total_flops *= 10**-12 # Convert to TFlops
        total_time = total_flops / 53.45 # Seconds
        print(problem_size, ':', total_time * 1000) # Convert to ms
        compute_time.append((problem_size, total_time))
    return compute_time

def calc_bytes(problem_sizes):
    # print('MB')
    data = []
    for problem_size in problem_sizes:
        M, N, K = problem_size[0], problem_size[1], problem_size[2]
        load_data = (M*K + K*N) * 4 # Floats are 4 bytes
        store_data = (M*N) * 4
        # print(problem_size, ':', load_data * 10**-6, ", ", store_data * 10**-6) # Convert to MB
        data.append((problem_size, load_data, store_data))
    return data

def bytes_to_time(memory_data):
    # print('Bytes time (ms)')
    memory_time = []
    for problem_size, load_data, store_data in memory_data:
        total_data = (load_data + store_data) * 10**-9 # Convert to GB
        total_time = total_data / 360 # Seconds
        # print(problem_size, ':', total_time * 1000) # Convert to ms
        memory_time.append((problem_size, total_time))
    return memory_time

def calc_throughput(flops, compute_time, memory_time):
    print('TFlops/s')
    for t1, t2, t3 in zip(flops, compute_time, memory_time):
        problem_size, f = t1
        _, ct = t2
        _, mt = t3
        t = max(ct, mt)
        f *= 10**-12 # Convert to TFlops
        T = f/t # TFlops/s
        print(problem_size, ':', T)

if __name__ == "__main__":
    problem_sizes = [
        [3072, 3072, 3072],
        [2048, 3072, 3072],
        [1024, 3072, 3072],
        [512, 3072, 3072],
        [256, 3072, 3072],
        [128, 3072, 3072],
        [64, 3072, 3072],
        [32, 3072, 3072],
        [16, 3072, 3072],
    ]
    # 1.1
    compute_data = calc_flops(problem_sizes)
    compute_time = flops_to_time(compute_data)
    print()
    # 1.2
    memory_data = calc_bytes(problem_sizes)
    memory_time = bytes_to_time(memory_data)
    print("Minimum times (ms)")
    for t1, t2 in zip(compute_time, memory_time):
        problem_size, ct = t1
        _, mt = t2
        t = max(ct, mt)
        t *= 1000 # Convert to ms
        print(problem_size, ':', t)
    print()
    # 1.3
    calc_throughput(compute_data, compute_time, memory_time)