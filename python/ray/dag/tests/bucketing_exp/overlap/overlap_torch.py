import torch
import torch.distributed as dist
import os
import time
import argparse

def run_bucket_overlap_exp(num_computes, bucket_size, tensor, res, profile = True):
    
    allreduce_stream = torch.cuda.Stream()
    if profile:
        torch.cuda.profiler.start()
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    ar_futures = []
    for _ in range(num_computes // bucket_size):
        comp_lst = []
        for _ in range(bucket_size):
            res = torch.matmul(tensor, res)
            comp_lst.append(res)
        
        with torch.cuda.stream(allreduce_stream):
            # flatten the buffer
            buf = torch.nn.utils.parameters_to_vector(comp_lst)
            ar_res_fut = dist.all_reduce(buf, async_op=True)
            ar_futures.append(ar_res_fut)

    for ar_future in ar_futures:
        ar_future.wait()
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    if profile:
        torch.cuda.profiler.stop()
    
    total_time = end_time - start_time

    return total_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DAG with configurable bucket size')
    parser.add_argument('--bucket-size', type=int, default=10,
                        help='Bucket size for communication (default: 10)')
    parser.add_argument('--num-computes', type=int, default=1000,
                        help='Total number of compute operations (default: 1000)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations to run (default: 5)')
    args = parser.parse_args()

    num_computes = args.num_computes
    bucket_size = args.bucket_size
    iterations = args.iterations

    # Read environment variables
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

    tensor = torch.randn(1000, 1000, device="cuda:%d" % LOCAL_RANK)
    res = torch.randn(1000, 1, device="cuda:%d" % LOCAL_RANK)

    print('init process group')
    dist.init_process_group(
        backend='nccl',
        world_size=WORLD_SIZE,
        rank=WORLD_RANK
    )
    print('Done init process group')

    # run warmup 
    for _ in range(1):
        run_bucket_overlap_exp(num_computes, bucket_size, tensor, res, profile=False)

    total_time = 0
    for i in range(iterations):
        exec_time = run_bucket_overlap_exp(num_computes, bucket_size, tensor, res)
        total_time += exec_time

    avg_time = total_time / iterations

    if WORLD_RANK == 0:
        print(f"Bucket size: {bucket_size}, Average time: {avg_time:.4f} seconds")

    dist.destroy_process_group()