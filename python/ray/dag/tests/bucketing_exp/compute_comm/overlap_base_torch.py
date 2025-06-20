import torch
import torch.distributed as dist
import os
import time

def run_overlap_exp(tensor, res):
    
    allreduce_stream = torch.cuda.Stream()
    
    # torch.cuda.profiler.start()
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.matmul(tensor, res)
        
    with torch.cuda.stream(allreduce_stream):
        start_event.record(allreduce_stream)
        dist.all_reduce(res, async_op=True)
        end_event.record(allreduce_stream)
        
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"All-reduce operation took {elapsed_time_ms} ms")
    end_time = time.perf_counter()

    # torch.cuda.profiler.stop()
    
    total_time = end_time - start_time

    return total_time

if __name__ == '__main__':
    iterations = 5

    # Read environment variables
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

    tensor = torch.randn(100000, 1000, device="cuda:%d" % LOCAL_RANK)
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
        run_overlap_exp(tensor, res)

    total_time = 0
    for i in range(iterations):
        exec_time = run_overlap_exp(tensor, res)
        total_time += exec_time

    avg_time = total_time / iterations

    print(f"Average execution time over {iterations} runs: {avg_time:.4f} seconds")

    dist.destroy_process_group()