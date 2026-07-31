"""实验03：Pinned Memory、异步H2D与双Stream流水。

Pinned host memory使DMA传输不必先复制到临时页锁定缓冲区；
``non_blocking=True``允许CPU在拷贝完成前继续提交工作。两条非默认stream
分别处理交错chunk，使H2D与Kernel具备重叠机会。

“具备机会”不等于“一定加速”：消费级GPU、WDDM、chunk大小和copy engine
数量都会影响结果，所以本实验同时输出串行和双stream实测，并只断言数值正确。
"""

from __future__ import annotations

from statistics import median

import torch

from cudarepo.kernels import get_kernels


def run_pipeline(
    host_chunks: list[torch.Tensor],
    streams: list[torch.cuda.Stream],
) -> tuple[float, list[torch.Tensor]]:
    kernels = get_kernels()
    default_stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    outputs: list[torch.Tensor | None] = [None] * len(host_chunks)

    start.record(default_stream)
    for stream in streams:
        stream.wait_event(start)

    for index, host_chunk in enumerate(host_chunks):
        stream = streams[index % len(streams)]
        with torch.cuda.stream(stream):
            device_chunk = host_chunk.to("cuda", non_blocking=True)
            outputs[index] = kernels.vector_add(device_chunk, device_chunk)

    for stream in streams:
        default_stream.wait_stream(stream)
    end.record(default_stream)
    end.synchronize()
    return start.elapsed_time(end), [value for value in outputs if value is not None]


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA不可用。")
        return

    # 4 x 16 MiB，足以让传输与Kernel在时间线上可见，同时适配4GB显存。
    elements_per_chunk = 4 * 1024 * 1024
    host_chunks = [
        torch.randn(elements_per_chunk, dtype=torch.float32, pin_memory=True)
        for _ in range(4)
    ]
    default_stream = torch.cuda.current_stream()
    worker_streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    # 第一次运行用于NVRTC、context和allocator预热。
    run_pipeline(host_chunks, [default_stream])
    run_pipeline(host_chunks, worker_streams)

    serial_samples = [run_pipeline(host_chunks, [default_stream])[0] for _ in range(5)]
    parallel_outputs: list[torch.Tensor] = []
    parallel_samples: list[float] = []
    for _ in range(5):
        elapsed, parallel_outputs = run_pipeline(host_chunks, worker_streams)
        parallel_samples.append(elapsed)

    for host, output in zip(host_chunks, parallel_outputs, strict=True):
        torch.testing.assert_close(output.cpu(), host * 2)

    print(f"serial p50:       {median(serial_samples):.3f} ms")
    print(f"two-stream p50:   {median(parallel_samples):.3f} ms")
    print("PASS: pinned H2D、stream依赖和自定义Kernel结果均正确。")
    print("注意：是否获得重叠收益必须以当前机器的时间线与实测为准。")


if __name__ == "__main__":
    main()

