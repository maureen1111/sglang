import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Tuple

import torch


TensorCopyItem = Tuple[torch.Tensor, torch.Tensor, int]


def iter_chunked_tensor_views(
    dst: torch.Tensor,
    src: torch.Tensor,
    num_bytes: int,
    chunk_bytes: int,
):
    if (
        chunk_bytes <= 0
        or num_bytes <= chunk_bytes
        or dst.numel() != src.numel()
        or not dst.is_contiguous()
        or not src.is_contiguous()
    ):
        yield dst, src, num_bytes
        return

    chunk_elements = max(1, chunk_bytes // dst.element_size())
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    for start in range(0, dst_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, dst_flat.numel())
        yield (
            dst_flat[start:end],
            src_flat[start:end],
            (end - start) * dst.element_size(),
        )


@dataclass
class TailCopyJob:
    owner: Any
    target_index: int
    fork_event: torch.cuda.Event
    copy_items: Tuple[TensorCopyItem, ...]
    next_item: int = 0


class TailCopyScheduler:
    """Pace next-forward H2D copies so they do not monopolize PCIe."""

    def __init__(self, device, copy_stream: torch.cuda.Stream):
        self.device = device
        self.copy_stream = copy_stream
        self._condition = threading.Condition()
        self._jobs: Deque[TailCopyJob] = deque()
        self._thread = threading.Thread(
            target=self._pump,
            name="sglang-offload-tail-copy",
            daemon=True,
        )
        self._thread.start()

    def submit(self, job: TailCopyJob):
        with self._condition:
            self._jobs.append(job)
            self._condition.notify()

    def _pop_ready_job(self):
        ready: List[Tuple[int, int, TailCopyJob]] = []
        for position, job in enumerate(self._jobs):
            if job.fork_event.query():
                ready.append((job.target_index, position, job))
        if not ready:
            return None
        _, position, job = min(ready)
        del self._jobs[position]
        return job

    def _pump(self):
        torch.cuda.set_device(self.device)
        while True:
            with self._condition:
                while not self._jobs:
                    self._condition.wait()
                try:
                    job = self._pop_ready_job()
                except Exception as error:
                    self._fail_all_jobs(error)
                    continue
                if job is None:
                    self._condition.wait(timeout=0.0005)
                    continue

            self._copy_one_chunk(job)

            with self._condition:
                if job.next_item < len(job.copy_items):
                    self._jobs.append(job)
                else:
                    self._complete(job)
                self._condition.notify()

    def _copy_one_chunk(self, job: TailCopyJob):
        try:
            dst, src, _ = job.copy_items[job.next_item]
            with torch.cuda.stream(self.copy_stream):
                dst.copy_(src, non_blocking=True)
                chunk_done = torch.cuda.Event()
                chunk_done.record(self.copy_stream)
            chunk_done.synchronize()
            job.next_item += 1
        except Exception as error:
            job.owner._copy_thread_error = error
            job.next_item = len(job.copy_items)

    def _complete(self, job: TailCopyJob):
        try:
            with torch.cuda.stream(self.copy_stream):
                job.owner._load_event.record(self.copy_stream)
        except Exception as error:
            job.owner._copy_thread_error = error
        finally:
            job.owner._load_event_recorded.set()

    def _fail_all_jobs(self, error: Exception):
        while self._jobs:
            job = self._jobs.popleft()
            job.owner._copy_thread_error = error
            job.owner._load_event_recorded.set()
