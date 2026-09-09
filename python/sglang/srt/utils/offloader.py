import logging
import os
import threading
from abc import ABC
from typing import Callable, Generator, List, Optional

import torch
from torch.func import functional_call

from sglang.srt.environ import envs
from sglang.srt.distributed.naive_distributed import (
    NaiveDistributed,
    get_naive_distributed,
    set_naive_distributed,
)
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_stream,
)
from sglang.srt.utils import MultiprocessingSerializer, is_pin_memory_available
from sglang.srt.utils.host_shared_memory import (
    HostSharedMemoryManager,
    get_host_shared_memory_manager,
    set_host_shared_memory_manager,
)
from sglang.srt.utils.offload_tail_copy import (
    TailCopyJob,
    TailCopyScheduler,
    iter_chunked_tensor_views,
)

logger = logging.getLogger(__name__)

_OFFLOAD_SLAB_ALIGNMENT_BYTES = 256

_SubmoduleAccessor = Callable[[torch.nn.Module], torch.nn.Module]
_WhitelistParamNamesCreator = Callable[[torch.nn.Module], List[str]]


class BaseOffloader(ABC):
    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return list(all_modules_generator)

    def post_init(self):
        pass

    @property
    def forbid_copy_engine_usage(self):
        return False


class NoopOffloader(BaseOffloader):
    pass


# For simplicity use singleton, but can surely support multi instance
_instance: Optional[BaseOffloader] = NoopOffloader()


def get_offloader():
    assert _instance is not None
    return _instance


def set_offloader(instance: BaseOffloader):
    global _instance
    _instance = instance


def create_offloader(dp_rank: int):
    if get_exec().offload.cpu_offload_gb > 0:
        return OffloaderV1(
            cpu_offload_max_bytes=int(get_exec().offload.cpu_offload_gb * 1024**3)
        )
    if get_exec().offload.offload_group_size > 0:
        assert get_exec().offload.cpu_offload_gb == 0, (
            "V2 offload does not support cpu_offload_gb yet"
        )
        return OffloaderV2(
            group_size=get_exec().offload.offload_group_size,
            num_in_group=get_exec().offload.offload_num_in_group,
            prefetch_step=get_exec().offload.offload_prefetch_step,
            mode=get_exec().offload.offload_mode,
            dp_rank=dp_rank,
            dp_size=get_parallel().dp_size,
        )
    return NoopOffloader()


class OffloaderV1(BaseOffloader):
    def __init__(self, cpu_offload_max_bytes: int):
        self._cpu_offload_bytes = 0
        self._cpu_offload_max_bytes = cpu_offload_max_bytes

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return [self.maybe_offload_to_cpu(module) for module in all_modules_generator]

    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
            return module

        pin_memory = is_pin_memory_available()
        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for p in module.parameters():
            if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # `torch.empty_like` does not support `pin_memory` argument
            cpu_data = torch.empty_strided(
                size=p.data.size(),
                stride=p.data.stride(),
                dtype=p.data.dtype,
                layout=p.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(p.data)
            p.data = cpu_data
            self._cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device, it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }
                output = functional_call(module, device_state, args=args, kwargs=kwargs)
                module.forward = forward
                return output

            module.forward = forward

        return module


class OffloaderV2(BaseOffloader):
    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        mode: str,
        dp_rank: int,
        dp_size: int,
    ):
        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.mode = mode

        run_id = os.environ["SGLANG_RUN_ID"]

        # Temporarily init inside Offloader, can move if other modules also need this
        if self.mode in {"sharded_gpu", "shm_cpu"}:
            assert get_parallel().tp_size == 1, "not yet support tp_size!=1"
            set_naive_distributed(
                NaiveDistributed(
                    rank=dp_rank,
                    world_size=dp_size,
                    rendezvous=f"/tmp/{run_id}",
                )
            )
        if self.mode in {"shm_cpu"}:
            set_host_shared_memory_manager(
                HostSharedMemoryManager(
                    base_name=run_id,
                )
            )

        self.offloaders = []

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        assert len(self.offloaders) == 0, "should only call wrap_modules once"

        # The offloader's async prefetch/offload copies run on their own
        # stream — sharing the models' "alt" overlap stream would serialize
        # unrelated copy and compute work.
        alt_stream = get_stream("offload")
        tail_copy_scheduler = None
        if envs.SGLANG_OFFLOAD_PACED_TAIL_COPY.get():
            tail_copy_scheduler = TailCopyScheduler(
                device=torch.cuda.current_device(), copy_stream=alt_stream
            )

        all_modules = []
        offload_submodules = []
        for module_index, module in enumerate(all_modules_generator):
            all_modules.append(module)
            if module_index % self.group_size >= self.group_size - self.num_in_group:
                submodule = submodule_accessor(module)
                whitelist_param_names = whitelist_param_names_creator(submodule)
                logger.info(
                    f"[offloader] offload {module_index=} submodule={type(submodule)} params={whitelist_param_names} memory_allocated={torch.cuda.memory_allocated()}"
                )
                offload_submodules.append(submodule)
                self.offloaders.append(
                    _ModuleOffloader(
                        mode=self.mode,
                        module=submodule,
                        alt_stream=alt_stream,
                        whitelist_param_names=whitelist_param_names,
                        module_index=len(self.offloaders),
                        tail_copy_scheduler=tail_copy_scheduler,
                    )
                )

        prefetch_after = _build_slot_prefetch_schedule(
            len(offload_submodules), self.prefetch_step
        )
        for index, module in enumerate(offload_submodules):
            _hook_module_forward_for_offloader(
                index=index,
                module=module,
                offloaders=self.offloaders,
                next_index=prefetch_after[index],
            )

        return all_modules

    def post_init(self):
        for offloader in self.offloaders:
            offloader.post_init()

        use_direct_static_prefetch = (
            self.mode == "cpu"
            and envs.SGLANG_OFFLOAD_DIRECT_STATIC_PREFETCH.get()
        )
        if self.mode == "cpu" and (
            envs.SGLANG_OFFLOAD_STATIC_BUFFER_RING.get()
            or use_direct_static_prefetch
        ):
            self._assign_static_buffer_ring(
                direct_parameter_binding=use_direct_static_prefetch
            )

        for i in range(min(self.prefetch_step, len(self.offloaders))):
            self.offloaders[i].start_onload()

    def _assign_static_buffer_ring(self, direct_parameter_binding: bool = False):
        if not self.offloaders:
            return

        pools = {}
        allocated_bytes = 0
        slot_count = min(self.prefetch_step, len(self.offloaders))
        for index, offloader in enumerate(self.offloaders):
            signature = offloader.parameter_signature()
            key = (signature, index % slot_count)
            device_buffers = pools.get(key)
            if device_buffers is None:
                device_buffers = offloader.allocate_static_device_tensors()
                pools[key] = device_buffers
                device_tensors, device_slab = device_buffers
                allocated_bytes += (
                    device_slab.numel() * device_slab.element_size()
                    if device_slab is not None
                    else sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in device_tensors.values()
                    )
                )
            device_tensors, device_slab = device_buffers
            offloader.assign_static_device_tensors(
                device_tensors,
                device_slab=device_slab,
                direct_parameter_binding=direct_parameter_binding,
            )

        logger.info(
            "[offloader] enabled static GPU buffer ring: modules=%s slots=%s "
            "layouts=%s allocated_bytes=%s direct_parameter_binding=%s",
            len(self.offloaders),
            slot_count,
            len(pools),
            allocated_bytes,
            direct_parameter_binding,
        )

    @property
    def forbid_copy_engine_usage(self):
        return self.mode == "cpu"


def _build_slot_prefetch_schedule(num_offloaders: int, prefetch_step: int):
    """Return the next owner of each static-buffer slot.

    Advancing by ``prefetch_step`` and applying modulo ``num_offloaders`` is
    unsafe when those values are not divisible: it can overwrite a slot still
    needed by a later module in the current forward. Build one circular owner
    chain per slot instead.
    """
    if num_offloaders == 0:
        return []
    slot_count = min(prefetch_step, num_offloaders)
    slot_owners = [[] for _ in range(slot_count)]
    for index in range(num_offloaders):
        slot_owners[index % slot_count].append(index)

    prefetch_after = [None] * num_offloaders
    for owners in slot_owners:
        for owner_index, owner in enumerate(owners):
            prefetch_after[owner] = owners[(owner_index + 1) % len(owners)]
    return prefetch_after


def _hook_module_forward_for_offloader(index, module, offloaders, next_index):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        current_offloader = offloaders[index]
        try:
            if current_offloader.uses_direct_parameter_binding:
                current_offloader.wait_until_loaded()
                output = original_forward(*args, **kwargs)
            else:
                output = functional_call(
                    module,
                    current_offloader.wait_and_get_device_tensors(),
                    args=args,
                    kwargs=kwargs,
                )

            offloaders[next_index].start_onload(
                allow_paced_chunking=(next_index <= index)
            )
            current_offloader.offload()
            return output
        finally:
            module.forward = forward

    module.forward = forward


def _hook_module_forward_raw(module, on_forward_end, get_parameter_and_buffer_dicts):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        output = functional_call(
            module, get_parameter_and_buffer_dicts(), args=args, kwargs=kwargs
        )
        on_forward_end()
        module.forward = forward
        return output

    module.forward = forward


class _ModuleOffloader(ABC):
    def __init__(
        self,
        mode: str,
        module: torch.nn.Module,
        alt_stream: torch.cuda.Stream,
        whitelist_param_names: List[str],
        module_index: int,
        tail_copy_scheduler: Optional[TailCopyScheduler],
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.alt_stream = alt_stream
        self.module_index = module_index
        self.tail_copy_scheduler = tail_copy_scheduler

        assert self.device != torch.device("cpu"), (
            "not handled device=cpu case yet (should skip this tensor)"
        )

        self._device_tensors = None
        self._static_device_tensors = None
        self._cpu_slab = None
        self._static_device_slab = None
        self._slab_layout = None
        self._uses_direct_parameter_binding = False
        self._load_event = None
        self._load_event_recorded = threading.Event()
        self._load_event_recorded.set()
        self._copy_thread_error = None

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names), (
            f"{whitelist_param_names=} {list(param_dict.keys())=}"
        )

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        for name, param_offloader in self._param_offloaders.items():
            param_offloader.post_init()
        if self.mode == "cpu" and envs.SGLANG_OFFLOAD_CONTIGUOUS_SLAB.get():
            self._build_contiguous_cpu_slab()

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    @staticmethod
    def _view_slab_tensor(
        slab: torch.Tensor,
        offset_bytes: int,
        source: torch.Tensor,
    ) -> torch.Tensor:
        storage_bytes = _strided_storage_size_bytes(source)
        byte_view = slab.narrow(0, offset_bytes, storage_bytes)
        typed_storage = byte_view.view(source.dtype)
        return typed_storage.as_strided(source.shape, source.stride())

    def _build_contiguous_cpu_slab(self):
        sources = {
            name: offloader.get_offload_source()
            for name, offloader in self._param_offloaders.items()
        }
        unsupported = [
            name
            for name, tensor in sources.items()
            if any(stride < 0 for stride in tensor.stride())
        ]
        if unsupported:
            logger.warning(
                "[offloader] contiguous slab disabled for %s: negative-stride params=%s",
                type(self.module).__name__,
                unsupported,
            )
            return

        layout = {}
        offset = 0
        for name, source in sources.items():
            offset = self._align_up(
                offset,
                max(_OFFLOAD_SLAB_ALIGNMENT_BYTES, source.element_size()),
            )
            layout[name] = offset
            offset += _strided_storage_size_bytes(source)
        total_bytes = self._align_up(offset, _OFFLOAD_SLAB_ALIGNMENT_BYTES)
        cpu_slab = torch.empty(
            total_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        for name, source in sources.items():
            view = self._view_slab_tensor(cpu_slab, layout[name], source)
            view.copy_(source)
            self._param_offloaders[name].assign_cpu_storage(view)

        self._cpu_slab = cpu_slab
        self._slab_layout = layout
        logger.info(
            "[offloader] packed contiguous CPU slab: module=%s params=%s bytes=%s pinned=%s",
            type(self.module).__name__,
            list(layout),
            total_bytes,
            cpu_slab.is_pinned(),
        )

    def start_onload(self, allow_paced_chunking: bool = False):
        self._load_event_recorded.wait()
        if self._copy_thread_error is not None:
            raise RuntimeError(
                "Paced offload H2D copy failed"
            ) from self._copy_thread_error
        if torch.cuda.is_current_stream_capturing():
            self._device_tensors = self._create_device_tensors()
            self._load_event = None
            return

        use_paced_copy = (
            allow_paced_chunking
            and self.tail_copy_scheduler is not None
            and self._static_device_tensors is not None
        )
        if use_paced_copy:
            self._start_paced_onload()
            return

        self.alt_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.alt_stream):
            self._device_tensors = self._create_device_tensors()
            self._load_event = torch.cuda.Event()
            self._load_event.record()
        self._load_event_recorded.set()

    def _start_paced_onload(self):
        fork_event = torch.cuda.Event()
        fork_event.record(torch.cuda.current_stream())
        self._device_tensors = self._static_device_tensors
        self._load_event = torch.cuda.Event()
        self._copy_thread_error = None
        self._load_event_recorded.clear()

        chunk_bytes = envs.SGLANG_OFFLOAD_TAIL_COPY_CHUNK_MB.get() * 1024**2
        copy_items = []
        if self._static_device_slab is not None:
            copy_items.extend(
                iter_chunked_tensor_views(
                    self._static_device_slab,
                    self._cpu_slab,
                    self._cpu_slab.numel() * self._cpu_slab.element_size(),
                    chunk_bytes,
                )
            )
        else:
            for name, param_offloader in self._param_offloaders.items():
                source = param_offloader.get_offload_source()
                destination = self._static_device_tensors[name]
                copy_items.extend(
                    iter_chunked_tensor_views(
                        destination,
                        source,
                        source.numel() * source.element_size(),
                        chunk_bytes,
                    )
                )

        self.tail_copy_scheduler.submit(
            TailCopyJob(
                owner=self,
                target_index=self.module_index,
                fork_event=fork_event,
                copy_items=tuple(copy_items),
            )
        )

    def offload(self):
        if self._static_device_tensors is None:
            self._device_tensors = None
        self._load_event = None

    def wait_and_get_device_tensors(self):
        self.wait_until_loaded()
        return self._device_tensors

    def wait_until_loaded(self):
        assert self._device_tensors is not None
        self._load_event_recorded.wait()
        if self._copy_thread_error is not None:
            raise RuntimeError(
                "Paced offload H2D copy failed"
            ) from self._copy_thread_error
        if torch.cuda.is_current_stream_capturing():
            if self._load_event is not None:
                self._device_tensors = self._create_device_tensors()
                self._load_event = None
            return
        if self._load_event is not None:
            self._load_event.wait()

    def _create_device_tensors(self):
        if self._static_device_tensors is not None:
            if self._static_device_slab is not None:
                self._static_device_slab.copy_(self._cpu_slab, non_blocking=True)
            else:
                for name, param_offloader in self._param_offloaders.items():
                    self._static_device_tensors[name].copy_(
                        param_offloader.get_offload_source(), non_blocking=True
                    )
            return self._static_device_tensors
        return {k: v.create_device_tensor() for k, v in self._param_offloaders.items()}

    def parameter_signature(self):
        return (
            self._cpu_slab is not None,
            tuple(
                (
                    name,
                    tuple(param_offloader.get_offload_source().shape),
                    tuple(param_offloader.get_offload_source().stride()),
                    param_offloader.get_offload_source().dtype,
                )
                for name, param_offloader in self._param_offloaders.items()
            )
        )

    def allocate_static_device_tensors(self):
        if self._cpu_slab is not None:
            device_slab = torch.empty_like(self._cpu_slab, device=self.device)
            device_tensors = {
                name: self._view_slab_tensor(
                    device_slab,
                    self._slab_layout[name],
                    param_offloader.get_offload_source(),
                )
                for name, param_offloader in self._param_offloaders.items()
            }
            return device_tensors, device_slab
        device_tensors = {
            name: torch.empty_strided(
                size=param_offloader.get_offload_source().size(),
                stride=param_offloader.get_offload_source().stride(),
                dtype=param_offloader.get_offload_source().dtype,
                device=self.device,
            )
            for name, param_offloader in self._param_offloaders.items()
        }
        return device_tensors, None

    def assign_static_device_tensors(
        self,
        device_tensors,
        device_slab=None,
        direct_parameter_binding: bool = False,
    ):
        self._static_device_tensors = device_tensors
        self._static_device_slab = device_slab
        self._uses_direct_parameter_binding = direct_parameter_binding
        if direct_parameter_binding:
            for name, param_offloader in self._param_offloaders.items():
                param_offloader.assign_static_device_tensor(device_tensors[name])

    @property
    def uses_direct_parameter_binding(self):
        return self._uses_direct_parameter_binding


class _BaseParamOffloader(ABC):
    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        return {
            "meta": _MetaParamOffloader,
            "cpu": _CpuParamOffloader,
            "shm_cpu": _ShmCpuParamOffloader,
            "sharded_gpu": _ShardedGpuParamOffloader,
        }[mode](**kwargs)

    def __init__(self, module, param_name):
        self._module = module
        self._param_name = param_name

    @property
    def _param(self):
        return getattr(self._module, self._param_name)

    def post_init(self):
        pass

    def create_device_tensor(self):
        raise NotImplementedError

    def get_offload_source(self):
        return self._param

    def assign_static_device_tensor(self, device_tensor):
        raise NotImplementedError


class _MetaParamOffloader(_BaseParamOffloader):
    """Usually used for debugging."""

    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        _move_param_to_meta(module, param_name)

    def create_device_tensor(self):
        return torch.empty_like(self._param.data, device="cuda")


class _CpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        _move_param_to_cpu(self._param, pin_memory=True)
        self._cpu_storage = None

    def post_init(self):
        if self._param.device.type != "cpu":
            raise RuntimeError(
                f"Expected offloaded parameter {self._param_name} on CPU before "
                f"static buffer assignment, got {self._param.device}"
            )
        if is_pin_memory_available() and not self._param.is_pinned():
            _move_param_to_cpu(self._param, pin_memory=True)
        self._cpu_storage = self._param.data

    def create_device_tensor(self):
        return self.get_offload_source().to("cuda", non_blocking=True)

    def get_offload_source(self):
        return self._cpu_storage if self._cpu_storage is not None else self._param

    def assign_cpu_storage(self, cpu_storage):
        if cpu_storage.device.type != "cpu":
            raise ValueError("Offload slab views must be CPU tensors")
        self._cpu_storage = cpu_storage
        self._param.data = cpu_storage

    def assign_static_device_tensor(self, device_tensor):
        if self._cpu_storage is None:
            self.post_init()
        self._param.data = device_tensor


class _ShmCpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        self._rank = get_naive_distributed().get_rank()
        self._world_size = get_naive_distributed().get_world_size()

        assert get_parallel().tp_size == 1, "not yet support tp_size!=1"
        assert self._param.data.is_contiguous(), (
            f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"
        )

        self.shm_cpu_data = get_host_shared_memory_manager().malloc(
            shape=self._param.shape, dtype=self._param.dtype
        )

        if self._rank == 0:
            self.shm_cpu_data.copy_(self._param.data.to("cpu"))
            self._param.data = self.shm_cpu_data
        else:
            _move_param_to_meta(self._module, self._param_name)
        get_naive_distributed().barrier()

    def post_init(self):
        if self._rank == 0:
            assert self.shm_cpu_data.data_ptr() == self._param.data.data_ptr(), (
                f"{self.shm_cpu_data.data_ptr()=} {self._param.data.data_ptr()=} {self.shm_cpu_data=} {self._param.data=}"
            )

        _move_param_to_meta(self._module, self._param_name)

    def create_device_tensor(self):
        return self.shm_cpu_data.to("cuda", non_blocking=True)


def update_param(param, new_tensor):
    """Update parameter while keeping properties needed by Offloader (e.g. pinned host memory)."""

    if param.device == new_tensor.device:
        param.data = new_tensor
    else:
        assert param.device == torch.device("cpu"), (
            f"{param.device=} {new_tensor.device=}"
        )
        param.data = _create_cpu_data(new_tensor, pin_memory=True)


def _move_param_to_cpu(param, pin_memory: bool):
    param.data = _create_cpu_data(param.data, pin_memory=pin_memory)


def _create_cpu_data(data, pin_memory: bool):
    cpu_data = _empty_strided_like(
        data,
        device="cpu",
        pin_memory=pin_memory,
    )
    cpu_data.copy_(data)
    return cpu_data


def _move_param_to_meta(module, param_name):
    old_param = getattr(module, param_name)
    old_param_type = type(old_param)

    new_data = old_param.data.to("meta")

    if old_param_type == ModelWeightParameter:
        # manually checked how `w13_weight` and `w2_weight` are constructed
        new_param = ModelWeightParameter(
            data=new_data,
            **{
                k: getattr(old_param, k)
                for k in ["input_dim", "output_dim", "weight_loader"]
            },
        )
    elif old_param_type == torch.nn.Parameter:
        new_param = torch.nn.Parameter(
            data=new_data,
            requires_grad=False,
        )
        if hasattr(old_param, "weight_loader"):
            new_param.weight_loader = old_param.weight_loader
        else:
            new_param.weight_loader = lambda *args, **kwargs: None
    else:
        raise ValueError(f"Unknown {old_param_type=} {old_param=}")

    setattr(module, param_name, new_param)


def _empty_strided_like(x: torch.Tensor, device, pin_memory=False):
    return torch.empty_strided(
        size=x.size(),
        stride=x.stride(),
        dtype=x.dtype,
        layout=x.layout,
        device=device,
        pin_memory=pin_memory,
    )


def _strided_storage_size_bytes(tensor: torch.Tensor) -> int:
    """Return the storage span needed for a positive-stride tensor view."""
    if tensor.numel() == 0:
        return 0
    if tensor.ndim == 0:
        return tensor.element_size()
    storage_elements = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride())
    )
    return storage_elements * tensor.element_size()


# ----------------------------------------- ShardedGpu ------------------------------------------------------


# TODO unify with ShmCpu mode
class _ShardedGpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        self._rank = get_naive_distributed().get_rank()
        self._world_size = get_naive_distributed().get_world_size()

        assert get_parallel().tp_size == 1, "not yet support tp_size!=1"
        assert self._param.data.is_contiguous(), (
            f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"
        )

        if self._rank == 0:
            _move_param_to_cpu(self._param, pin_memory=True)
        else:
            _move_param_to_meta(self._module, self._param_name)

        self.sharded_param_handles = None

    def post_init(self):
        # check again since it may be changed
        assert self._param.data.is_contiguous(), (
            f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"
        )

        scatter_src = self._param.data

        logger.info(
            f"[offloader] post_init {scatter_src.nbytes=} {scatter_src.dtype=} {scatter_src.shape=} {torch.cuda.memory_allocated()=}"
        )

        if self._rank == 0:
            scatter_src = scatter_src.to("cuda")
        scatter_list = _even_chunk(scatter_src, self._world_size)

        sharded_param = torch.empty(
            scatter_list[0].shape, dtype=scatter_list[0].dtype, device="cuda"
        )
        self.sharded_param_handles = _create_shared_buffer_tensors(
            local_tensor=sharded_param
        )

        get_naive_distributed().scatter(
            sharded_param, scatter_list if self._rank == 0 else None
        )

        _move_param_to_meta(self._module, self._param_name)

    def create_device_tensor(self):
        output = _empty_strided_like(self._param, device="cuda")
        output_chunks = output.chunk(self._world_size)

        for index in range(self._world_size):
            src_rank = (self._rank + index) % self._world_size
            src_buf = self.sharded_param_handles[src_rank]
            output_chunks[src_rank].copy_(src_buf)

        return output


def _even_chunk(x: torch.Tensor, chunks: int):
    assert x.shape[0] % chunks == 0, f"{x.shape=} {chunks=}"
    return list(x.chunk(chunks))


def _create_shared_buffer_tensors(local_tensor: torch.Tensor) -> List[torch.Tensor]:
    self_rank = get_naive_distributed().get_rank()
    world_size = get_naive_distributed().get_world_size()

    object_list = get_naive_distributed().all_gather_object(
        dict(
            dup_serialized_local_tensor=[
                (
                    None
                    if interesting_rank == self_rank
                    else MultiprocessingSerializer.serialize(local_tensor)
                )
                for interesting_rank in range(world_size)
            ]
        )
    )

    output_tensors = []
    for output_rank in range(world_size):
        remote_serialized_tensor = object_list[output_rank][
            "dup_serialized_local_tensor"
        ][self_rank]
        if output_rank == self_rank:
            assert remote_serialized_tensor is None
            output_tensors.append(local_tensor)
        else:
            output_tensors.append(
                MultiprocessingSerializer.deserialize(remote_serialized_tensor)
            )

    return output_tensors
