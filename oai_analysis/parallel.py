"""
Running a project's cases in parallel, on the CPUs and GPUs this process may use, e.g. those of a SLURM job.

The steps of a case fall in three parts: the CPU steps before the networks (preprocessing), the GPU steps
(pipeline.GPU_STEPS) and the CPU steps after them. A part runs in one process and saves its outputs in the case
folder, where the next part, in another process, reads them:
- A pool of CPU worker processes runs the first and the last part. A free worker finishes the earliest case whose
  GPU steps are done, unless few cases wait for the GPUs; then it preprocesses the next case. So the cases started
  first finish first, and idle CPUs preprocess ahead.
- One process per GPU loads the networks once and takes cases from a queue that all GPUs share, one at a time.
  It saves the outputs, and leaves their checks and thumbnails to the CPU worker that finishes the case.
Each process gets an equal share of the CPUs for its threads (ITK, torch, BLAS), so they do not oversubscribe them.
Threads would be simpler than processes, but Python's global interpreter lock does not let them work at the same
time: on a 24-thread CPU, finishing cases in 12 threads was 2x faster than in one, and in 12 processes 7x. A thread
saving outputs while another used the GPU stalled the GPU one, because ITK holds the lock while it writes a file.
"""
import multiprocessing
import os
import pickle
import queue
import sys
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait


def allocated_cpus() -> int:
    """The CPUs this process may use: on Linux, those its CPU affinity allows, e.g. a SLURM job's."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def gpu_devices(device: str) -> list[str | None]:
    """
    The GPUs to use, as CUDA_VISIBLE_DEVICES entries: all visible ones (e.g. a SLURM job's) for "cuda", one for
    "cuda:N". [None] runs the GPU steps on the CPU.
    """
    if not device.startswith("cuda"):
        return [None]
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    return [ids[int(device[5:])]] if device.startswith("cuda:") else ids


def default_workers(cpus: int, gpus: int) -> int:
    """CPU worker processes: one per two CPUs, which did best with two threads each, less one per GPU process."""
    return max(1, cpus // 2 - gpus)


def parts(pipeline) -> tuple[list[str], list[str], list[str]]:
    """The names of the steps before, of, and after the GPU steps."""
    from oai_analysis.pipeline import GPU_STEPS

    names = [step.name for step in pipeline.steps()]
    gpu = [i for i, name in enumerate(names) if name in GPU_STEPS]
    return names[:gpu[0]], names[gpu[0]:gpu[-1] + 1], names[gpu[-1] + 1:]


def run(project, pipeline, case_ids, rerun, run_id, workers, progress, log_write, heartbeat, case_finished):
    """Process the cases with `workers` CPU worker processes and one process per GPU; see the module docstring."""
    cpus, gpus = allocated_cpus(), gpu_devices(pipeline.device)
    workers = min(default_workers(cpus, len(gpus)) if workers == "auto" else workers, len(case_ids))
    threads = max(1, cpus // (workers + len(gpus)))
    progress(f"{workers} CPU worker processes and {len(gpus)} for the GPU steps "
             f"({', '.join(f'GPU {g}' if g is not None else 'on the CPU' for g in gpus)}), {threads} threads each")
    before, _, after = parts(pipeline)
    context = multiprocessing.get_context("spawn")
    messages, tasks, done = context.Queue(), context.Queue(), context.Queue()
    common = (str(project.path), pickle.dumps(pipeline), run_id, sorted(rerun), messages)
    listener = threading.Thread(target=_listen, args=(messages, progress, log_write, heartbeat), daemon=True)
    listener.start()
    gpu_processes = [context.Process(target=_gpu_process, args=(*common, threads, gpu, tasks, done), daemon=True)
                     for gpu in gpus]
    for process in gpu_processes:
        process.start()
    pool = ProcessPoolExecutor(workers, context, initializer=_init, initargs=(*common, threads, "cpu"))
    order = {case_id: i for i, case_id in enumerate(case_ids)}
    to_preprocess, ready, at_gpu, running = list(case_ids), [], 0, {}  # running: future -> (case_id, part)

    def finished(case_id):
        heartbeat.remove(case_id)
        case_finished(case_id)

    try:
        while to_preprocess or ready or running or at_gpu:
            while len(running) < workers and (to_preprocess or ready):
                if ready and (at_gpu >= 2 * len(gpus) or not to_preprocess):
                    case_id = min(ready, key=order.get)
                    ready.remove(case_id)
                    running[pool.submit(_cpu_part, case_id, after)] = (case_id, after)
                else:
                    case_id = to_preprocess.pop(0)
                    heartbeat.add(case_id, project.case_folder(case_id) / "running.json")
                    running[pool.submit(_cpu_part, case_id, before)] = (case_id, before)
            for future in wait(running, timeout=0.5, return_when=FIRST_COMPLETED)[0]:
                case_id, part = running.pop(future)
                if part is before and future.result():
                    tasks.put(case_id)
                    at_gpu += 1
                else:
                    finished(case_id)
            while True:
                try:
                    case_id, ok = done.get_nowait()
                except queue.Empty:
                    break
                at_gpu -= 1
                if ok:
                    ready.append(case_id)
                else:
                    finished(case_id)
            if at_gpu and not all(process.is_alive() for process in gpu_processes):
                raise RuntimeError("A GPU process stopped unexpectedly (see the log). Run again to continue.")
    finally:
        pool.shutdown(cancel_futures=True)
        for _ in gpus:
            tasks.put(None)
        for process in gpu_processes:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
        messages.put(None)
        listener.join()


def _listen(messages, progress, log_write, heartbeat):
    """In the main process: pass the workers' messages on to the run's log and heartbeat."""
    while (message := messages.get()) is not None:
        kind, *args = message
        if kind == "log":
            log_write(*args)
        elif kind == "progress":
            progress(*args)
        else:
            heartbeat.set_step(*args)


# in a worker process

_state = {}  # what _init sets up, for the tasks of this process
_case = threading.local()  # the case the process works on, to say in the log


def _init(project_path, pipeline_bytes, run_id, rerun, messages, threads, kind, gpu=None):
    """Set up a worker process: its CPU share, the GPU it may use, the stages it runs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else "-1"  # before anything starts CUDA
    import itk
    import torch
    from threadpoolctl import threadpool_limits

    from oai_analysis.project import Project

    threadpool_limits(threads)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(threads)
    torch.set_num_threads(threads)
    sys.stdout = sys.stderr = _QueueWriter(messages)
    pipeline = pickle.loads(pipeline_bytes)
    pipeline.device = "cuda" if gpu is not None else "cpu"
    names = parts(pipeline)[1] if kind == "gpu" else [*parts(pipeline)[0], *parts(pipeline)[2]]
    pipeline.setup([step.stage for step in pipeline.steps() if step.name in names])
    _state.update(project=Project(project_path), pipeline=pipeline, run_id=run_id, rerun=set(rerun),
                  messages=messages)


def _gpu_process(project_path, pipeline_bytes, run_id, rerun, messages, threads, gpu, tasks, done):
    """A process for one GPU: takes cases from tasks, one at a time, until it gets None."""
    _init(project_path, pipeline_bytes, run_id, rerun, messages, threads, "gpu", gpu)
    while (case_id := tasks.get()) is not None:
        done.put((case_id, _run_part(case_id, parts(_state["pipeline"])[1], checks=False)))


def _cpu_part(case_id, names) -> bool:
    return _run_part(case_id, names)


def _run_part(case_id, names, checks=True) -> bool:
    """
    Run the named steps of a case, whose earlier steps are done. Returns whether the case can go on. checks=False
    leaves the steps' checks to the next part, which reruns them as it loads the outputs.
    """
    from oai_analysis.project import _CaseRunner

    project, pipeline, messages = _state["project"], _state["pipeline"], _state["messages"]
    steps = pipeline.steps()
    last = [step.name for step in steps].index(names[-1])
    earlier = {step.name for step in steps[:last + 1]} - set(names)
    _case.id = case_id
    try:
        runner = _CaseRunner(project, pipeline, case_id, _state["run_id"], _state["rerun"] - earlier,
                             lambda text: messages.put(("progress", text)), _Reporter(messages), checks)
        for step in steps[:last + 1]:
            if runner.needs(step):
                if step.name in earlier:  # it did not work in its own part
                    return False
                runner.run_step(step)
        if last == len(steps) - 1:
            runner.finish()
        return not runner.failed
    except Exception:
        print(f"Could not process {case_id}:\n{traceback.format_exc()}", flush=True)
        return False
    finally:
        messages.put(("step", case_id, None))
        _case.id = None


class _Reporter:
    """Stands in for the heartbeat in a worker: the main process keeps running.json, and is told the step."""

    def __init__(self, messages):
        self.messages = messages

    def add(self, case_id, marker):
        pass

    def remove(self, case_id):
        pass

    def set_step(self, case_id, step):
        self.messages.put(("step", case_id, step))


class _QueueWriter:
    """Sends what a worker prints to the run's log, a line at a time, starting with the case it is about."""

    def __init__(self, messages):
        self.messages, self.local = messages, threading.local()

    def write(self, text):
        *lines, self.local.rest = (getattr(self.local, "rest", "") + text).split("\n")
        prefix = f"{case_id}: " if (case_id := getattr(_case, "id", None)) else ""
        for line in lines:
            self.messages.put(("log", f"{prefix}{line}\n"))
        return len(text)

    def flush(self):
        pass
