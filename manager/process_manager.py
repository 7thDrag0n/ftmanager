"""Process manager for FreqTrade subprocesses on Windows."""

import os
import subprocess
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Callable

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

from .config import AppConfig, StrategyConfig
from .state import AppState, ProcessInfo, ProcessStats, ProcessType, ProcessStatus

logger = logging.getLogger(__name__)

CREATE_NEW_PROCESS_GROUP = 0x00000200


def calc_timerange(start_days_ago: int, end_days_ago: int) -> str:
    """Calculate freqtrade timerange string from days-ago values."""
    start = (datetime.now() - timedelta(days=start_days_ago)).strftime("%Y%m%d")
    if end_days_ago == 0:
        return f"{start}-"
    end = (datetime.now() - timedelta(days=end_days_ago)).strftime("%Y%m%d")
    return f"{start}-{end}"


def _collect_gpu_for_pids(pids: set[int]) -> dict | None:
    """Check if any PIDs are using a GPU. Returns stats for the first matching GPU."""
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return None

    for i in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Check compute + graphics processes on this GPU
            gpu_pids = set()
            try:
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    gpu_pids.add(p.pid)
            except Exception:
                pass
            try:
                for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle):
                    gpu_pids.add(p.pid)
            except Exception:
                pass

            if not pids & gpu_pids:
                continue

            # Our process is on this GPU — collect stats
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                fan = None

            return {
                "util": util.gpu,
                "mem_mb": mem_info.used / (1024 * 1024),
                "temp": temp,
                "fan": fan,
                "name": name,
            }
        except Exception:
            continue

    return None


class ProcessManager:
    """Manages freqtrade subprocesses with output capture and stats."""

    def __init__(self, config: AppConfig, state: AppState):
        self.config = config
        self.state = state
        self._procs: dict[str, subprocess.Popen] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._stats_threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def _proc_key(self, ptype: ProcessType, strategy: str) -> str:
        return f"{ptype.value}:{strategy}"

    def _cfg_path(self, strategy: StrategyConfig, config_attr: str) -> str:
        """Resolve a config path relative to freqtrade dir."""
        rel = getattr(strategy, config_attr, "config.json")
        return os.path.join(self.config.freqtrade_dir, rel)

    # ── Command builders ──────────────────────────────────────────────

    def build_trade_cmd(self, strategy: StrategyConfig) -> list[str]:
        exe = self.config.freqtrade_exe
        cfg = self._cfg_path(strategy, "trade_config")
        cmd = [exe, "trade", "--config", cfg, "--strategy", strategy.strategy_name]
        if strategy.trade_extra_args:
            cmd += strategy.trade_extra_args.split()
        return cmd

    def build_download_cmd(self, strategy: StrategyConfig) -> list[str]:
        exe = self.config.freqtrade_exe
        cfg = self._cfg_path(strategy, "download_config")
        cmd = [exe, "download-data", "--config", cfg]
        timerange = calc_timerange(strategy.download_data.days_back, 0)
        cmd += ["--timerange", timerange]
        cmd += ["--timeframes"] + strategy.download_data.timeframes
        if strategy.download_data.extra_args:
            cmd += strategy.download_data.extra_args.split()
        return cmd

    def build_backtest_cmd(self, strategy: StrategyConfig) -> list[str]:
        exe = self.config.freqtrade_exe
        cfg = self._cfg_path(strategy, "backtest_config")
        cmd = [exe, "backtesting", "--config", cfg, "--strategy", strategy.strategy_name]
        timerange = calc_timerange(
            strategy.backtest.timerange_start_days_ago,
            strategy.backtest.timerange_end_days_ago,
        )
        cmd += ["--timerange", timerange]
        if strategy.backtest.extra_args:
            cmd += strategy.backtest.extra_args.split()
        return cmd

    def build_hyperopt_cmd(self, strategy: StrategyConfig) -> list[str]:
        exe = self.config.freqtrade_exe
        cfg = self._cfg_path(strategy, "hyperopt_config")
        cmd = [exe, "hyperopt", "--config", cfg, "--strategy", strategy.strategy_name]
        cmd += ["--hyperopt-loss", strategy.hyperopt.loss_function]
        cmd += ["--spaces"] + strategy.hyperopt.spaces.split()
        cmd += ["-e", str(strategy.hyperopt.epochs)]
        cmd += ["-j", str(strategy.hyperopt.jobs)]
        cmd += ["--min-trades", str(strategy.hyperopt.min_trades)]
        if strategy.hyperopt.timeframe_detail:
            cmd += ["--timeframe-detail", strategy.hyperopt.timeframe_detail]
        if strategy.hyperopt.disable_param_export:
            cmd += ["--disable-param-export"]
        timerange = calc_timerange(
            strategy.hyperopt.timerange_start_days_ago,
            strategy.hyperopt.timerange_end_days_ago,
        )
        cmd += ["--timerange", timerange]
        if strategy.hyperopt.extra_args:
            cmd += strategy.hyperopt.extra_args.split()
        return cmd

    def build_hyperopt_show_cmd(self, strategy: StrategyConfig, epoch_num: int) -> list[str]:
        exe = self.config.freqtrade_exe
        cfg = self._cfg_path(strategy, "hyperopt_config")
        return [exe, "hyperopt-show", "--config", cfg, "-n", str(epoch_num)]

    def build_reload_cmd(self, strategy: StrategyConfig) -> list[str]:
        """Build freqtrade-client reload_config command.
        Config is relative to the manager folder."""
        # freqtrade-client should be in the same venv
        client_exe = os.path.join(
            self.config.freqtrade_dir, self.config.venv_path, "Scripts", "freqtrade-client.exe"
        )
        cfg = os.path.join(self.config.manager_dir, strategy.reload_client_config)
        return [client_exe, "--config", cfg, "reload_config"]

    # ── Process lifecycle ─────────────────────────────────────────────

    def start_process(
        self,
        ptype: ProcessType,
        strategy: StrategyConfig,
        cmd: list[str] | None = None,
        on_line: Callable[[str], None] | None = None,
        on_complete: Callable[[int], None] | None = None,
    ) -> bool:
        key = self._proc_key(ptype, strategy.name)

        with self._lock:
            existing = self.state.get_process(ptype, strategy.name)
            if existing and existing.status == ProcessStatus.RUNNING:
                logger.warning(f"Process {key} already running (PID {existing.pid})")
                return False

            if cmd is None:
                builders = {
                    ProcessType.TRADE: self.build_trade_cmd,
                    ProcessType.DOWNLOAD: self.build_download_cmd,
                    ProcessType.BACKTEST: self.build_backtest_cmd,
                    ProcessType.HYPEROPT: self.build_hyperopt_cmd,
                    ProcessType.RELOAD: self.build_reload_cmd,
                }
                builder = builders.get(ptype)
                if not builder:
                    logger.error(f"No builder for process type: {ptype}")
                    return False
                cmd = builder(strategy)

            cmd_str = " ".join(cmd)
            info = ProcessInfo(
                process_type=ptype,
                strategy=strategy.name,
                status=ProcessStatus.STARTING,
                started_at=time.time(),
                stopped_at=0.0,
                command=cmd_str,
            )
            self.state.set_process(ptype, strategy.name, info)
            stop_event = threading.Event()
            self._stop_events[key] = stop_event

        logger.info(f"Starting {key}: {cmd_str}")
        self.state.add_log(f"[{key}] Starting: {cmd_str}")
        self.state.broadcast("process_starting", {"key": key, "cmd": cmd_str})

        # Determine cwd: reload commands run from manager dir, others from freqtrade dir
        cwd = self.config.manager_dir if ptype == ProcessType.RELOAD else self.config.freqtrade_dir

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                text=True,
                bufsize=1,
                creationflags=CREATE_NEW_PROCESS_GROUP,
            )
        except Exception as e:
            logger.error(f"Failed to start {key}: {e}")
            info.status = ProcessStatus.FAILED
            info.error = str(e)
            info.stopped_at = time.time()
            self.state.set_process(ptype, strategy.name, info)
            self.state.add_log(f"[{key}] FAILED: {e}")
            self.state.broadcast("process_failed", {"key": key, "error": str(e)})
            return False

        with self._lock:
            self._procs[key] = proc
            info.pid = proc.pid
            info.status = ProcessStatus.RUNNING
            self.state.set_process(ptype, strategy.name, info)

        self.state.broadcast("process_started", {"key": key, "pid": proc.pid})

        # Start stats collector
        self._start_stats_collector(key, proc.pid, ptype, strategy.name, stop_event)

        # Start output reader thread
        def _reader():
            try:
                for line in iter(proc.stdout.readline, ""):
                    if stop_event.is_set():
                        break
                    line = line.rstrip("\n\r")
                    if line:
                        self.state.append_output(ptype, strategy.name, line)
                        if on_line:
                            try:
                                on_line(line)
                            except Exception as e:
                                logger.error(f"on_line callback error: {e}")
                        self.state.broadcast("process_output", {"key": key, "line": line})
            except Exception as e:
                if not stop_event.is_set():
                    logger.error(f"Reader error for {key}: {e}")
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Reader for {key}: proc.wait timed out, process may be stuck")
                    rc = -9
                except Exception:
                    rc = -9
                with self._lock:
                    info_now = self.state.get_process(ptype, strategy.name)
                    if info_now:
                        info_now.return_code = rc
                        info_now.stopped_at = time.time()
                        if info_now.status == ProcessStatus.STOPPING:
                            info_now.status = ProcessStatus.COMPLETED
                        elif rc == 0:
                            info_now.status = ProcessStatus.COMPLETED
                        else:
                            info_now.status = ProcessStatus.FAILED
                            info_now.error = f"Exit code {rc}"
                        self.state.set_process(ptype, strategy.name, info_now)
                    if key in self._procs:
                        del self._procs[key]
                    # Signal stats collector to stop BEFORE removing the event
                    evt = self._stop_events.get(key)
                    if evt:
                        evt.set()
                    if key in self._stop_events:
                        del self._stop_events[key]

                logger.info(f"Process {key} ended with code {rc}")
                self.state.add_log(f"[{key}] Ended with code {rc}")
                self.state.broadcast("process_ended", {"key": key, "return_code": rc})

                if on_complete:
                    try:
                        on_complete(rc)
                    except Exception as e:
                        logger.error(f"on_complete callback error: {e}")

        t = threading.Thread(target=_reader, name=f"reader-{key}", daemon=True)
        self._threads[key] = t
        t.start()
        return True

    def stop_process(self, ptype: ProcessType, strategy_name: str, timeout: int = 15) -> bool:
        """Stop a process and all its children reliably.

        Strategy:
        1. Signal stop_event (stops our reader thread)
        2. Try graceful kill via taskkill /T (process tree, no /F) — 5s
        3. Force kill entire process tree via psutil (recursive children)
        4. Fallback to taskkill /F /T if psutil unavailable
        5. Verify all dead
        """
        key = self._proc_key(ptype, strategy_name)

        with self._lock:
            proc = self._procs.get(key)
            stop_event = self._stop_events.get(key)
            info = self.state.get_process(ptype, strategy_name)

        if not proc or not info or info.status != ProcessStatus.RUNNING:
            logger.warning(f"Process {key} not running")
            return False

        pid = proc.pid
        logger.info(f"Stopping {key} (PID {pid})")
        info.status = ProcessStatus.STOPPING
        self.state.set_process(ptype, strategy_name, info)
        self.state.broadcast("process_stopping", {"key": key})

        # Signal our reader thread to stop
        if stop_event:
            stop_event.set()

        # Step 1: Try graceful termination (5s)
        try:
            if HAS_PSUTIL:
                try:
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    parent.terminate()  # SIGTERM equivalent
                    for child in children:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    logger.info(f"Process {key} already exited")
                    self._cleanup_after_stop(proc, key, ptype, strategy_name)
                    return True
            else:
                subprocess.run(["taskkill", "/PID", str(pid), "/T"],
                               capture_output=True, timeout=5)
        except Exception as e:
            logger.debug(f"Graceful terminate for {key}: {e}")

        # Wait for graceful exit
        try:
            proc.wait(timeout=5)
            logger.info(f"Process {key} exited gracefully")
            self._kill_orphan_children(pid)
            return True
        except subprocess.TimeoutExpired:
            pass

        # Step 2: Force kill entire process tree
        logger.warning(f"Force killing {key}")
        killed = self._force_kill_tree(pid)

        if not killed:
            # Fallback: taskkill /F /T
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"],
                               capture_output=True, timeout=10)
            except Exception as e:
                logger.error(f"taskkill /F failed for {key}: {e}")
            # Last resort
            try:
                proc.kill()
            except Exception:
                pass

        # Wait for main process to finish
        try:
            proc.wait(timeout=10)
        except Exception:
            logger.error(f"Process {key} (PID {pid}) did not exit after force kill")

        # Final sweep: kill any remaining children
        self._kill_orphan_children(pid)
        return True

    def _force_kill_tree(self, pid: int) -> bool:
        """Force kill a process and all its children using psutil. Returns True if successful."""
        if not HAS_PSUTIL:
            return False
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Kill children first (bottom-up), then parent
            for child in reversed(children):
                try:
                    child.kill()  # SIGKILL equivalent on Windows
                    logger.debug(f"  Killed child PID {child.pid} ({child.name()})")
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    logger.debug(f"  Failed to kill child PID {child.pid}: {e}")

            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass

            # Verify all dead (give OS a moment)
            gone, alive = psutil.wait_procs([parent] + children, timeout=5)
            if alive:
                for p in alive:
                    logger.warning(f"  Process PID {p.pid} still alive after kill")
                    try:
                        p.kill()
                    except Exception:
                        pass
            return True
        except psutil.NoSuchProcess:
            return True  # Already dead
        except Exception as e:
            logger.error(f"psutil tree kill failed for PID {pid}: {e}")
            return False

    def _kill_orphan_children(self, parent_pid: int):
        """After parent exits, sweep for any orphaned children that might have survived."""
        if not HAS_PSUTIL:
            return
        try:
            # On Windows, children of a killed process may get reparented
            # Use psutil to find any processes whose ppid was our parent
            for proc in psutil.process_iter(['pid', 'ppid', 'name']):
                try:
                    if proc.info['ppid'] == parent_pid:
                        logger.info(f"  Killing orphan child PID {proc.info['pid']} ({proc.info['name']})")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass

    def _cleanup_after_stop(self, proc, key, ptype, strategy_name):
        """Clean up state after a process has exited."""
        try:
            rc = proc.wait(timeout=1)
        except Exception:
            rc = -1
        with self._lock:
            if key in self._procs:
                del self._procs[key]
            if key in self._stop_events:
                del self._stop_events[key]

    def is_running(self, ptype: ProcessType, strategy_name: str) -> bool:
        info = self.state.get_process(ptype, strategy_name)
        return info is not None and info.status == ProcessStatus.RUNNING

    def run_and_wait(
        self,
        ptype: ProcessType,
        strategy: StrategyConfig,
        cmd: list[str] | None = None,
        on_line: Callable[[str], None] | None = None,
        timeout: int = 0,
        cancel_event: threading.Event | None = None,
    ) -> int:
        completion = threading.Event()
        result_code = [-2]

        def _on_complete(rc):
            result_code[0] = rc
            completion.set()

        if not self.start_process(ptype, strategy, cmd, on_line, _on_complete):
            return -2

        start = time.time()

        # Set timeout_at on the ProcessInfo so frontend can show remaining time
        if timeout > 0:
            key = f"{ptype.value}:{strategy.name}"
            info = self.state.get_process(ptype, strategy.name)
            if info:
                info.timeout_at = start + timeout
                self.state.set_process(ptype, strategy.name, info)

        while not completion.is_set():
            if cancel_event and cancel_event.is_set():
                logger.info(f"Cancellation requested for {ptype.value}:{strategy.name}")
                self.stop_process(ptype, strategy.name)
                return -1
            if timeout > 0 and (time.time() - start) > timeout:
                logger.warning(f"Timeout for {ptype.value}:{strategy.name}")
                self.stop_process(ptype, strategy.name)
                return -1
            completion.wait(timeout=1)

        return result_code[0]

    def stop_all(self):
        """Stop all running processes in parallel for fast shutdown."""
        with self._lock:
            keys = list(self._procs.keys())
        if not keys:
            return

        logger.info(f"Stopping all processes: {keys}")

        # Run stop_process in parallel threads for speed
        threads = []
        for key in keys:
            parts = key.split(":", 1)
            if len(parts) == 2:
                ptype = ProcessType(parts[0])
                t = threading.Thread(
                    target=self.stop_process,
                    args=(ptype, parts[1]),
                    name=f"stop-{key}",
                    daemon=True,
                )
                t.start()
                threads.append(t)

        # Wait for all stop threads (max 20s total)
        for t in threads:
            t.join(timeout=20)

    # ── Stats collector ───────────────────────────────────────────────

    def _start_stats_collector(self, key: str, pid: int, ptype: ProcessType,
                                strategy_name: str, stop_event: threading.Event):
        """Periodically collect CPU/memory/thread stats for a process."""
        if not HAS_PSUTIL:
            return

        interval = self.config.process_stats_interval

        def _collect():
            try:
                p = psutil.Process(pid)
                # Record create_time to detect PID reuse after process dies
                p_create_time = p.create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return

            while not stop_event.is_set():
                try:
                    # Check our process is still the same (PID reuse protection)
                    if not p.is_running() or p.create_time() != p_create_time:
                        break

                    # Collect all pids (parent + children)
                    procs = [p]
                    try:
                        procs.extend(p.children(recursive=True))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                    # Snapshot 1: prime cpu_percent for all current processes
                    for proc in procs:
                        try:
                            proc.cpu_percent(interval=None)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    # Wait measurement interval
                    stop_event.wait(timeout=interval)
                    if stop_event.is_set():
                        break

                    # Verify parent still alive after wait
                    if not p.is_running() or p.create_time() != p_create_time:
                        break

                    # Snapshot 2: read real cpu_percent values
                    # Re-fetch children in case some died during wait
                    cpu = 0.0
                    mem = 0.0
                    nthreads = 0
                    all_pids = set()

                    for proc in procs:
                        try:
                            if not proc.is_running():
                                continue
                            c = proc.cpu_percent(interval=None)
                            m = proc.memory_info().rss / (1024 * 1024)
                            t = proc.num_threads()
                            cpu += c
                            mem += m
                            nthreads += t
                            all_pids.add(proc.pid)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    stats = ProcessStats(
                        cpu_percent=round(cpu, 1),
                        memory_mb=round(mem, 1),
                        num_threads=nthreads,
                        updated_at=time.time(),
                    )

                    # GPU stats
                    if HAS_NVML and all_pids:
                        try:
                            gpu_stats = _collect_gpu_for_pids(all_pids)
                            if gpu_stats:
                                stats.gpu_util = gpu_stats["util"]
                                stats.gpu_mem_mb = gpu_stats["mem_mb"]
                                stats.gpu_temp = gpu_stats["temp"]
                                stats.gpu_fan = gpu_stats["fan"]
                                stats.gpu_name = gpu_stats["name"]
                        except Exception as e:
                            logger.debug(f"GPU stats error: {e}")

                    self.state.update_stats(ptype, strategy_name, stats)
                    self.state.broadcast("process_stats", {
                        "key": key,
                        "stats": stats.to_dict(),
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                except Exception as e:
                    logger.debug(f"Stats error for {key}: {e}")
                    stop_event.wait(timeout=interval)

        t = threading.Thread(target=_collect, name=f"stats-{key}", daemon=True)
        self._stats_threads[key] = t
        t.start()
