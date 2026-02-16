"""FreqTrade Manager - Entry point.

Usage:
    python main.py [--config config.yaml] [--port 8080]
"""

import argparse
import logging
import logging.handlers
import os
import signal
import sys

# Ensure the script's directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

from manager.config import AppConfig, load_config
from manager.state import AppState
from manager.process_manager import ProcessManager
from manager.hyperopt_monitor import HyperoptMonitor
from manager.workflow import Workflow
from manager.scheduler import WorkflowScheduler
from manager.web_app import create_app


def setup_logging(cfg: AppConfig) -> None:
    """Configure logging with file rotation."""
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(log_level)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        cfg.logging.file,
        maxBytes=cfg.logging.max_bytes,
        backupCount=cfg.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Also pipe logs into the state log buffer
    class StateHandler(logging.Handler):
        def __init__(self, state: AppState):
            super().__init__()
            self.state = state

        def emit(self, record):
            try:
                self.state.add_log(self.format(record))
            except Exception:
                pass

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(console)
    root.addHandler(file_handler)

    return StateHandler


def main():
    parser = argparse.ArgumentParser(description="FreqTrade Manager")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--port", type=int, default=None, help="Override web port")
    parser.add_argument("--host", default=None, help="Override web host")
    args = parser.parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"Config file not found: {config_path}")
        print("Please create config.yaml from the template and restart.")
        sys.exit(1)

    config = load_config(config_path)
    config.manager_dir = os.path.dirname(config_path)

    # Override from CLI
    if args.port:
        config.web.port = args.port
    if args.host:
        config.web.host = args.host

    # Setup logging
    StateHandlerClass = setup_logging(config)

    # Initialize components
    state = AppState()

    # Add state handler for logging
    state_handler = StateHandlerClass(state)
    state_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(state_handler)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("FreqTrade Manager starting")
    logger.info(f"Config: {config_path}")
    logger.info(f"FreqTrade dir: {config.freqtrade_dir}")
    logger.info(f"Strategies: {[s.name for s in config.strategies]}")
    logger.info(f"Web: http://{config.web.host}:{config.web.port}")
    logger.info("=" * 60)

    # Validate freqtrade installation
    ft_exe = config.freqtrade_exe
    if not os.path.isfile(ft_exe):
        logger.error(f"FreqTrade executable not found: {ft_exe}")
        logger.error("Check freqtrade.directory and freqtrade.venv_path in config")
        sys.exit(1)

    proc_mgr = ProcessManager(config, state)
    hyperopt_mon = HyperoptMonitor(config, state)
    wf = Workflow(config, state, proc_mgr, hyperopt_mon, config_path)
    sched = WorkflowScheduler(config, state, wf, proc_mgr, config_path)

    # Create FastAPI app
    app = create_app(config, state, proc_mgr, hyperopt_mon, wf, sched, config_path)

    # Start scheduler
    sched.start()

    # Graceful shutdown
    _shutting_down = False
    def shutdown(signum=None, frame=None):
        nonlocal _shutting_down
        if _shutting_down:
            logger.warning("Forced exit (second Ctrl+C)")
            os._exit(1)
        _shutting_down = True
        logger.info("Shutting down...")
        sched.stop()
        proc_mgr.stop_all()  # Parallel kill, waits up to 20s
        logger.info("All processes stopped, exiting")
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, 'SIGTERM'):
        try:
            signal.signal(signal.SIGTERM, shutdown)
        except OSError:
            pass  # SIGTERM not fully supported on Windows

    # Start web server
    logger.info(f"Starting uvicorn on {config.web.host}:{config.web.port} ...")
    try:
        uvicorn.run(
            app,
            host=config.web.host,
            port=config.web.port,
            log_level="info",
            access_log=False,
        )
    except KeyboardInterrupt:
        shutdown()
    except Exception as e:
        logger.error(f"Uvicorn failed to start: {e}")
        shutdown()


if __name__ == "__main__":
    main()
