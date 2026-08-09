"""
M.1 — Monitoring GPU des workloads éphémères (pods RunPod).

Doctrine [D-M.1] : éphémère → push MLflow (par-run) ; persistant → Prometheus.
Deux mécanismes, mêmes clés de métriques (comparables dans Streamlit) :

    1. GpuStatsCallback (Lightning) — fusions M3/M3.2 via LightningExperiment.
       Hooks batch → dataloader_wait_ratio en plus des métriques GPU.
    2. gpu_sampling() (context manager, thread) — base learners fit,
       eval_gold_champion, reevaluate_actives. Échantillonnage périodique.

Clés loggées :
    gpu/utilization_pct      — occupation SM (LA métrique ; <80% soutenu = goulot)
    gpu/memory_used_gb       — dimensionnement batch_size / choix GPU
    gpu/power_watts          — proxy de saturation réelle
    throughput/samples_per_sec
    profiler/dataloader_wait_ratio  — Lightning only ; >0.15 = data bottleneck

Dégradation gracieuse : sans CUDA ou sans pynvml → no-op silencieux
(les tests locaux CPU et les contextes non-GPU ne cassent jamais).
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time

logger = logging.getLogger(__name__)

SAMPLE_EVERY_N_BATCHES = 20   # [D-M.1] validé user
THREAD_SAMPLE_PERIOD_S = 5.0


def _try_nvml():
    """Retourne (pynvml, handle) ou (None, None) si indisponible."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml, handle
    except Exception as e:
        logger.info(f"[gpu_stats] NVML indisponible ({e}) → monitoring GPU désactivé")
        return None, None


def _read_gpu(pynvml, handle) -> dict:
    """Une lecture NVML (~µs). Clés = celles loggées en MLflow."""
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    out = {
        "gpu/utilization_pct": float(util.gpu),
        "gpu/memory_used_gb": mem.used / 1e9,
    }
    try:
        out["gpu/power_watts"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        pass  # certains GPU n'exposent pas la télémétrie power
    return out


# ====================================================================== #
# 1. Callback Lightning (fusions M3/M3.2)                                 #
# ====================================================================== #

try:
    import lightning as L

    class GpuStatsCallback(L.Callback):
        """
        Échantillonne GPU + débit + décomposition data/compute toutes les
        N batches, pousse dans le run MLflow actif (step = global_step).
        """

        def __init__(self, every_n_batches: int = SAMPLE_EVERY_N_BATCHES):
            self.every_n = every_n_batches
            self._pynvml = None
            self._handle = None
            # fenêtre glissante entre deux échantillons
            self._t_prev_batch_end: float | None = None
            self._t_batch_start: float | None = None
            self._win_compute = 0.0
            self._win_wait = 0.0
            self._win_samples = 0
            self._win_t0: float | None = None
            # agrégats du fit
            self._all_util: list[float] = []
            self._all_ratio: list[float] = []

        def on_fit_start(self, trainer, pl_module):
            self._pynvml, self._handle = _try_nvml()
            self._win_t0 = time.time()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            now = time.time()
            self._t_batch_start = now
            if self._t_prev_batch_end is not None:
                self._win_wait += now - self._t_prev_batch_end

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            now = time.time()
            if self._t_batch_start is not None:
                self._win_compute += now - self._t_batch_start
            self._t_prev_batch_end = now
            # taille du batch : premier tenseur trouvé
            try:
                first = batch[0] if isinstance(batch, (list, tuple)) else batch
                self._win_samples += len(first)
            except Exception:
                pass

            if (batch_idx + 1) % self.every_n != 0:
                return

            import mlflow
            step = trainer.global_step
            metrics = {}

            if self._pynvml is not None:
                try:
                    metrics.update(_read_gpu(self._pynvml, self._handle))
                    self._all_util.append(metrics["gpu/utilization_pct"])
                except Exception:
                    pass

            elapsed = now - (self._win_t0 or now)
            if elapsed > 0 and self._win_samples > 0:
                metrics["throughput/samples_per_sec"] = self._win_samples / elapsed

            denom = self._win_compute + self._win_wait
            if denom > 0:
                ratio = self._win_wait / denom
                metrics["profiler/dataloader_wait_ratio"] = ratio
                self._all_ratio.append(ratio)

            if metrics:
                try:
                    mlflow.log_metrics(metrics, step=step)
                except Exception as e:
                    logger.debug(f"[gpu_stats] log_metrics failed: {e}")

            # reset fenêtre
            self._win_compute = 0.0
            self._win_wait = 0.0
            self._win_samples = 0
            self._win_t0 = time.time()

        def on_fit_end(self, trainer, pl_module):
            import mlflow
            aggregates = {}
            if self._all_util:
                aggregates["gpu/utilization_mean_pct"] = sum(self._all_util) / len(self._all_util)
                aggregates["gpu/utilization_max_pct"] = max(self._all_util)
            if self._all_ratio:
                aggregates["profiler/dataloader_wait_ratio_mean"] = (
                    sum(self._all_ratio) / len(self._all_ratio)
                )
            if aggregates:
                try:
                    mlflow.log_metrics(aggregates)
                except Exception:
                    pass
            if self._pynvml is not None:
                with contextlib.suppress(Exception):
                    self._pynvml.nvmlShutdown()

except ImportError:
    GpuStatsCallback = None  # contexte sans Lightning (ex: image Airflow)


# ====================================================================== #
# 2. Context manager thread (BL fits, eval_gold, reevaluate)              #
# ====================================================================== #

@contextlib.contextmanager
def gpu_sampling(n_samples_hint: int | None = None,
                 period_s: float = THREAD_SAMPLE_PERIOD_S):
    """
    Échantillonne le GPU dans un thread pendant le bloc, pousse les
    agrégats dans le run MLflow ACTIF à la sortie.

    Usage :
        with gpu_sampling(n_samples_hint=len(df)):
            learner.fit(...)          # ou scorer.score(...)

    Sans CUDA/pynvml/MLflow actif : no-op silencieux.
    """
    pynvml, handle = _try_nvml()
    readings: list[dict] = []
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            try:
                readings.append(_read_gpu(pynvml, handle))
            except Exception:
                pass
            stop.wait(period_s)

    thread = None
    t0 = time.time()
    if pynvml is not None:
        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()

    try:
        yield
    finally:
        duration = time.time() - t0
        if thread is not None:
            stop.set()
            thread.join(timeout=2.0)
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

        try:
            import mlflow
            if mlflow.active_run() is None:
                return
            metrics = {}
            if readings:
                for key in readings[0]:
                    vals = [r[key] for r in readings if key in r]
                    metrics[f"{key.replace('gpu/', 'gpu/')}_mean"] = sum(vals) / len(vals)
                    metrics[f"{key}_max"] = max(vals)
            if n_samples_hint and duration > 0:
                metrics["throughput/samples_per_sec"] = n_samples_hint / duration
            metrics["gpu/sampling_duration_s"] = duration
            if metrics:
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.debug(f"[gpu_sampling] flush failed: {e}")