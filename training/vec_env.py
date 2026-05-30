"""
Subprocess-based vectorised environment for parallel data collection.
Each env runs in its own process — true CPU parallelism.
"""

import multiprocessing as mp
import numpy as np


def _worker(conn, env_fn):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent handles Ctrl+C
    env = env_fn()
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == 'reset':
                obs = env.reset()
                conn.send(obs)
            elif cmd == 'step':
                obs, reward, done, info = env.step(payload)
                if done:
                    obs = env.reset()
                conn.send((obs, reward, done, info))
            elif cmd == 'close':
                env.close()
                return
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocVecEnv:
    """
    Runs N environments in separate subprocesses.
    Exposes reset() / step(actions) / close() on batches of size N.
    Done envs are auto-reset so the loop never needs to call reset manually.
    """

    def __init__(self, env_fns):
        self.n = len(env_fns)
        self._conns, self._procs = [], []
        for fn in env_fns:
            parent, child = mp.Pipe()
            proc = mp.Process(target=_worker, args=(child, fn), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)

    def reset(self):
        for c in self._conns:
            c.send(('reset', None))
        return np.stack([c.recv() for c in self._conns])

    def step(self, actions):
        for c, a in zip(self._conns, actions):
            c.send(('step', int(a)))
        results = [c.recv() for c in self._conns]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.array(rews, np.float32), np.array(dones), list(infos)

    def close(self):
        for c in self._conns:
            try:
                c.send(('close', None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
