"""One persistent spawned decoder per GPU worker, with hard timeout recovery."""
import atexit
import multiprocessing

from .decode import worker


class DecoderProcess:
    def __init__(self, target=worker):
        self.target = target
        self.process = self.connection = None
        atexit.register(self.close)

    def _start(self):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=self.target, args=(child,), daemon=True)
        self.process.start()
        child.close()

    def request(self, job, timeout):
        if self.process is None or not self.process.is_alive():
            self.close()
            self._start()
        try:
            self.connection.send(job)
            if not self.connection.poll(timeout):
                raise TimeoutError(f'Video decode exceeded {timeout} seconds')
            result = self.connection.recv()
            if not result['ok']:
                raise RuntimeError('Video decode failed: ' + result['error'])
            return result
        except Exception:
            # A damaged codec process must not poison subsequent video jobs.
            self.close()
            raise

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
            self.process = None
