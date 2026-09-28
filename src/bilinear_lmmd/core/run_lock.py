from __future__ import annotations
import json,os,threading,time,uuid
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def exclusive_training_lock(output_root: str|Path,*,lock_name:str,stale_seconds:int=300):
    output_root=Path(output_root); output_root.mkdir(parents=True,exist_ok=True); lock=output_root/lock_name; token=uuid.uuid4().hex
    while True:
        try:
            descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(descriptor,json.dumps({"token":token,"pid":os.getpid()}).encode()); os.close(descriptor); break
        except FileExistsError:
            age=time.time()-lock.stat().st_mtime
            if age<=stale_seconds: raise RuntimeError(f"Run sedang ditulis runtime lain ({age:.0f}s sejak heartbeat): {lock}")
            lock.unlink(missing_ok=True)
    stopped=threading.Event()
    def heartbeat():
        while not stopped.wait(30):
            try: lock.touch()
            except OSError: return
    thread=threading.Thread(target=heartbeat,daemon=True); thread.start()
    try: yield lock
    finally:
        stopped.set(); thread.join(timeout=2)
        try: payload=json.loads(lock.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): payload={}
        if payload.get("token")==token: lock.unlink(missing_ok=True)
