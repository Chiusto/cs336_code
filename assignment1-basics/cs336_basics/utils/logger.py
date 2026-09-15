import time, json
from pathlib import Path

class Logger:
    """minimal: track step and wall-clock"""
    def __init__(self, log_dir="logs/run", config=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        self.f = open(self.log_dir / "log.jsonl", "a", encoding="utf-8")
        if config:
            (self.log_dir / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def log(self, step, train_loss=None, val_loss=None, **kw):
        elapsed = time.monotonic() - self.t0
        row = {"step": step, "wall_s": round(elapsed, 2), "train_loss": train_loss, "val_loss": val_loss, **kw}
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()
        print(f"step {step:5d} | wall {elapsed:.1f}s | train {train_loss} | val {val_loss}")
        return row

    def close(self):
        self.f.close()

    def plot(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return print("pip install matplotlib")
        rows = [json.loads(l) for l in open(self.log_dir / "log.jsonl", encoding="utf-8")]
        if not rows:
            print("no data")
            return
        steps = [r["step"] for r in rows]
        walls = [r["wall_s"]/60 for r in rows]
        for k in ["train_loss", "val_loss"]:
            ys = [r.get(k) for r in rows]
            if any(v is not None for v in ys):
                plt.figure()
                plt.plot(steps, ys, label=k)
                plt.xlabel("step"); plt.ylabel(k); plt.legend()
                plt.savefig(self.log_dir / f"{k}_vs_step.png")
                plt.figure()
                plt.plot(walls, ys, label=k)
                plt.xlabel("wall (min)"); plt.ylabel(k); plt.legend()
                plt.savefig(self.log_dir / f"{k}_vs_wall.png")
        print(f"saved to {self.log_dir}")
