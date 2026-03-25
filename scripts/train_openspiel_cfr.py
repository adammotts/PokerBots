import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm, trange  # noqa: E402

from agents.openspiel_cfr_agent import OpenSpielCFRAgent  # noqa: E402

MODEL_DIR = str(ROOT / "models" / "openspiel_cfr")
TOTAL_ITERATIONS = 1_000_000
CHECKPOINT_EVERY = 10_000

print(
    f"Training OpenSpiel MCCFR for {TOTAL_ITERATIONS} iterations, "
    f"saving every {CHECKPOINT_EVERY}"
)

agent = OpenSpielCFRAgent(iterations=1)
agent.load(MODEL_DIR)
start = agent.total_iterations
if start > 0:
    print(f"Resuming from {start} existing iterations")
remaining = TOTAL_ITERATIONS - start

for _ in trange(
    remaining, initial=start, total=TOTAL_ITERATIONS, desc="MCCFR Training"
):
    agent.update()
    if agent.total_iterations % CHECKPOINT_EVERY == 0:
        try:
            agent.save(MODEL_DIR)
            tqdm.write(f"Checkpoint saved at iteration {agent.total_iterations}")
        except Exception as e:
            tqdm.write(f"[WARNING] Checkpoint failed at iteration {agent.total_iterations}: {e} — continuing")

agent.save(MODEL_DIR)
print(f"Training complete. Model saved to {MODEL_DIR}")
