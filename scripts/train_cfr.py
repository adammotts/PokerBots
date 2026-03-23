import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm, trange  # noqa: E402

from agents.cfr_agent import CFRAgent  # noqa: E402

MODEL_DIR = str(ROOT / "models" / "cfr")
TOTAL_ITERATIONS = 100_000
CHECKPOINT_EVERY = 1000

print(
    f"Training CFR for {TOTAL_ITERATIONS} iterations, saving every {CHECKPOINT_EVERY}"
)

agent = CFRAgent(model_path=MODEL_DIR, iterations=1)
for i in trange(TOTAL_ITERATIONS, desc="CFR Training"):
    agent.update()
    if i % CHECKPOINT_EVERY == 0 and i > 0:
        agent.save(MODEL_DIR)
        tqdm.write(f"Checkpoint saved at iteration {i}")
