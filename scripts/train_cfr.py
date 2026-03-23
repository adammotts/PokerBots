import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm, trange  # noqa: E402

from agents.cfr_agent import CFRAgent  # noqa: E402

MODEL_DIR = str(ROOT / "models" / "cfr")
TOTAL_ITERATIONS = 100_000
CHECKPOINT_EVERY = 5

print(
    f"Training CFR for {TOTAL_ITERATIONS} iterations, saving every {CHECKPOINT_EVERY}"
)

agent = CFRAgent(model_path=MODEL_DIR, iterations=1)
agent.load(MODEL_DIR)
start = agent._cfr.iteration
if start > 0:
    print(f"Resuming from {start} existing iterations")
remaining = TOTAL_ITERATIONS - start

for _ in trange(remaining, initial=start, total=TOTAL_ITERATIONS,
                desc="CFR Training"):
    agent.update()
    if agent._cfr.iteration % CHECKPOINT_EVERY == 0:
        agent.save(MODEL_DIR)
        tqdm.write(f"Checkpoint saved at iteration {agent._cfr.iteration}")
