RFE-Core

Minimal prototype for the RFE-Core architecture: a small, self-contained cognitive vector system with a Transformer-based Generator, Watcher, Witness, Substrate, Interference, and a continuous Recursion loop.

---

Repository layout

```text
/RFE-Core
├── agents
│   ├── generator.py
│   ├── watcher.py
│   ├── witness.py
│   └── rhythm_config.json
├── substrate
│   ├── vector_space.py
│   └── topological_log.py
├── interference
│   ├── differential.py
│   └── wave_collapse.py
├── loop
│   └── recursion1188.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

Quick summary

RFE-Core is a minimal, extensible prototype that converts token sequences into dense vectors using a small Transformer encoder, monitors vector coherence, anchors relational state, stores context topologically, injects creative perturbations, and runs a continuous feedback loop.

---

Files included

- agents/generator.py Small Transformer encoder generator producing normalized vectors.
- agents/watcher.py Dissonance detector and simple stream filter.
- agents/witness.py Running anchor and relational scoring.
- agents/rhythm_config.json Rhythm modulation parameters.
- substrate/vector_space.py In-memory vector store with nearest neighbor by cosine similarity.
- substrate/topological_log.py Graph-like log with parent references and depth tracking.
- interference/differential.py Vector perturbation utilities.
- interference/wave_collapse.py Weighted superposition to compute emergent state.
- loop/recursion1188.py Orchestration loop tying components together.

---

Dependencies

Primary runtime dependencies

- Python 3.9 or newer
- PyTorch for the Transformer encoder
- NumPy for numeric utilities

Recommended installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

requirements.txt

```text
numpy>=1.23
torch>=2.0
```

If you need a specific PyTorch build for CUDA, follow the official PyTorch install instructions and replace the pip install torch step accordingly.

---

Walkthrough

1 Setup

1. Clone the repository into your workspace.
2. Create and activate a virtual environment.
3. Install dependencies from requirements.txt or install PyTorch manually for GPU support.

```bash
git clone <https://github.com/SamuelJacksonGrim/RFE-Core> RFE-Core
cd RFE-Core
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need a CUDA-enabled PyTorch, install it with the command from the official PyTorch site.

2 Run the demo

Run the recursion loop demo to verify everything is wired correctly.

```bash
python -m loop.recursion1188
```

Expected output is a small list of dictionaries printed to stdout showing key, vectornorm, relationalscore, and emergent_norm.

3 Inspect components

- Generator  
  - Located at agents/generator.py.  
  - Produces deterministic vectors from token sequences using a small Transformer encoder.  
  - To test directly:

```bash
python -c "from agents.generator import Generator; import numpy as np; g=Generator(dim=64, seed=42); v=g.generate(['hello','world']); print(v.shape, np.linalg.norm(v))"
```

- Watcher  
  - Use Watcher.dissonancescore and Watcher.filterstream to detect and filter low-variance vectors.

- Witness  
  - Use Witness.updateanchor and Witness.relationalscore to maintain and query the anchor.

- VectorSpace  
  - Use put, get, and nearest to store and query vectors.

- TopologicalLog  
  - Use add, get, and recent to track context with parent relationships and depth.

- Interference  
  - Use injectambiguity to perturb vectors and collapsewave to compute emergent states.

4 Extending the Generator

- Load pretrained weights  
  Add a loadstatedict call in Generator.init and provide a path parameter to load saved model weights.

- Tokenizer integration  
  Replace the hash-based tokento_id with a real tokenizer if you want subword consistency.

- GPU support  
  Set self.device = torch.device("cuda") in Generator when CUDA is available and ensure the environment has the correct PyTorch build.

5 Persisting state

- VectorSpace persistence  
  Replace the in-memory store with a lightweight DB such as SQLite, LMDB, or a vector DB for production.

- Model checkpoints  
  Save model weights with torch.save(model.statedict(), "models/generator.pth") and load with model.loadstate_dict(torch.load(...)).

---

Testing and CI suggestions

- Add unit tests for:
  - Generator output shape and normalization
  - Watcher dissonance thresholds
  - Witness anchor updates and relational scoring
  - VectorSpace nearest neighbor behavior
  - TopologicalLog parent/depth logic

- Example test runner

```bash
pip install pytest
pytest tests/
```

---

Notes and best practices

- Determinism  
  The generator seeds NumPy and PyTorch for reproducible behavior. For full determinism on GPU, follow PyTorch deterministic settings.

- Resource usage  
  The Transformer is intentionally small. Increase numlayers, nhead, and dimfeedforward only after profiling CPU and memory.

- Security  
  Do not commit model checkpoints or large binary files to the repository. Add them to .gitignore.

---

Next steps

- Add a lightweight REST API using FastAPI to expose RecursionLoop.step and run_stream.
- Add a persistence layer for VectorSpace and TopologicalLog.
- Add a CLI for batch processing and model checkpoint management.
- Add unit tests and a GitHub Actions workflow for CI.

---

Contributions are welcome. Open issues for feature requests, bug reports, or performance tuning. Include reproducible examples when possible.
