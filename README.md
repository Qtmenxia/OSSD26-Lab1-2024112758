# Home Task 1 - Improved TSP Program

This package completes the first homework from **Lecture 1: AI Intro**.

## Assignment implemented

- Keep the lecture's TSP/simulated-annealing setup.
- Generate 100 two-dimensional points with coordinates in `[0, 10)`.
- Improve the route search by replacing the original simple swap with a **2-opt path reversal**.
- Show that the improved method finds a shorter route.

## Included files

| File | Purpose |
|---|---|
| `homework1_minimal.py` | Minimal submission version that closely follows the code shown in the lecture and changes the candidate move to 2-opt. |
| `tsp_homework1.py` | Extended, reproducible comparison of simple-swap SA and 2-opt SA. |
| `TSP_Homework1.ipynb` | Ready-to-submit Jupyter notebook with explanation, execution output, and figures. |
| `test_tsp.py` | Unit tests for city generation, route length, 2-opt delta, and improvement. |
| `requirements.txt` | Python dependencies. |
| `results/` | Generated plots and JSON result summary. |

## Quick start

```bash
python -m pip install -r requirements.txt
python tsp_homework1.py
```

The result images and `summary.json` will be written to `results/`.

For the minimal lecture-style version:

```bash
python homework1_minimal.py
```

## Reproduce the submitted result

```bash
python tsp_homework1.py \
  --city-seed 42 \
  --optimizer-seed 43 \
  --iterations 100000 \
  --metric euclidean
```

On the included deterministic run:

- Initial route length: **542.756155**
- Simple-swap simulated annealing: **139.081994**
- Improved 2-opt simulated annealing: **80.533170**
- Improvement over the simple-swap baseline: **42.10%**

## Why 2-opt is better here

The simple-swap move exchanges two city positions. This can disrupt several good route segments at once and often leaves crossing or unnecessarily long edges.

A 2-opt move removes two route edges and reverses the segment between them. For a symmetric TSP, this preserves all internal segment distances and changes only two boundary edges. It is therefore well suited to removing route crossings and making focused local improvements.

The extended implementation also keeps the **best route ever found**, rather than returning only the final accepted state.

## Distance metric note

- `homework1_minimal.py` preserves the distance function shown in the lecture.
- `tsp_homework1.py` defaults to standard Euclidean distance.
- To reproduce the lecture distance behavior in the extended program, run:

```bash
python tsp_homework1.py --metric lecture
```

## Tests

```bash
python -m unittest -v test_tsp.py
```

## Submission

Upload or share either:

1. `TSP_Homework1.ipynb`, or
2. this complete folder/ZIP archive.

Before submitting, replace the placeholder student information in the first notebook cell.

## Git Version Control Practice

This section records the Git operations
performed during the OSSD26-Lab1 experiment.

Git tracks changes through commits.
Each commit records a snapshot of the project.
The commit history helps developers review
and manage different versions of the software.

### Key Git Concepts

- Branches support parallel development.
- Merging integrates changes from different branches.
- Remote repositories support collaboration and backup.
