# retrieval-under-deadline

Analysis code and result files for the paper
**"Retrieval Under Deadline: Placement, Certified Gating, and Poisoning in Agentic 6G O-RAN"**.

Every table and figure in the paper's results section is produced by these scripts from public data.

## Data (third-party, not redistributed)

Create a `data/` folder containing:

| Path | Source |
|---|---|
| `data/TeleQnA.txt` | https://github.com/netop-team/TeleQnA (unzip `TeleQnA.zip`; the password is given in that repository) |
| `data/ORAN-Bench-13K/` | clone of https://github.com/prnshv/ORAN-Bench-13K |
| `data/glove-wiki-gigaword-100.gz` | gensim-data release `glove-wiki-gigaword-100` |

## Run

    pip install -r requirements.txt
    python prep.py                 # corpora and encoders -> *.npy, prep.pkl
    python e1_index.py telecom     # index storage / build / recall / latency (single thread)
    python e1_index.py glove       # same, 390,000 vectors (slow: tens of minutes)
    python e2_embed.py             # query-embedding time, corpus statistics, library versions
    python e3_gate.py              # certified gate, shift tests, random tiering, injection attack
    python e5_scoped.py            # scoped local index vs full index
    python make_tex.py             # results/*.json -> frag/*.tex (tables, figures, number macros)

Run all scripts from the repository root. Run the timing scripts (`e1_index.py`, `e2_embed.py`) alone on the machine.

## Results

`results/` holds the JSON behind every number in the paper. All random draws are seeded, so the
non-timing results reproduce exactly. Timings were taken on a 2-vCPU Intel Xeon 2.1 GHz virtual
machine and will differ on other hardware.

## Limits

The retrievers are lexical (LSA, GloVe-SIF, sparse TF-IDF), no language model is in the loop,
and the poisoning attack is simulated on real questions and distractors. See the paper for details.

## Citation

Citation details will be added on publication.
