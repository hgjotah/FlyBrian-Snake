#  FlyBrian Snake

**A fruit fly brain, simulated neuron by neuron, playing Snake.**

FlyBrain Snake is a Windows desktop lab where **Brian**, an adult female *Drosophila melanogaster* reconstructed from the **FlyWire FAFB v783** connectome (139,255 neurons, 54.5 million synaptic contacts), steers a snake. Every turn comes from the simulated electrical activity of Brian's descending neurons, not from a lookup table or an artificial neural network.

Watch the brain fire in 3D, explore the connectome contact by contact, train Brian with dopamine-gated synaptic plasticity, and challenge him to a live duel in the **Arena**.

---

## Highlights

-  **Whole-brain simulation**: the complete reviewed FlyWire v783 graph, run as leaky integrate-and-fire neurons on the CPU.
-  **Arena**: play against Brian side by side, on boards from 4×4 to 100×100, and watch his neurons light up in real time.
-  **Plasticity lab**: dopamine-gated depression on 4,157 KC→MBON synapses in the mushroom body, the fly's associative memory center.
-  **Built-in controls**: frozen weights, omitted dopamine, permuted graphs, shuffled rewards, MBON ablation and more.
-  **3D brain and anatomy atlas**: real FlyWire coordinates, with filters by neuropil, neurotransmitter, neuron and proofreading status.
-  **Reproducible sessions**: SQLite checkpoints, deterministic seeds, and a full command-line interface.

---

## How it works

```
   Snake game ──► Sensory encoder ──► Simulated brain ──► Motor readout ──► Turn
  (apple and       (apple and wall      (139,255 LIF         (DNa01 / DNa02:    (left,
   wall            cues become          neurons, 0.1 ms      which hemisphere   straight,
   positions)      Poisson spikes in    time step)           fires more?)       right)
                   visual neurons)              ▲
   Reward ──► Dopamine pulse (PAM01 / PPL101) ──┘   reshapes KC→MBON synapses
```

| Component | What it does |
| --- | --- |
| **Connectome** | The full graph of reviewed FlyWire v783 neurons, with neurotransmitter signs (ACh excites; GABA and glutamate inhibit). |
| **Simulator** | Leaky integrate-and-fire neurons in Numba-compiled CPU code. 0.1 ms time step, 2.2 ms refractory period, 1.8 ms synaptic delay. Cross-checked against Brian2. |
| **Sensory interface** | Apple direction and wall proximity are encoded as Poisson spike trains delivered to identified visual populations (LC10a, LC10d, LC4, LC6) and to tagged Kenyon cell subsets. |
| **Motor readout** | Each turn follows the difference in spikes between left and right descending neurons (DNa01/DNa02). A tie continues straight. |
| **Plasticity** | Dopaminergic neurons (PAM01 and PPL101) gate time-dependent depression of KC→MBON synapses, following the mushroom body learning rules described by Handler et al. 2019 and Li et al. 2020. |

---

## Installation (Windows 11)

Requirements: Windows 11 x64 and an internet connection for the first install. The installer fetches Python 3.12.11 and all dependencies by itself.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Test
.\Brian.cmd
```

- Downloads about **885 MB** of official data (FlyWire v783 from Zenodo, plus the Schlegel et al. annotations). Every file is verified by size and hash, and interrupted downloads resume automatically.
- Reserve about **3 GB** of disk. The anatomy atlas, required by the Arena, adds a **5.43 GB** local index.
- Installer options: `-SkipData` skips the data download, and `-Test` runs the test suite when finished.
- The first window takes a few seconds while Numba compiles the kernel.

Build the anatomy atlas (needed for the Arena):

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m flybrain_snake anatomy-prepare --source .
```

---

## Quick start

**Play against Brian.** Open `Brian.cmd`, go to **Arena** and press **Play**. Your snake moves with the arrow keys on the right board. The first player to reach the apple **Goal** wins, as does the one who survives when the other hits a wall.

**Train Brian.** In **Training**, choose a stage and press **New block**. Use **Pause** and **Single step** to inspect every decision.

**Command line**

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m flybrain_snake doctor                                    # check the environment
& $py -m flybrain_snake validate --trials 12                      # neural validation
& $py -m flybrain_snake train data/sessions/my_brian.sqlite --episodes 20
& $py -m flybrain_snake train data/sessions/my_brian.sqlite --resume --episodes 20
& $py -m flybrain_snake evaluate --session data/sessions/my_brian.sqlite --episodes 10
& $py -m flybrain_snake study data/runs/my_study --train 8 --test 10      # controlled experiments A-I
& $py -m flybrain_snake compare data/sessions/Brian_v783_1000.sqlite data/runs/my_comparison
& $py -m flybrain_snake export data/sessions/my_brian.sqlite data/runs/my_export
& $py -m pytest -q
```

Configuration files go before the subcommand, for example `--config configs/paredes.json`. Available presets: `default.json`, `paredes.json` (walls), `perimetro.json` (strict perimeter), `contactos.json` (per-contact transmission) and `milestones.json`. Use `--data PATH` to choose another data directory.

---

## Training stages

| Stage | Protocol |
| --- | --- |
| 0 | Propagation, stability, descending output and checkpoint retention |
| 1 | Conditioning of the KC-A pattern with PAM01 or PPL101, with KC-B as control |
| 2 | Food search on a torus, with a single-cell snake |
| 3 | Food search with terminal walls and graded penalties |
| 4 | Full Snake with growth, successive apples and terminal walls |

Dedicated practice modes (straight runs and turns, wall avoidance, strict perimeter, full games) let you train specific skills, and candidate weights are checked against held-out scenarios before they replace the active memory.

---

## Experiments

The `study` command runs a matched set of conditions on identical seeds, with paired bootstrap confidence intervals:

| Condition | Intervention |
| --- | --- |
| A | Initial weights |
| B | Learning enabled |
| C | Weights frozen during training |
| D | Dopamine reinforcement omitted |
| E | Destinations permuted within each neuropil |
| F | Reward sequence shuffled from donor episodes |
| G | Session restored from disk |
| H | Train on even cells, evaluate on odd cells |
| I | MBON functional ablation |

---

## Project structure

```
Brian.cmd              Desktop app launcher
configs/               Configuration presets
docs/                  Design, method and measured results (in Spanish)
scripts/               Windows installer and pinned dependencies
src/flybrain_snake/    Simulator, connectome, circuits, game, GUI and atlas
tests/                 pytest suite: game rules, neurons, persistence, desktop UI
```

---

## Documentation

- [Measured results](docs/resultados.md): benchmarks, controls and confidence intervals
- [Design, assumptions and provenance](docs/diseno.md): implementation decisions and sources
- [Method and reproducibility](docs/metodo.md): equations, protocols and interpretation limits

---

## Sources and credits

- [FlyWire v783](https://zenodo.org/records/10676866) (CC BY 4.0) and [FlyWire Codex](https://codex.flywire.ai/)
- [Schlegel et al. annotations](https://github.com/flyconnectome/flywire_annotations) · [paper](https://doi.org/10.1038/s41586-024-07686-5)
- [Shiu et al. 2024](https://doi.org/10.1038/s41586-024-07763-9) and their [original model](https://github.com/philshiu/Drosophila_brain_model): LIF equations and constants
- [Handler et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012144/), [Li et al. 2020](https://doi.org/10.7554/eLife.62576) and [Aso et al. 2014](https://elifesciences.org/articles/4580): mushroom body plasticity
- [Yang et al., Cell 2024](https://doi.org/10.1016/j.cell.2024.08.033): lateralization of descending neurons

The code in this project is original. Parameters, equations, classifications and data with recorded provenance are taken from the sources above. See the `LICENSE` file for terms of use.
