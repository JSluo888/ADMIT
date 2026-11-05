# ADMIT

Calibrated cell typing for spatial proteomics when markers are missing. ADMIT keeps the cell types that a
marker panel can support and fits one conformal threshold per type. It works on top of any classifier.

## Install

```bash
pip install -e ".[experiments,figures]"
```

## Example

```bash
python examples/toy_demo.py --delete CD8
```

## Use as an agent skill

`skill/SKILL.md` describes ADMIT for AI coding agents. Copy the folder into the agent's skills directory, for
example `~/.claude/skills/admit` (Claude Code) or `~/.codex/skills/admit` (Codex).

## Reproducing the paper

`experiments/` holds the experiment scripts and `figures/` the figure scripts. Set `ADMIT_DATA_ROOT` to the
cohort data and `ADMIT_RUNS` to an output folder. Cohort sources are listed in the paper.

## License

MIT
