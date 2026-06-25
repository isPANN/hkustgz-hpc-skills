# HKUST-GZ HPC Skills

Agent skills for working on the HKUST-GZ High Performance Computing clusters. Each skill is a portable [`SKILL.md`](https://docs.claude.com/en/docs/claude-code/skills) (`name` + `description` frontmatter) and works in **both [Claude Code](https://docs.claude.com/en/docs/claude-code) and [Codex](https://github.com/openai/skills)**.

## Skills

| Skill | Description |
|---|---|
| [`hkustgz-hpc2`](./hkustgz-hpc2/) | Slurm cheat sheet for the HKUST-GZ HPC **Phase 2** cluster — job submission, partitions, GPU jobs, monitoring, cancellation, walltime sizing, and storage layout. |

More clusters/workflows (Phase 1 / LSF, K8s container jobs, …) can be added as additional skill folders over time.

## Install

Each skill is a self-contained directory containing a `SKILL.md`. The agent surfaces it automatically when you ask about submitting Slurm jobs, writing `sbatch` scripts, or working on the cluster.

### Claude Code

Copy the skill folder into a skills directory:

```bash
git clone https://github.com/isPANN/hkustgz-hpc-skills.git

# Personal — available in all projects
cp -r hkustgz-hpc-skills/hkustgz-hpc2 ~/.claude/skills/

# Or per-project
cp -r hkustgz-hpc-skills/hkustgz-hpc2 .claude/skills/
```

### Codex

Use the bundled `skill-installer` (installs into `$CODEX_HOME/skills`, default `~/.codex/skills`):

```bash
# from within Codex, the skill-installer runs:
install-skill-from-github.py --repo isPANN/hkustgz-hpc-skills --path hkustgz-hpc2
```

…or just clone and copy the folder yourself:

```bash
git clone https://github.com/isPANN/hkustgz-hpc-skills.git
cp -r hkustgz-hpc-skills/hkustgz-hpc2 ~/.codex/skills/
```

Restart Codex to pick up new skills.

## License

[MIT](./LICENSE)
