# Contributing

- Follow the [README](../README.md) to set up the environment (`uv sync`).
- Create a branch and keep changes focused. Follow the existing code style and update documentation or tests when needed.
- Before opening a PR, run `uv run pytest` and `uv run ruff check --select E9,F63,F7,F82 .`. If you changed the environment, rewards or networks, also run `uv run pytest -m slow` (CPU training smoke test, ~3 min).
- Open a pull request describing the change and how you tested it.

## Experiments vs. code

- **Experiments don't need a PR.** Changing reward weights or PPO settings in a config, training and evaluating is an experiment: record it in the experiment log (config, git commit, seed, eval numbers, video).
- **Code changes do:** new reward terms, environment or observation changes, network changes, tooling. Branch, PR, green CI, review.
- **Promoting a policy to NEMO is a PR** that includes the config, eval numbers versus the current policy, and a video. Once merged, the policy is published as a GitHub Release, and prairie-ros-sim updates to that release.
- **Changing the observation or action layout** also requires a matching change in prairie-ros-sim (`prairie_control/helpers/policy_network.py`). The `policy-contract` CI job checks that both repos' network code agree.

## Branch names

| Prefix   | For                          | Example                         |
|----------|------------------------------|---------------------------------|
| `feat/`  | new capability               | `feat/config-files`             |
| `fix/`   | bug fix                      | `fix/hip-yaw-gain`              |
| `docs/`  | README and docs only         | `docs/training-guide`           |
| `test/`  | adding tests                 | `test/reward-terms`             |
| `ci/`    | workflows                    | `ci/train-smoke`                |
| `chore/` | cleanup, dependency bumps    | `chore/dedupe-textures`         |
| `exp/`   | experiment branches (optional; no PR needed) | `exp/torque-limits`   |
