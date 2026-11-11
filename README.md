# Benchmarking Model-Based and Model-Free Reinforcement Learning for Ammonia Process Optimization

## Abstract

This project evaluates the performance of Bayesian optimization (BO), DreamerV3, and proximal policy optimization (PPO) for ammonia process-control optimization in a simulated industrial environment. The objective is to compare learning-based and optimization-based control strategies in terms of operating-point quality, stability, and benchmark performance under realistic process constraints.

The analysis emphasizes reproducibility and presentation of final results in a format suitable for scientific communication and technical review. The repository includes the source analysis material, supporting simulation and optimization code, and the final publication-ready benchmark outputs.

## Problem setting

The study considers a constrained ammonia process optimization problem in which the controller must balance process efficiency, operating limits, and system stability. The benchmark compares model-free reinforcement learning, model-based reinforcement learning, and a classical black-box optimization baseline to assess their relative effectiveness in a dynamic industrial control setting.

## Methodology

The project combines:

- a process simulation environment for ammonia control
- optimization- and learning-based control strategies
- benchmark evaluation over operating metrics and summary statistics
- final figure and table export in a format suitable for publication and technical reporting

The source analysis is organized in a set of notebooks and scripts that were used to generate the final benchmark tables, ranking summaries, and visual comparison outputs.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── final/
│   ├── ammonia_benchmark_overview.png
│   ├── ammonia_benchmark_overview.pdf
│   ├── benchmark_summary.csv
│   ├── benchmark_summary.tex
│   ├── dreamerv3_best_run_summary.csv
│   ├── dreamerv3_ranking.csv
│   ├── dreamerv3_ranking.tex
│   ├── operating_point_comparison.csv
│   ├── operating_point_comparison.tex
│   ├── action_profile_comparison.csv
│   ├── ppo_baseline_summary.csv
│   └── statistical_summary.csv
├── ammonia_bo_dreamerv3_ppo_publish_ready_comparison.ipynb
├── ammonia_bo_summary.ipynb
├── ammonia_dreamerv3_gap_closure.ipynb
├── ammonia_dreamerv3_summary.ipynb
├── ammonia_dreamerv3_target95_elite_replay.ipynb
├── ammonia_dreamerv3_target95_elite_replay_close_gap.ipynb
├── ammonia_dreamerv3_target96to100_adaptive.ipynb
├── ammonia_ppo_target98.ipynb
├── dreamerv3_haberbosch_helpers.py
├── env.py
├── optimize.py
├── .dwsim_support/
│   └── dwsim_rosetta_bridge.py
├── DWSIM Simulation/
│   └── Ammonia_RL.dwxmz
├── third_party/
│   └── dreamerv3_torch/
├── artifacts/
├── .gitignore
└── .venv/ (excluded from the repository)
```

## Core analysis components

- `ammonia_bo_dreamerv3_ppo_publish_ready_comparison.ipynb`: primary benchmark analysis and final export notebook
- `ammonia_bo_summary.ipynb`: BO-specific evaluation summary
- `ammonia_dreamerv3_summary.ipynb`: DreamerV3 benchmark summary and diagnostic analysis
- `ammonia_dreamerv3_gap_closure.ipynb`: gap-closure analysis and improvement tracking
- `ammonia_dreamerv3_target95_elite_replay.ipynb`: elite replay target-95 analysis
- `ammonia_dreamerv3_target96to100_adaptive.ipynb`: adaptive target optimization analysis
- `ammonia_ppo_target98.ipynb`: PPO baseline and target-98 evaluation
- `dreamerv3_haberbosch_helpers.py`: helper routines for benchmark processing and model analysis
- `env.py`: simulation environment interface and reward logic
- `optimize.py`: optimization and parameter-selection utilities
- `DWSIM Simulation/Ammonia_RL.dwxmz`: process model used for the ammonia simulation environment

## Final result bundle

The `final/` directory contains the publication-ready outputs used to summarize the study, including:

- benchmark ranking and summary tables
- operating-point comparisons
- action-profile comparison data
- benchmark overview plots in PNG and PDF formats
- statistical summary exports for comparative evaluation

## Scientific relevance

This work contributes to the evaluation of reinforcement learning methods for industrial process optimization by benchmarking representative approaches under a shared environmental specification. The project is designed to support methodological comparison, reproducible analysis, and communication of outcomes in a form suitable for technical discussion and continuation into further research.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
