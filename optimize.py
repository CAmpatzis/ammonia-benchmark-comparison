import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound

from env import HaberBoschEnv


env = HaberBoschEnv()


def evaluate_flowsheet(a1, a2, a3, a4, a5, a6, a7):
    action = np.array([a1, a2, a3, a4, a5, a6, a7], dtype=np.float64)
    env.reset()
    _, reward, _, _, _ = env.step(action)
    return float(reward)


pbounds = {f"a{i}": (-1.0, 1.0) for i in range(1, 8)}
optimizer = BayesianOptimization(
    f=evaluate_flowsheet,
    pbounds=pbounds,
    acquisition_function=UpperConfidenceBound(kappa=7.5, random_state=42),
    random_state=42,
)


def main() -> None:
    optimizer.maximize(init_points=50, n_iter=500)

    best_normalized = np.array(
        [optimizer.max["params"][f"a{i}"] for i in range(1, 8)],
        dtype=np.float64,
    )
    best_physical = env._scale_action(best_normalized)

    print(f"Maximum target: {optimizer.max['target']:.6f}")
    print("Best normalized parameters:", optimizer.max["params"])
    print("Best physical parameters:", best_physical)


if __name__ == "__main__":
    main()