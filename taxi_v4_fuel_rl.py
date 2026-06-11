"""
실행 예시
---------
python taxi_v4_fuel_rl.py --test
python taxi_v4_fuel_rl.py --demo --fuel tight --algo Q-Learning --demo-episodes 3000
python taxi_v4_fuel_rl.py --episodes 5000
python taxi_v4_fuel_rl.py --plot-only
"""

import argparse
import pickle
import time

import gymnasium as gym
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 상수 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENV_ID = "Taxi-v4"

# 연료 조건
FUEL_NONE = "none"
FUEL_GENEROUS = "generous"
FUEL_TIGHT = "tight"

FUEL_MAX_MAP = {
    FUEL_NONE: None,
    FUEL_GENEROUS: 50,
    FUEL_TIGHT: 30,
}

FUEL_LABELS = {
    FUEL_NONE: "No Fuel Limit",
    FUEL_GENEROUS: "Generous (max=50)",
    FUEL_TIGHT: "Tight (max=30)",
}

FUEL_ORDER = [FUEL_NONE, FUEL_GENEROUS, FUEL_TIGHT]
ALGOS = ["Q-Learning", "SARSA"]

# Taxi-v4 기본 행동
ACTION_SOUTH = 0
ACTION_NORTH = 1
ACTION_EAST = 2
ACTION_WEST = 3
ACTION_PICKUP = 4
ACTION_DROPOFF = 5
ACTION_REFUEL = 6

BASE_N_ACTIONS = 6
FUEL_N_ACTIONS = 7

# Taxi-v4 고정 승객/목적지 위치: R, G, Y, B
TAXI_LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

# 추가 주유소 위치
FUEL_STATIONS = {(2, 2), (1, 3)}

# generous=50까지 상태에 반영하기 위해 50으로 설정
MAX_FUEL_DISC = 50

# 학습 하이퍼파라미터
LR = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998
EPISODES = 5000
MAX_STEPS = 200
SMOOTH_WINDOW = 100
SEEDS = [0, 1, 2, 3, 4]
RESULTS_PATH = "results.pkl"

# 시각화 설정
ALGO_COLORS = {"Q-Learning": "#2196F3", "SARSA": "#FF5722"}
ALGO_LINES = {"Q-Learning": "-", "SARSA": "--"}
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. 환경 관련 함수: class 없이 dictionary로 상태 관리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_env(fuel_mode=FUEL_NONE, seed=None, render_mode=None, env_id=ENV_ID):
    """Fuel Taxi 환경을 dictionary로 생성한다."""
    assert fuel_mode in FUEL_MAX_MAP, f"Unknown fuel_mode: {fuel_mode}"

#    base_env = gym.make("Taxi-v4", render_mode="human")
    base_env = gym.make(env_id, render_mode=render_mode)
    base_state_size = base_env.observation_space.n
    max_fuel = FUEL_MAX_MAP[fuel_mode]

    # baseline은 원본 Taxi와 동일하게 500 state, 6 action만 사용한다.
    if fuel_mode == FUEL_NONE:
        state_space_size = base_state_size
        action_space_size = BASE_N_ACTIONS
    else:
        state_space_size = base_state_size * (MAX_FUEL_DISC + 1)
        action_space_size = FUEL_N_ACTIONS

    env = {
        "fuel_mode": fuel_mode,
        "max_fuel": max_fuel,
        "base_env": base_env,
        "rng": np.random.RandomState(seed),
        "state_space_size": state_space_size,
        "action_space_size": action_space_size,
        "base_obs": None,
        "fuel": None,
        "steps": 0,
        "delivered": False,
        "render_mode": render_mode,
        "env_id": env_id,
    }
    reset_env(env)
    return env


def close_env(env):
    """Gymnasium 환경을 종료한다."""
    env["base_env"].close()


def reset_env(env):
    """환경을 초기화하고 state index를 반환한다."""
    seed = int(env["rng"].randint(0, 2**31))
    base_obs, _ = env["base_env"].reset(seed=seed)

    env["base_obs"] = int(base_obs)
    env["steps"] = 0
    env["delivered"] = False

    if env["max_fuel"] is None:
        env["fuel"] = None
    else:
        env["fuel"] = env["max_fuel"]

    return get_state_idx(env)


def decode_base_obs(env):
    """base_obs를 (taxi_row, taxi_col, pass_idx, dest_idx)로 변환한다."""
    return env["base_env"].unwrapped.decode(env["base_obs"])


def goal_distance(env):
    """현재 목표까지의 맨해튼 거리.

    승객 탑승 전에는 승객 위치까지의 거리,
    승객 탑승 후에는 목적지까지의 거리로 계산한다.
    """
    taxi_row, taxi_col, pass_idx, dest_idx = decode_base_obs(env)

    if pass_idx < 4:
        goal_row, goal_col = TAXI_LOCS[pass_idx]
    else:
        goal_row, goal_col = TAXI_LOCS[dest_idx]

    return abs(taxi_row - goal_row) + abs(taxi_col - goal_col)


def get_state_idx(env):
    """Q-table에서 사용할 state index를 계산한다."""
    if env["fuel_mode"] == FUEL_NONE:
        return env["base_obs"]

    fuel_level = min(int(env["fuel"]), MAX_FUEL_DISC)
    return env["base_obs"] * (MAX_FUEL_DISC + 1) + fuel_level


def do_refuel(env):
    """REFUEL 행동 처리.

    주유소 위치이고 연료가 가득 차 있지 않을 때만 성공한다.
    그 외에는 잘못된 주유 행동으로 패널티를 부여한다.
    """
    taxi_row, taxi_col, _, _ = decode_base_obs(env)

    valid_station = (taxi_row, taxi_col) in FUEL_STATIONS
    need_fuel = env["fuel"] is not None and env["fuel"] < env["max_fuel"]

    if valid_station and need_fuel:
        env["fuel"] = env["max_fuel"]
        return -1.0

    return -11.0


def step_env(env, action):
    """환경에서 action을 실행하고 (s_next, reward, done, info)를 반환한다."""
    assert 0 <= action < env["action_space_size"], (
        f"action={action} is invalid for action_space_size={env['action_space_size']}"
    )

    env["steps"] += 1
    done = False
    prev_dist = goal_distance(env)

    if action == ACTION_REFUEL:
        # REFUEL은 fuel 조건에서만 action_space에 포함된다.
        reward = do_refuel(env)
    else:
        base_obs, reward, terminated, truncated, _ = env["base_env"].step(action)
        env["base_obs"] = int(base_obs)
        done = terminated or truncated

        # 이동 행동일 때만 연료 차감
        if env["max_fuel"] is not None and action <= ACTION_WEST:
            env["fuel"] -= 1

        if terminated and reward == 20:
            env["delivered"] = True

    # 연료 소진 시 조기 종료
    if env["max_fuel"] is not None and env["fuel"] <= 0 and not done:
        reward -= 15
        done = True

    # 거리 기반 shaping reward
    if not done or env["delivered"]:
        reward += 0.4 * (prev_dist - goal_distance(env))

    # 최대 step 제한
    if env["steps"] >= MAX_STEPS:
        done = True

    return get_state_idx(env), reward, done, {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. 에이전트 관련 함수: Q-table과 epsilon도 dictionary로 관리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_agent(state_size, action_size):
    """Q-table 기반 에이전트를 dictionary로 생성한다."""
    return {
        "q_table": np.zeros((state_size, action_size)),
        "epsilon": EPSILON_START,
        "action_size": action_size,
    }


def choose_action(agent, state):
    """epsilon-greedy 방식으로 행동을 선택한다."""
    if np.random.random() < agent["epsilon"]:
        return int(np.random.randint(agent["action_size"]))
    return int(np.argmax(agent["q_table"][state]))


def choose_greedy_action(agent, state):
    """평가용 greedy action 선택."""
    return int(np.argmax(agent["q_table"][state]))


def update_q_learning(agent, s, a, r, s_next, done):
    """Q-Learning 업데이트."""
    q_table = agent["q_table"]
    td_target = r if done else r + GAMMA * np.max(q_table[s_next])
    q_table[s, a] += LR * (td_target - q_table[s, a])


def update_sarsa(agent, s, a, r, s_next, a_next, done):
    """SARSA 업데이트."""
    q_table = agent["q_table"]
    td_target = r if done else r + GAMMA * q_table[s_next, a_next]
    q_table[s, a] += LR * (td_target - q_table[s, a])


def decay_epsilon(agent):
    """episode 종료 후 epsilon을 1회 감소시킨다."""
    agent["epsilon"] = max(EPSILON_MIN, agent["epsilon"] * EPSILON_DECAY)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. 학습 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_episode_q_learning(env, agent):
    """Q-Learning으로 한 episode를 학습한다."""
    s = reset_env(env)
    total_reward = 0.0
    done = False

    while not done:
        a = choose_action(agent, s)
        s_next, r, done, _ = step_env(env, a)
        update_q_learning(agent, s, a, r, s_next, done)
        s = s_next
        total_reward += r

    decay_epsilon(agent)
    return total_reward, int(env["delivered"]), env["steps"]


def run_episode_sarsa(env, agent):
    """SARSA로 한 episode를 학습한다."""
    s = reset_env(env)
    a = choose_action(agent, s)
    total_reward = 0.0
    done = False

    while not done:
        s_next, r, done, _ = step_env(env, a)
        a_next = choose_action(agent, s_next) if not done else 0
        update_sarsa(agent, s, a, r, s_next, a_next, done)

        s = s_next
        a = a_next
        total_reward += r

    decay_epsilon(agent)
    return total_reward, int(env["delivered"]), env["steps"]


def train_single_agent(fuel_mode, algo, seed=0, episodes=EPISODES, env_id=ENV_ID, verbose=True):
    """하나의 fuel 조건과 알고리즘으로 에이전트를 학습한다."""
    np.random.seed(seed)
    env = make_env(fuel_mode=fuel_mode, seed=seed, env_id=env_id)
    agent = make_agent(env["state_space_size"], env["action_space_size"])

    run_episode = run_episode_q_learning if algo == "Q-Learning" else run_episode_sarsa

    rewards, successes, steps_list = [], [], []
    log_interval = max(1, episodes // 10)

    for ep in range(episodes):
        r, s, t = run_episode(env, agent)
        rewards.append(r)
        successes.append(s)
        steps_list.append(t)

        if verbose and (ep + 1) % log_interval == 0:
            recent_success = np.mean(successes[-log_interval:])
            recent_reward = np.mean(rewards[-log_interval:])
            print(
                f"episode={ep + 1:5d}/{episodes}  "
                f"epsilon={agent['epsilon']:.3f}  "
                f"recent_success={recent_success:.3f}  "
                f"recent_reward={recent_reward:.2f}"
            )

    close_env(env)

    history = {
        "rewards": np.array(rewards),
        "successes": np.array(successes),
        "steps": np.array(steps_list),
    }
    return agent, history


def run_experiment(fuel_mode, algo, seed, episodes=EPISODES, env_id=ENV_ID):
    """실험 저장용 단일 run."""
    agent, history = train_single_agent(
        fuel_mode=fuel_mode,
        algo=algo,
        seed=seed,
        episodes=episodes,
        env_id=env_id,
        verbose=False,
    )
    return history


def run_all_experiments(episodes=EPISODES, save_path=RESULTS_PATH, env_id=ENV_ID):
    """전체 실험을 수행하고 결과를 pickle로 저장한다."""
    total = len(FUEL_ORDER) * len(ALGOS) * len(SEEDS)
    count = 0
    results = {}

    for fuel in FUEL_ORDER:
        for algo in ALGOS:
            seed_rewards, seed_successes, seed_steps = [], [], []

            for seed in SEEDS:
                count += 1
                print(f"[{count:2d}/{total}] fuel={fuel:8s} algo={algo:10s} seed={seed}")

                res = run_experiment(
                    fuel_mode=fuel,
                    algo=algo,
                    seed=seed,
                    episodes=episodes,
                    env_id=env_id,
                )
                seed_rewards.append(res["rewards"])
                seed_successes.append(res["successes"])
                seed_steps.append(res["steps"])

            results[(fuel, algo)] = {
                "rewards": np.array(seed_rewards),
                "successes": np.array(seed_successes),
                "steps": np.array(seed_steps),
            }

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n결과 저장: {save_path}")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. 평가 및 화면 렌더링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_with_render(agent, fuel_mode, seed=999, delay=0.5, env_id=ENV_ID):
    """학습된 Q-table을 사용하여 화면을 보면서 greedy policy를 실행한다."""
    print("\n🚕 학습된 택시의 운행을 시작합니다...")
    print(f"fuel={fuel_mode}, delay={delay}\n")

    env = make_env(
        fuel_mode=fuel_mode,
        seed=seed,
        render_mode="human",
        env_id=env_id,
    )

    s = reset_env(env)
    done = False
    total_reward = 0.0

    while not done:
        a = choose_greedy_action(agent, s)
        s, r, done, _ = step_env(env, a)
        total_reward += r

        fuel_text = "N/A" if env["fuel"] is None else str(env["fuel"])
        print(
            f"step={env['steps']:3d}  "
            f"action={a}  "
            f"reward={r:6.2f}  "
            f"fuel={fuel_text:>3s}  "
            f"total_reward={total_reward:7.2f}  "
            f"delivered={env['delivered']}"
        )
        time.sleep(delay)

    print(f"\n🏁 운행 종료! total_reward={total_reward:.2f}, steps={env['steps']}, delivered={env['delivered']}")
    close_env(env)


def run_demo(fuel_mode, algo, episodes, seed, delay, env_id=ENV_ID):
    """학습 후 화면 렌더링까지 한 번에 실행한다."""
    print("=" * 60)
    print(" Fuel-Constrained Taxi RL Demo")
    print(f" fuel={fuel_mode}, algo={algo}, episodes={episodes}")
    print("=" * 60)

    agent, _ = train_single_agent(
        fuel_mode=fuel_mode,
        algo=algo,
        seed=seed,
        episodes=episodes,
        env_id=env_id,
        verbose=True,
    )

    # 평가 단계에서는 탐험하지 않음
    agent["epsilon"] = 0.0
    evaluate_with_render(agent, fuel_mode=fuel_mode, seed=seed + 1000, delay=delay, env_id=env_id)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. 결과 요약 및 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def smooth(arr, window=SMOOTH_WINDOW):
    """이동평균 smoothing."""
    window = min(window, len(arr))
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def ci95(data):
    """seed 평균에 대한 95% confidence interval 계산."""
    n = data.shape[0]
    return data.mean(axis=0), 1.96 * data.std(axis=0) / np.sqrt(n)


def success_rate_per_seed(results, fuel, algo, last_n=500):
    return results[(fuel, algo)]["successes"][:, -last_n:].mean(axis=1)


def steps_per_seed(results, fuel, algo, last_n=500):
    successes = results[(fuel, algo)]["successes"][:, -last_n:]
    steps = results[(fuel, algo)]["steps"][:, -last_n:]

    values = []
    for i in range(successes.shape[0]):
        mask = successes[i] == 1
        values.append(steps[i][mask].mean() if mask.sum() > 0 else MAX_STEPS)
    return np.array(values)


def plot_learning_curves(results, metric="rewards", save_path="fig_reward_curves.png"):
    ylabel = "Episode Reward (smoothed)" if metric == "rewards" else "Success Rate (smoothed)"
    title = "Learning Curves — Reward" if metric == "rewards" else "Learning Curves — Success Rate"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, fuel in zip(axes, FUEL_ORDER):
        for algo in ALGOS:
            raw = results[(fuel, algo)][metric]
            smoothed = np.array([smooth(raw[i]) for i in range(raw.shape[0])])
            mean, ci = ci95(smoothed)
            x = np.arange(len(mean))

            ax.plot(
                x,
                mean,
                label=algo,
                linestyle=ALGO_LINES[algo],
                color=ALGO_COLORS[algo],
                lw=1.8,
            )
            ax.fill_between(
                x,
                mean - ci,
                mean + ci,
                alpha=0.18,
                color=ALGO_COLORS[algo],
            )

        ax.set_title(FUEL_LABELS[fuel], fontsize=10)
        ax.set_xlabel(f"Episode (window={SMOOTH_WINDOW})")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"저장: {save_path}")
    plt.close()


def bar_chart(results, get_fn, ylabel, title, save_path):
    x = np.arange(len(FUEL_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, algo in enumerate(ALGOS):
        values = [get_fn(results, fuel, algo) for fuel in FUEL_ORDER]
        means = [v.mean() for v in values]
        cis = [1.96 * v.std() / np.sqrt(len(v)) for v in values]
        offset = (i - 0.5) * width

        ax.bar(
            x + offset,
            means,
            width,
            label=algo,
            color=ALGO_COLORS[algo],
            alpha=0.85,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=cis,
            fmt="none",
            color="black",
            capsize=5,
            lw=1.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([FUEL_LABELS[fuel] for fuel in FUEL_ORDER], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"저장: {save_path}")
    plt.close()


def generate_all_figures(results_path=RESULTS_PATH):
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    plot_learning_curves(results, "rewards", "fig_reward_curves.png")
    plot_learning_curves(results, "successes", "fig_success_curves.png")
    bar_chart(
        results,
        success_rate_per_seed,
        "Success Rate (last 500 episodes)",
        "Success Rate by Fuel Condition & Algorithm",
        "fig_success_rate.png",
    )
    bar_chart(
        results,
        steps_per_seed,
        "Avg Steps per Successful Episode",
        "Avg Steps (Successful Episodes) by Fuel Condition",
        "fig_avg_steps.png",
    )
    print("모든 그래프 생성 완료!")


def print_summary(results, last_n=500):
    print("\n" + "=" * 68)
    print(f"{'Fuel':10s} {'Algorithm':12s} {'Success Rate':>14s} {'Avg Steps(succ)':>16s}")
    print("-" * 68)

    for fuel in FUEL_ORDER:
        for algo in ALGOS:
            success_rate = success_rate_per_seed(results, fuel, algo, last_n).mean()
            avg_steps = steps_per_seed(results, fuel, algo, last_n).mean()
            print(f"{fuel:10s} {algo:12s} {success_rate:>14.3f} {avg_steps:>16.1f}")

    print("=" * 68)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. 테스트 및 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_env(env_id=ENV_ID):
    print("── 환경 기본 정보 ──────────────────────────────")
    for fuel_mode in FUEL_ORDER:
        env = make_env(fuel_mode=fuel_mode, seed=0, env_id=env_id)
        print(
            f"  {fuel_mode:8s}  "
            f"max_fuel={str(env['max_fuel']):4s}  "
            f"state_space={env['state_space_size']:5d}  "
            f"actions={env['action_space_size']}"
        )
        close_env(env)

    print("\n── TIGHT 조건 랜덤 에피소드 ────────────────────")
    env = make_env(fuel_mode=FUEL_TIGHT, seed=42, env_id=env_id)
    done = False
    total_reward = 0.0
    s = reset_env(env)

    while not done:
        action = int(np.random.randint(env["action_space_size"]))
        s, reward, done, _ = step_env(env, action)
        total_reward += reward

    print(
        f"  reward={total_reward:.1f}  "
        f"steps={env['steps']}  "
        f"delivered={env['delivered']}"
    )
    close_env(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="환경 동작만 간단히 확인")
    parser.add_argument("--plot-only", action="store_true", help="기존 results.pkl로 그래프만 재생성")
    parser.add_argument("--demo", action="store_true", help="학습 후 Taxi 화면 렌더링 실행")
    parser.add_argument("--fuel", choices=FUEL_ORDER, default=FUEL_TIGHT, help="demo에 사용할 연료 조건")
    parser.add_argument("--algo", choices=ALGOS, default="Q-Learning", help="demo에 사용할 알고리즘")
    parser.add_argument("--episodes", type=int, default=EPISODES, help="전체 실험 episode 수")
    parser.add_argument("--demo-episodes", type=int, default=3000, help="demo 학습 episode 수")
    parser.add_argument("--demo-delay", type=float, default=0.5, help="렌더링 step 사이 대기 시간")
    parser.add_argument("--seed", type=int, default=0, help="demo seed")
    parser.add_argument("--results-path", default=RESULTS_PATH, help="결과 pickle 저장/로드 경로")
    parser.add_argument("--env-id", default=ENV_ID, help="Gymnasium 환경 ID. 필요 시 Taxi-v3로 변경 가능")
    args = parser.parse_args()

    if args.test:
        test_env(env_id=args.env_id)
        return

    if args.plot_only:
        generate_all_figures(results_path=args.results_path)
        return

    if args.demo:
        run_demo(
            fuel_mode=args.fuel,
            algo=args.algo,
            episodes=args.demo_episodes,
            seed=args.seed,
            delay=args.demo_delay,
            env_id=args.env_id,
        )
        return

    print("=" * 60)
    print("  Fuel-Constrained Taxi RL — Procedural Version")
    print("  Q-Learning vs SARSA")
    print("  Conditions: none / generous / tight")
    print("=" * 60 + "\n")

    results = run_all_experiments(
        episodes=args.episodes,
        save_path=args.results_path,
        env_id=args.env_id,
    )
    print_summary(results)
    generate_all_figures(results_path=args.results_path)

    print("\n생성 파일:")
    for filename in [
        args.results_path,
        "fig_reward_curves.png",
        "fig_success_curves.png",
        "fig_success_rate.png",
        "fig_avg_steps.png",
    ]:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
