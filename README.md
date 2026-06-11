# Fuel-Constrained Taxi RL

## 1. 프로젝트 개요

본 프로젝트는 Gymnasium의 `Taxi-v4` 환경을 기반으로, 기존 Taxi 문제에 **연료 제약 조건**을 추가한 강화학습 실험이다. 기본 Taxi 환경에서는 택시가 승객을 태우고 목적지에 내려주는 것이 목표이다. 본 프로젝트에서는 여기에 연료 개념을 추가하여, 에이전트가 제한된 연료 안에서 승객을 운송하거나 필요할 경우 주유소에서 연료를 보충하도록 환경을 확장하였다.

본 프로젝트의 목적은 다음과 같다.

- 연료 제약 조건이 Taxi 강화학습 에이전트의 학습 성능에 미치는 영향을 분석한다.
- 연료 조건에 따라 성공률, 평균 보상, 평균 step 수가 어떻게 달라지는지 비교한다.
- Q-Learning과 SARSA 알고리즘의 성능 차이를 비교한다.
- 여러 random seed를 사용하여 실험 결과의 신뢰도를 높인다.

---

## 2. 연구 가설과 최종 결론

초기 가설은 다음과 같다.

> 연료 제약 조건을 추가하면 에이전트가 더 효율적인 경로를 학습할 것이다. 단, 연료가 너무 빡빡하면 성공률이 낮아지는 trade-off가 존재할 것이다.

실험 결과, 이 가설은 **충분히 지지되지 않았다**. 연료 제약이 없는 baseline에서는 두 알고리즘 모두 안정적으로 수렴했지만, 연료 제약 조건에서는 성공률이 급격히 낮아졌고 대부분의 설정에서 수렴하지 못했다.

최종 결론은 다음과 같다.

> 연료 제약은 기대와 달리 경로 효율성을 높이지 못했고, 오히려 상태공간과 행동공간의 확대로 인해 Q-table 기반 알고리즘의 학습 난이도를 크게 증가시켰다. 다만 연료 제약이 강해질수록 성공률이 낮아지는 경향은 확인되었으므로, 성공률과 제약 강도 사이의 trade-off 가능성은 일부 확인되었다.

---

## 3. 프로젝트 동기

기본 Taxi 환경은 강화학습의 state, action, reward 개념을 이해하기 좋은 대표적인 discrete control 예제이다. 그러나 실제 경로 탐색 문제에서는 연료, 배터리, 시간, 비용과 같은 자원 제약이 함께 고려되는 경우가 많다.

본 프로젝트는 단순한 Taxi 문제 해결을 넘어, 기존 환경에 다음 요소를 추가하여 **자원 제약 기반 경로 최적화 문제**로 확장하였다.

- 연료량 추적
- 주유소 위치 설정
- 주유 행동 추가
- 연료 소진 시 패널티 부여
- 목표와의 거리 변화를 반영한 shaping reward 설계
- 연료 조건별 성능 비교

---

## 4. 환경 설계

### 4.1 기본 환경

| 항목 | 내용 |
|---|---|
| 환경 | Gymnasium Taxi-v4 |
| 기본 상태공간 | 500개 discrete state |
| 기본 행동공간 | 6개 discrete action |
| 목표 | 승객을 태운 뒤 올바른 목적지에 내려주기 |
| 최대 step 수 | episode당 200 step |

### 4.2 연료 조건

| 연료 조건 | 설명 |
|---|---|
| `none` | 연료 제약 없음. 원본 Taxi 환경과 동일한 baseline |
| `generous` | 최대 연료 50. 비교적 여유 있는 연료 조건 |
| `tight` | 최대 연료 30. 효율적 경로 선택이 중요한 조건 |

`none` 조건은 공정한 baseline 비교를 위해 원본 Taxi 환경과 동일하게 상태공간 500개, 행동공간 6개만 사용한다. `generous`와 `tight` 조건에서는 연료 상태와 `REFUEL` 행동을 추가하여 확장된 환경을 구성하였다.

---

## 5. State, Action, Reward 설계

### 5.1 State 설계

Baseline 조건에서는 원본 Taxi 상태를 그대로 사용한다.

```text
state = base_taxi_state
state_space = 500
```

연료 제약 조건에서는 기존 Taxi 상태에 연료 수준을 결합한다.

```text
state = base_taxi_state × fuel_level
state_space = 500 × 51 = 25,500
```

연료 수준은 0부터 50까지 표현하였다. 이를 통해 `generous(max=50)`와 `tight(max=30)` 조건의 차이가 상태공간에 반영되도록 설계하였다.

### 5.2 Action 설계

기본 Taxi 환경의 행동은 다음 6개이다.

| Action Index | 행동 |
|---:|---|
| 0 | South |
| 1 | North |
| 2 | East |
| 3 | West |
| 4 | Pickup |
| 5 | Dropoff |

연료 제약 조건에서는 다음 행동을 추가하였다.

| Action Index | 행동 |
|---:|---|
| 6 | Refuel |

`REFUEL` 행동은 택시가 주유소 위치에 있고, 현재 연료가 최대 연료보다 적을 때만 유효하다. 주유소가 아닌 위치에서 주유를 시도하면 잘못된 주유 행동으로 간주하여 패널티를 부여한다.

### 5.3 Reward 설계

보상 구조는 원본 Taxi 보상을 기반으로 하되, 연료 제약과 경로 효율성을 반영하도록 확장하였다.

| 보상 | 의미 |
|---:|---|
| -1 | 매 step 기본 패널티 |
| +20 | 승객을 목적지에 성공적으로 내려줌 |
| -10 | 잘못된 pickup 또는 dropoff |
| -10 | 잘못된 refuel 시도 |
| -15 | 연료 소진으로 인한 episode 종료 |
| ±0.4 × Δdistance | 목표와의 거리 변화 기반 shaping reward |

거리 기반 shaping reward는 택시가 현재 목표에 가까워지면 양의 보상을, 멀어지면 음의 보상을 준다.

- 승객 탑승 전 목표: 승객 위치
- 승객 탑승 후 목표: 목적지 위치

---

## 6. 알고리즘

본 프로젝트에서는 두 개의 tabular 강화학습 알고리즘을 비교하였다.

### 6.1 Q-Learning

Q-Learning은 다음 상태에서 가능한 행동 중 가장 큰 Q값을 사용하여 업데이트한다. 즉, 실제로 선택한 다음 행동과 관계없이 최적 행동을 가정하고 Q-table을 갱신한다.

### 6.2 SARSA

SARSA는 다음 상태에서 실제로 선택한 행동 `a'`의 Q값을 사용하여 업데이트한다. 따라서 탐험 정책의 영향을 직접 반영하며, Q-Learning보다 상대적으로 보수적인 정책을 학습할 수 있다.

---

## 7. 하이퍼파라미터

| 하이퍼파라미터 | 값 |
|---|---:|
| Learning rate `α` | 0.1 |
| Discount factor `γ` | 0.99 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.01 |
| Epsilon decay | 0.998 |
| Episodes | 5,000 |
| Max steps per episode | 200 |
| Smoothing window | 100 |
| Random seeds | 0, 1, 2, 3, 4 |

epsilon decay는 step마다 적용하지 않고, episode가 끝난 뒤 1회만 적용하였다. 이를 통해 학습 초반에 다양한 경로와 주유 행동을 충분히 탐색할 수 있도록 하였다.

---

## 8. 실험 설계

전체 실험 구조는 다음과 같다.

```text
3개 연료 조건 × 2개 알고리즘 × 5개 random seed
= 총 30회 실행
```

### 평가 지표

| 지표 | 설명 |
|---|---|
| Episode reward | episode별 누적 보상 |
| Success rate | 승객을 목적지에 성공적으로 내려준 비율 |
| Average steps | 성공 episode에서 사용한 평균 step 수 |
| 95% confidence interval | 5개 random seed 결과를 바탕으로 계산한 신뢰구간 |
| Convergence episode | 안정적 성공률에 도달한 것으로 판단되는 episode |

최종 성능 요약은 마지막 500개 episode와 5개 random seed 결과를 기준으로 계산하였다.

---

## 9. 실험 결과

### 9.1 종합 결과

| 연료 조건 | 알고리즘 | 성공률 | 평균 step(성공) | 수렴 episode |
|---|---|---:|---:|---|
| No Fuel | Q-Learning | 100.0% | 13.2 | 855 |
| No Fuel | SARSA | 100.0% | 13.2 | 1,050 |
| Generous | Q-Learning | 1.1% | 122.4 | 미수렴 |
| Generous | SARSA | 1.4% | 110.7 | 미수렴 |
| Tight | Q-Learning | 0.0% | 200.0 | 미수렴 |
| Tight | SARSA | 0.0% | 200.0 | 미수렴 |

### 9.2 결과 해석

실험 결과, 연료 제약이 없는 `No Fuel` 조건에서는 Q-Learning과 SARSA 모두 100.0%의 성공률을 보였으며, 평균 step도 13.2로 매우 낮았다. 이는 기본 Taxi 환경에서는 두 알고리즘 모두 충분히 안정적으로 수렴했음을 의미한다.

반면 `Generous` 조건에서는 최대 연료가 50으로 비교적 여유 있게 설정되었음에도 성공률이 Q-Learning 1.1%, SARSA 1.4%에 그쳤다. 평균 step 역시 각각 122.4, 110.7로 증가하였다. 이는 연료 조건이 추가되면서 상태공간이 500개에서 25,500개로 증가하고, 행동공간도 6개에서 7개로 늘어나 Q-table 기반 알고리즘이 5,000 episode 안에 충분히 수렴하지 못했음을 보여준다.

`Tight` 조건에서는 두 알고리즘 모두 성공률이 0.0%였고, 평균 step은 200.0으로 나타났다. 이는 대부분의 episode가 성공하지 못한 채 최대 step에 도달했음을 의미한다.

따라서 연료 제약이 효율적 경로 학습을 유도한다는 가설은 본 실험 결과에서는 지지되지 않았다. 다만 연료 제약이 강해질수록 성공률이 낮아지는 현상은 확인되었다.

---

## 10. 결론

본 프로젝트의 결론은 다음과 같다.

1. 기본 Taxi 환경에서는 Q-Learning과 SARSA 모두 안정적으로 수렴하였다.
2. 연료 제약을 추가하면 상태공간과 행동공간이 크게 증가하여 Q-table 기반 학습이 어려워졌다.
3. `Generous` 조건에서도 성공률이 1% 내외에 그쳐, 연료 제약이 경로 효율성을 높였다고 보기는 어렵다.
4. `Tight` 조건에서는 성공률이 0%로 나타나, 강한 연료 제약이 성공 가능성을 크게 낮추는 것을 확인하였다.
5. 초기 가설 중 “연료가 빡빡할수록 성공률이 낮아진다”는 부분은 일부 확인되었으나, “연료 제약이 효율적 경로 학습을 유도한다”는 부분은 지지되지 않았다.

최종적으로 본 실험은 다음과 같이 정리할 수 있다.

> Fuel-Constrained Taxi 환경은 원본 Taxi 환경보다 훨씬 어려운 학습 문제이며, 단순 Q-table 기반 Q-Learning과 SARSA는 5,000 episode 안에 충분한 성능을 확보하지 못했다. 연료 제약은 효율성 향상 요인이라기보다 학습 난이도를 증가시키는 요인으로 작용하였다.

---

## 11. 한계 및 개선 방향

### 11.1 Q-table 기반 알고리즘의 탐색 한계

연료 조건이 추가되면서 상태공간이 500개에서 25,500개로 크게 증가하였다. 이로 인해 5,000 episode 안에 충분한 상태-행동 조합을 탐색하기 어려웠다.

### 11.2 연료 제약 조건의 난이도

`Generous` 조건도 성공률이 매우 낮게 나타났다. 이는 연료량 자체보다 상태공간 증가, `REFUEL` 행동 탐색, 주유소 도달 전략 학습이 함께 어려워졌기 때문으로 해석할 수 있다.

### 11.3 Reward shaping의 추가 조정 필요

거리 기반 shaping reward를 추가했지만, 연료 제약 환경에서 충분한 수렴을 유도하지 못했다. 향후에는 다음과 같은 추가 보상 설계를 고려할 수 있다.

- 주유소에 가까워질 때의 보상
- 연료가 낮을 때 주유소 방향 이동 보상
- 불필요한 이동에 대한 추가 패널티
- 성공적인 refuel 이후의 전략적 보상

### 11.4 향후 개선 방향

향후 연구에서는 다음 방법을 적용할 수 있다.

- episode 수 증가
- epsilon decay 속도 완화
- Double Q-Learning 적용
- Dyna-Q 적용
- function approximation 기반 Deep Q-Network 적용
- fuel level binning 재설계
- 주유소 위치와 연료량 조건에 대한 ablation study

---

## 12. 프로젝트 파일 구조

```text
.
├── taxi_v4_fuel_rl.py
├── README.md
├── results.pkl
├── fig_reward_curves.png
├── fig_success_curves.png
├── fig_success_rate.png
├── fig_avg_steps.png
└── fuel_constrained_taxi_rl_report.pptx
```

`results.pkl`과 그래프 파일은 전체 실험 실행 후 자동으로 생성된다.

---

## 13. 설치 방법

필요한 패키지는 다음과 같다.

```bash
pip install gymnasium numpy matplotlib
```

---

## 14. 실행 방법

### 14.1 환경 테스트

```bash
python taxi_v4_fuel_rl.py --test
```

예상 출력 구조는 다음과 같다.

```text
none      state_space=500    actions=6
generous  state_space=25500  actions=7
tight     state_space=25500  actions=7
```

## 15. 생성 결과 파일

전체 실험을 실행하면 다음 파일들이 생성된다.

| 파일명 | 설명 |
|---|---|
| `results.pkl` | 실험 결과 원본 데이터 |
| `fig_reward_curves.png` | episode reward 학습 곡선 |
| `fig_success_curves.png` | success rate 학습 곡선 |
| `fig_success_rate.png` | 마지막 500 episode 기준 성공률 비교 |
| `fig_avg_steps.png` | 성공 episode 기준 평균 step 수 비교 |

---

## 16. 보고서

프로젝트 보고서는 다음 파일로 제공한다.

```text
강화학습의 기초_과제_김진희.pptx
```

보고서에는 다음 내용이 포함된다.

- 프로젝트 주제 및 목표
- 환경 및 문제 정의
- State, Action, Reward 설계
- 강화 학습 알고리즘
- 실험 환경 및 하이퍼파라미터
- 실험 결과 시각화
- 토의 및 결론
- 한계 및 개선 방향

---

## 17. 재현성

본 프로젝트는 다섯 개의 random seed를 사용한다.

```text
0, 1, 2, 3, 4
```

각 seed별 결과를 저장하고, 평균 및 95% 신뢰구간을 계산한다. 이를 통해 단일 실행 결과에 의존하지 않고 실험 결과의 안정성을 높이고자 하였다.

---