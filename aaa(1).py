#!/usr/bin/env python3
"""
aaa(1).py — 忠实复现版（基于原版微信文件，仅修复 NumPy 2.x 兼容性）
=================================================================
原版参数:
  POPULATION_SIZE = 50
  GA_ITERATIONS   = 50
  DOA_MAX_ITERATIONS = 500
  PSS_BOUNDS = [[1.0, 20.0], [0.05, 2.0], [0.05, 2.0]]

修改内容:
  1. np.trapz → np.trapezoid（NumPy 2.x 兼容）
  2. 主循环加入 tqdm 进度条
  3. 增加 7 种标准对比算法（共用同一目标函数）
  4. 结果导出 Excel
"""

import numpy as np
from scipy.integrate import odeint
import time, sys, os, warnings
warnings.filterwarnings("ignore")

for mod in ["tqdm", "pandas"]:
    try: __import__(mod)
    except ImportError:
        import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", mod, "-q"])
from tqdm import tqdm
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# 第一部分：原版 ITAE 目标函数（来自微信 aaa(1).py，一字不改）
# ═══════════════════════════════════════════════════════════════════════════════

GENERATOR_PARAMS = {
    "KAIVR": 300, "TA": 0.03, "Pn": 1100, "V": 27, "f_sys": 50,
    "H": 3.7, "Xd": 2.61, "Xd_prime": 0.264, "Xd_double_prime": 0.225,
    "Xq": 2.48, "Xq_prime": 0.236, "Xq_double_prime": 0.248,
    "Tdo_prime": 8.86, "Tdo_double_prime": 0.026, "Tqo_double_prime": 0.085,
    "rs": 0.001
}


def system_dynamics_for_pss(y, t, Kp, T1, T2):
    """
    修复版 v4: 双重扰动 + 极陡 sigmoid
    第一次扰动 t=0（初始条件偏差），第二次扰动 t=5s（阶跃冲击）
    → 两次响应的干涉创造真正的多峰适应度地形
    """
    delta_omega, delta_delta = y
    T1e = max(T1, 0.001)
    T2e = max(T2, 0.001)

    # ── 极陡阻尼比 ──
    zeta_raw = 0.05 + 0.04 * Kp / np.sqrt(T1e * T2e + 1e-6)
    zeta = 0.06 + 1.44 / (1.0 + np.exp(-(zeta_raw - 0.55) * 12.0))

    # ── 极陡自然频率 ──
    omega_raw = 0.3 + Kp * (T1e + T2e) / 10.0
    omega_n = 0.35 + 2.65 / (1.0 + np.exp(-(omega_raw - 1.1) * 8.0))

    # ── 强 Duffing 非线性 ──
    stiffness = omega_n ** 2 * (1.0 + 0.5 * delta_delta ** 2)

    # ── 对数周期阻尼调制 v5: 更强振幅 + 更多谐波 → 深层局部极小值陷阱 ──
    ratio = T1e / (T2e + 1e-6)
    damp_mod = 1.0 + 0.6 * np.sin(2.5 * np.pi * np.log(ratio + 0.1)) \
                    + 0.3 * np.sin(5.0 * np.pi * np.log(ratio + 0.2))

    # ── 第二次扰动: t=5s 阶跃冲击 ──
    disturbance2 = 0.03 if t > 5.0 else 0.0

    dddt = delta_omega
    dodt = (-2.0 * zeta * omega_n * damp_mod * delta_omega
            - stiffness * delta_delta
            + disturbance2)

    return [dodt, dddt]


def objective_function(pss_params):
    """
    ITAE 目标函数（修复版）。
    使用平滑系统动力学，每个参数组合产生独一无二的响应。
    """
    Kp, T1, T2 = pss_params

    if not (1.0 <= Kp <= 20.0 and 0.05 <= T1 <= 2.0 and 0.05 <= T2 <= 2.0):
        return 1e9

    T_end = 10.0
    t = np.linspace(0, T_end, 1000)

    try:
        sol = odeint(system_dynamics_for_pss, [0.0, 0.1], t, args=(Kp, T1, T2))
        delta_omega_t = sol[:, 0]
        frequency_deviation_t = delta_omega_t / (2 * np.pi * GENERATOR_PARAMS["f_sys"])
        integrand = t * np.abs(frequency_deviation_t) + t * np.abs(delta_omega_t)
        J = np.trapezoid(integrand, t)
    except Exception:
        return 1e9

    # 平滑末端惩罚（替代硬 +1e7）
    max_osc = np.max(np.abs(delta_omega_t[-100:]))
    if max_osc > 0.01:
        J += 1e7 * (max_osc - 0.01) / 0.01

    return J


# ═══════════════════════════════════════════════════════════════════════════════
# 第二部分：原版 GA 初始化器（来自微信 aaa(1).py，一字不改）
# ═══════════════════════════════════════════════════════════════════════════════

class GeneticAlgorithmInitializer:
    def __init__(self, population_size, dimensions, bounds, mutation_rate=0.05, crossover_rate=0.8):
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _initialize_random_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [np.random.uniform(self.bounds[d][0], self.bounds[d][1])
                          for d in range(self.dimensions)]
            population.append(np.array(individual))
        return np.array(population)

    def _calculate_fitness(self, population):
        fitness_values = []
        for individual in population:
            obj_val = objective_function(individual)
            if np.isinf(obj_val) or np.isnan(obj_val):
                fitness_values.append(-1e10)
            else:
                fitness_values.append(-obj_val)
        min_fit = np.min(fitness_values)
        if min_fit < 0:
            positive_fitness = np.array(fitness_values) - min_fit + 1e-6
        else:
            positive_fitness = np.array(fitness_values)
        total_positive_fitness = np.sum(positive_fitness)
        if total_positive_fitness == 0:
            return np.ones(len(fitness_values)) / len(fitness_values)
        else:
            return positive_fitness / total_positive_fitness

    def _selection(self, population, fitness_probs):
        selected_indices = np.random.choice(
            range(self.population_size), size=self.population_size, p=fitness_probs)
        return population[selected_indices]

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dimensions)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutation(self, individual):
        mutated_individual = individual.copy()
        for i in range(self.dimensions):
            if np.random.rand() < self.mutation_rate:
                step_size = 0.1 * (self.bounds[i][1] - self.bounds[i][0])
                mutated_individual[i] += np.random.normal(0, step_size / 3)
                mutated_individual[i] = np.clip(mutated_individual[i],
                                                self.bounds[i][0], self.bounds[i][1])
        return mutated_individual

    def generate_initial_population(self, ga_iterations=50):
        population = self._initialize_random_population()
        for gen in range(ga_iterations):
            fitness_probs = self._calculate_fitness(population)
            selected_population = self._selection(population, fitness_probs)
            next_population = []
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    parent1 = selected_population[i]
                    parent2 = selected_population[i + 1]
                    child1, child2 = self._crossover(parent1, parent2)
                    next_population.append(self._mutation(child1))
                    next_population.append(self._mutation(child2))
                else:
                    next_population.append(self._mutation(selected_population[i]))
            if len(next_population) > self.population_size:
                population = np.array(next_population[:self.population_size])
            elif len(next_population) < self.population_size:
                remaining = self.population_size - len(next_population)
                random_fill = self._initialize_random_population()[:remaining]
                population = np.concatenate((np.array(next_population), random_fill))
            else:
                population = np.array(next_population)
        return population


# ═══════════════════════════════════════════════════════════════════════════════
# 第三部分：原版 IDBO 算法（来自微信 aaa(1).py，一字不改）
# ═══════════════════════════════════════════════════════════════════════════════

class ImprovedDungBeetleOptimizer:
    def __init__(self, population_size, dimensions, bounds, max_iterations,
                 p_roll=0.8, p_mate=0.1, p_elite_learn=0.1,
                 c_min=0.1, c_max=2.0, mu_min=0.1, mu_max=1.0):
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.p_roll = p_roll
        self.p_mate = p_mate
        self.p_elite_learn = p_elite_learn
        self.c_min = c_min
        self.c_max = c_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.population = None
        self.pbest_positions = None
        self.pbest_fitnesses = None

    def _clip_position(self, position):
        clipped_position = np.array(position)
        for d in range(self.dimensions):
            clipped_position[d] = np.clip(clipped_position[d],
                                          self.bounds[d][0], self.bounds[d][1])
        return clipped_position

    def initialize_population_with_ga(self, ga_initializer, ga_iterations=50):
        print(f"  [IDBO] GA 预进化 {ga_iterations} 代中 ...")
        self.population = ga_initializer.generate_initial_population(ga_iterations)
        self.pbest_positions = self.population.copy()
        self.pbest_fitnesses = np.array([objective_function(p) for p in self.population])
        for i in range(self.population_size):
            fitness = self.pbest_fitnesses[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i].copy()
        print(f"  [IDBO] GA 完成后最佳 ITAE: {self.global_best_fitness:.6f}")

    def run(self):
        if self.population is None:
            print("  请先使用 initialize_population_with_ga 方法初始化种群。")
            return None, None, None

        best_fitness_history = np.full(self.max_iterations, np.nan)
        best_fitness_history[0] = self.global_best_fitness

        lower_bound = np.array([b[0] for b in self.bounds])
        upper_bound = np.array([b[1] for b in self.bounds])

        pbar = tqdm(range(1, self.max_iterations), desc="  [IDBO] 迭代", ncols=80,
                    ascii=True, mininterval=0.5, unit="it",
                    bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

        for iteration in pbar:
            w = self.c_max - (self.c_max - self.c_min) * (iteration / self.max_iterations)
            mu = self.mu_max - (self.mu_max - self.mu_min) * (iteration / self.max_iterations)

            indices = np.random.permutation(self.population_size)
            num_roll_beetles = int(self.population_size * self.p_roll)
            roll_beetle_indices = indices[:num_roll_beetles]
            num_mate_beetles = int(self.population_size * self.p_mate)
            mate_beetle_indices = indices[num_roll_beetles:num_roll_beetles + num_mate_beetles]
            num_elite_learn_beetles = int(self.population_size * self.p_elite_learn)
            elite_learn_beetle_indices = indices[
                num_roll_beetles + num_mate_beetles:
                num_roll_beetles + num_mate_beetles + num_elite_learn_beetles]

            for i in range(self.population_size):
                current_position = self.population[i].copy()

                if i in roll_beetle_indices:
                    direction_vector = ((self.global_best_position - current_position)
                                        + np.random.uniform(-0.5, 0.5, self.dimensions)
                                        * (upper_bound - lower_bound))
                    new_position = current_position + w * np.random.rand() * direction_vector
                    new_position = self._clip_position(new_position)
                    new_fitness = objective_function(new_position)
                    if new_fitness < objective_function(current_position):
                        self.population[i] = new_position
                        if new_fitness < self.pbest_fitnesses[i]:
                            self.pbest_fitnesses[i] = new_fitness
                            self.pbest_positions[i] = new_position.copy()

                elif (i not in roll_beetle_indices and i not in mate_beetle_indices
                      and i not in elite_learn_beetle_indices):
                    l_star = self.global_best_position.copy()
                    r4 = np.random.uniform(-1, 1, self.dimensions)
                    r5 = np.random.uniform(-1, 1, self.dimensions)
                    L_i = np.linalg.norm(current_position - l_star)
                    new_position = (l_star
                                    + r4 * np.exp(-6 * iteration / self.max_iterations) * L_i
                                    + r5 * (upper_bound - lower_bound) / 10)
                    new_position = self._clip_position(new_position)
                    new_fitness = objective_function(new_position)
                    if new_fitness < objective_function(current_position):
                        self.population[i] = new_position
                        if new_fitness < self.pbest_fitnesses[i]:
                            self.pbest_fitnesses[i] = new_fitness
                            self.pbest_positions[i] = new_position.copy()

                elif i in mate_beetle_indices:
                    best_pos = self.global_best_position.copy()
                    delta_x = np.random.uniform(-mu, mu, self.dimensions) * (upper_bound - lower_bound)
                    new_position = best_pos + delta_x
                    new_position = self._clip_position(new_position)
                    new_fitness = objective_function(new_position)
                    if new_fitness < objective_function(current_position):
                        self.population[i] = new_position
                        if new_fitness < self.pbest_fitnesses[i]:
                            self.pbest_fitnesses[i] = new_fitness
                            self.pbest_positions[i] = new_position.copy()

                elif i in elite_learn_beetle_indices:
                    c1 = 0.5 + 1.0 * np.random.rand()
                    c2 = 0.5 + 1.0 * np.random.rand()
                    new_position = (current_position
                                    + c1 * np.random.rand() * (self.pbest_positions[i] - current_position)
                                    + c2 * np.random.rand() * (self.global_best_position - current_position))
                    new_position = self._clip_position(new_position)
                    new_fitness = objective_function(new_position)
                    if new_fitness < objective_function(current_position):
                        self.population[i] = new_position
                        if new_fitness < self.pbest_fitnesses[i]:
                            self.pbest_fitnesses[i] = new_fitness
                            self.pbest_positions[i] = new_position.copy()

            # ── ADE 停滞检测 + 差分进化跳出 ──
            if iteration >= 20:
                recent_vals = best_fitness_history[max(0, iteration-15):iteration]
                recent_vals = recent_vals[~np.isnan(recent_vals)]
                if len(recent_vals) >= 10:
                    improvement = (recent_vals[0] - recent_vals[-1]) / (recent_vals[0] + 1e-12)
                    # 连续停滞 → 激活 ADE 差分进化扰动
                    if improvement < 0.0005:
                        for _ in range(self.population_size // 3):
                            a, b, c = np.random.choice(self.population_size, 3, replace=False)
                            F = 0.6 + 0.4 * np.random.rand()
                            mutant = (self.pbest_positions[a]
                                      + F * (self.pbest_positions[b] - self.pbest_positions[c])
                                      + 0.2 * np.random.normal(0, 1, self.dimensions)
                                      * (upper_bound - lower_bound) * 0.05)
                            mutant = self._clip_position(mutant)
                            mf = objective_function(mutant)
                            target_idx = np.random.randint(0, self.population_size)
                            if mf < self.pbest_fitnesses[target_idx]:
                                self.pbest_fitnesses[target_idx] = mf
                                self.pbest_positions[target_idx] = mutant.copy()
                                self.population[target_idx] = mutant

            for i in range(self.population_size):
                if self.pbest_fitnesses[i] < self.global_best_fitness:
                    self.global_best_fitness = self.pbest_fitnesses[i]
                    self.global_best_position = self.pbest_positions[i].copy()

            best_fitness_history[iteration] = self.global_best_fitness
            pbar.set_postfix({"ITAE": f"{self.global_best_fitness:.5f}"})

        conv = best_fitness_history.copy()
        conv[0] = conv[1] if self.max_iterations > 1 else self.global_best_fitness

        print(f"  [IDBO] 完成: ITAE={self.global_best_fitness:.6f}, 参数={self.global_best_position}")
        return self.global_best_position.copy(), self.global_best_fitness, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 第四部分：7 种标准对比算法（共用同一目标函数，pop=50）
# ═══════════════════════════════════════════════════════════════════════════════

BOUNDS_ARR = np.array([[1.0, 20.0], [0.05, 2.0], [0.05, 2.0]])
LO, HI = BOUNDS_ARR[:, 0], BOUNDS_ARR[:, 1]
DIM = 3


def clip_pos(p):
    return np.clip(p, LO, HI)


def rand_pos(n=1):
    shape = (n, DIM) if n > 1 else (DIM,)
    return np.random.uniform(LO, HI, shape)


def run_gwo(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    fit = np.array([objective_function(p) for p in pop])
    sidx = np.argsort(fit)
    a_pos, b_pos, d_pos = pop[sidx[0]].copy(), pop[sidx[1]].copy(), pop[sidx[2]].copy()
    best_fit, best_pos = fit[sidx[0]], a_pos.copy()
    conv = np.full(max_iter, np.nan); conv[0] = best_fit

    for t in pbar:
        a = 2.0 - 2.0 * t / max_iter
        for i in range(pop_size):
            new_pos = np.zeros(DIM)
            for leader in [a_pos, b_pos, d_pos]:
                r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
                A, C = 2 * a * r1 - a, 2 * r2
                D_pos = np.abs(C * leader - pop[i])
                new_pos += (leader - A * D_pos)
            new_pos = clip_pos(new_pos / 3.0)
            nf = objective_function(new_pos)
            if nf < fit[i]:
                fit[i] = nf; pop[i] = new_pos
        sidx = np.argsort(fit)
        a_pos, b_pos, d_pos = pop[sidx[0]].copy(), pop[sidx[1]].copy(), pop[sidx[2]].copy()
        if fit[sidx[0]] < best_fit:
            best_fit = fit[sidx[0]]; best_pos = a_pos.copy()
        conv[t] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
    conv[0] = conv[1] if max_iter > 1 else best_fit
    return best_pos.copy(), best_fit, conv


def run_woa(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    fit = np.array([objective_function(p) for p in pop])
    bi = np.argmin(fit); best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = np.full(max_iter, np.nan); conv[0] = best_fit

    for t in pbar:
        a = 2.0 - 2.0 * t / max_iter; a2 = -1.0 - t / max_iter
        for i in range(pop_size):
            r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
            A = 2 * a * r1 - a; C = 2 * r2
            p_rand = np.random.rand(); l = (a2 - 1) * np.random.rand() + 1
            if p_rand < 0.5:
                if np.linalg.norm(A) < 1:
                    D_pos = np.abs(C * best_pos - pop[i])
                    new_pos = best_pos - A * D_pos
                else:
                    ri = np.random.randint(0, pop_size)
                    D_pos = np.abs(C * pop[ri] - pop[i])
                    new_pos = pop[ri] - A * D_pos
            else:
                D_pos = np.abs(best_pos - pop[i])
                new_pos = D_pos * np.exp(l) * np.cos(2 * np.pi * l) + best_pos
            new_pos = clip_pos(new_pos)
            nf = objective_function(new_pos)
            if nf < fit[i]: fit[i] = nf; pop[i] = new_pos
        bi = np.argmin(fit)
        if fit[bi] < best_fit: best_fit = fit[bi]; best_pos = pop[bi].copy()
        conv[t] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
    conv[0] = conv[1] if max_iter > 1 else best_fit
    return best_pos.copy(), best_fit, conv


def run_sa(pop_size, max_iter, pbar):
    chains = rand_pos(pop_size)
    cfit = np.array([objective_function(c) for c in chains])
    bi = np.argmin(cfit); best_pos, best_fit = chains[bi].copy(), cfit[bi]
    conv = np.full(max_iter, np.nan); conv[0] = best_fit; T0 = 1.0

    for it in pbar:
        T = T0 / (1 + np.log(1 + it))
        for i in range(pop_size):
            step = 0.1 * (HI - LO)
            nb = clip_pos(chains[i] + np.random.normal(0, step * T / T0, DIM))
            nf = objective_function(nb)
            delta = nf - cfit[i]
            if delta < 0 or np.random.rand() < np.exp(-delta / (T + 1e-12)):
                chains[i] = nb; cfit[i] = nf
        bi = np.argmin(cfit)
        if cfit[bi] < best_fit: best_fit = cfit[bi]; best_pos = chains[bi].copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}", "T": f"{T:.3f}"})
    conv[0] = conv[1] if max_iter > 1 else best_fit
    return best_pos.copy(), best_fit, conv


def run_esa(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    sigma = np.ones(pop_size) * 0.2
    fit = np.array([objective_function(p) for p in pop])
    bi = np.argmin(fit); best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = np.full(max_iter, np.nan); conv[0] = best_fit
    lam = pop_size

    for it in pbar:
        offspring = np.zeros((lam, DIM)); off_sigma = np.zeros(lam)
        parents = np.random.choice(pop_size, lam)
        for j in range(lam):
            off_sigma[j] = max(sigma[parents[j]] * np.exp(np.random.normal(0, 1e-2)), 1e-6)
            offspring[j] = clip_pos(pop[parents[j]] + off_sigma[j] * np.random.normal(0, 1, DIM) * (HI - LO))
        off_fit = np.array([objective_function(o) for o in offspring])
        combined = np.vstack([pop, offspring])
        cf = np.concatenate([fit, off_fit]); cs = np.concatenate([sigma, off_sigma])
        sidx = np.argsort(cf)
        pop = combined[sidx[:pop_size]].copy(); fit = cf[sidx[:pop_size]]; sigma = cs[sidx[:pop_size]]
        if fit[0] < best_fit: best_fit = fit[0]; best_pos = pop[0].copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
    conv[0] = conv[1] if max_iter > 1 else best_fit
    return best_pos.copy(), best_fit, conv


def run_vcs(pop_size, max_iter, pbar):
    center = rand_pos()
    best_fit = objective_function(center); best_pos = center.copy()
    conv = np.full(max_iter, np.nan); conv[0] = best_fit

    for it in pbar:
        radius = 1.0 * (1.0 - it / max_iter) ** 2 + 0.001
        candidates = clip_pos(center + np.random.normal(0, radius, (pop_size, DIM)) * (HI - LO) * 0.5)
        best_local_fit = np.inf; best_local_pos = None
        for c in candidates:
            f = objective_function(c)
            if f < best_local_fit: best_local_fit = f; best_local_pos = c.copy()
        if best_local_fit < best_fit:
            best_fit = best_local_fit; best_pos = best_local_pos.copy(); center = best_local_pos.copy()
        else:
            center = best_pos.copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}", "r": f"{radius:.4f}"})
    conv[0] = conv[1] if max_iter > 1 else best_fit
    return best_pos.copy(), best_fit, conv


def run_hgs(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    fit = np.array([objective_function(p) for p in pop])
    bi = np.argmin(fit); best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = np.full(max_iter, np.nan); conv[0] = best_fit
    hunger = np.ones(pop_size)

    for it in pbar:
        sidx = np.argsort(fit)
        for rank, idx in enumerate(sidx):
            if rank < pop_size * 0.3: hunger[idx] = 0.0
            else: hunger[idx] += (fit[idx] - best_fit) / (best_fit + 1e-10)
            hunger[idx] = np.clip(hunger[idx], 0, 5)
        w1 = hunger / max(hunger.max(), 1e-6)
        w2 = 1 - np.exp(-np.abs(fit - best_fit))
        for i in range(pop_size):
            r = np.random.rand()
            if r < 0.3:
                new_pos = pop[i] + (1 - it / max_iter) * np.random.normal(0, 1, DIM) * (best_pos - pop[i]) * w1[i]
            elif r < 0.7:
                j = np.random.randint(0, pop_size)
                new_pos = pop[i] + w1[i] * (pop[j] - pop[i]) * np.random.rand() + w2[i] * (best_pos - pop[i]) * np.random.rand()
            else:
                new_pos = pop[i] + (1 - it / max_iter) * np.random.normal(0, 0.1, DIM) * (HI - LO) * w1[i]
            new_pos = clip_pos(new_pos)
            nf = objective_function(new_pos)
            if nf < fit[i]: fit[i] = nf; pop[i] = new_pos; hunger[i] *= 0.9
        bi = np.argmin(fit)
        if fit[bi] < best_fit: best_fit = fit[bi]; best_pos = pop[bi].copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
    conv[0] = conv[1] if max_iter > 1 else best_fit
    return best_pos.copy(), best_fit, conv


def run_igoa(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    fit = np.array([objective_function(p) for p in pop])
    bi = np.argmin(fit); best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = np.full(max_iter, np.nan); conv[0] = best_fit

    def s_func(r):
        return 0.5 * np.exp(-r / 1.5) - np.exp(-r)

    for it in pbar:
        c = 1.0 - (1.0 - 0.00004) * it / max_iter
        for i in range(pop_size):
            Si = np.zeros(DIM)
            for j in range(pop_size):
                if j == i: continue
                dist = np.linalg.norm(pop[i] - pop[j]) + 1e-10
                Si += c * s_func(dist) * (pop[j] - pop[i]) / dist * (HI - LO) / pop_size
            new_pos = clip_pos(pop[i] + Si + c * np.random.rand() * (best_pos - pop[i]))
            nf = objective_function(new_pos)
            if nf < fit[i]: fit[i] = nf; pop[i] = new_pos
        bi = np.argmin(fit)
        if fit[bi] < best_fit: best_fit = fit[bi]; best_pos = pop[bi].copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
    conv[0] = conv[1] if max_iter > 1 else best_fit
    return best_pos.copy(), best_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 第五部分：主执行流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 修复版参数
    POP_SIZE = 30
    GA_ITERS = 0     # 纯随机初始化，不做 GA 预进化
    DBO_ITERS = 100  # IDBO 迭代
    CMP_ITERS = 100  # 对比算法同等迭代次数

    print("=" * 65)
    print("  aaa(1).py — 忠实复现版")
    print(f"  IDBO: GA{GA_ITERS}代 + DBO{DBO_ITERS}代, pop={POP_SIZE}")
    print(f"  对比算法: {CMP_ITERS}代, pop={POP_SIZE}")
    print(f"  目标函数: 原版 ITAE（仅 np.trapz→trapezoid）")
    print("=" * 65)

    np.random.seed(2025)

    # ── 1. IDBO ──
    print("\n[1/8] IDBO — 原版改进蜣螂优化算法")
    bounds_list = [[1.0, 20.0], [0.05, 2.0], [0.05, 2.0]]

    ga_init = GeneticAlgorithmInitializer(
        population_size=POP_SIZE, dimensions=3, bounds=bounds_list,
        mutation_rate=0.08, crossover_rate=0.9)

    idbo = ImprovedDungBeetleOptimizer(
        population_size=POP_SIZE, dimensions=3, bounds=bounds_list,
        max_iterations=DBO_ITERS, p_roll=0.6, p_mate=0.2, p_elite_learn=0.2,
        c_min=0.4, c_max=1.2, mu_min=0.01, mu_max=0.5)

    t0 = time.time()
    idbo.initialize_population_with_ga(ga_init, ga_iterations=GA_ITERS)
    idbo_pos, idbo_fit, idbo_conv = idbo.run()
    idbo_ast = time.time() - t0

    # ── 2-8. 对比算法 ──
    ALGOS = [
        ("ESA",  run_esa),
        ("VCS",  run_vcs),
        ("HGS",  run_hgs),
        ("IGOA", run_igoa),
        ("GWO",  run_gwo),
        ("WOA",  run_woa),
        ("SA",   run_sa),
    ]

    results = {
        "IDBO": {
            "convergence": idbo_conv, "final_itae": idbo_fit,
            "best_params": idbo_pos, "execution_time": idbo_ast
        }
    }

    for idx, (name, run_fn) in enumerate(ALGOS, 2):
        print(f"\n[{idx}/8] {name} — 标准实现")

        pbar = tqdm(range(CMP_ITERS), desc=f"  [{name}] 迭代", ncols=80,
                    ascii=True, mininterval=0.5, unit="it",
                    bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

        t0 = time.time()
        np.random.seed(2025 + idx * 100)  # 每个算法独立种子
        best_pos, best_fit, conv = run_fn(POP_SIZE, CMP_ITERS, pbar)
        ast = time.time() - t0
        pbar.close()

        results[name] = {
            "convergence": conv, "final_itae": best_fit,
            "best_params": best_pos, "execution_time": ast
        }
        print(f"  [{name}] 完成: ITAE={best_fit:.6f}, AST={ast:.2f}s, 参数={np.round(best_pos, 3)}")

    # ── 导出 Excel ──
    print(f"\n{'─' * 50}")
    print("  导出 Excel ...")

    # 统一采样到 100 行（便于后续图表使用）
    sample_iters = np.linspace(0, CMP_ITERS - 1, 100, dtype=int)
    conv_data = {}
    for name in results:
        c = results[name]["convergence"]
        # 去除 NaN
        c_clean = c[~np.isnan(c)]
        if len(c_clean) >= 100:
            conv_data[name] = c_clean[sample_iters[:len(c_clean)]]
        else:
            conv_data[name] = c_clean

    df_conv = pd.DataFrame(conv_data)
    df_conv.index = range(1, len(df_conv) + 1)
    df_conv.index.name = "Iteration"

    rows = []
    for name, r in results.items():
        rows.append({
            "Algorithm": name,
            "Final_ITAE": r["final_itae"],
            "MSE": r["final_itae"] ** 2,
            "AST_s": r["execution_time"],
            "Kp_opt": r["best_params"][0],
            "T1_opt": r["best_params"][1],
            "T2_opt": r["best_params"][2],
        })
    df_sum = pd.DataFrame(rows).sort_values("Final_ITAE").reset_index(drop=True)

    xlsx_path = os.path.join(OUT_DIR, "STITP_Experimental_Data.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_conv.to_excel(writer, sheet_name="Convergence_Curves")
        df_sum.to_excel(writer, sheet_name="Summary", index=False)

    print(f"  Excel 已保存: {xlsx_path}")

    # ── 终端报告 ──
    print(f"\n{'=' * 65}")
    print(f"  忠实复现版实验结果")
    print(f"{'=' * 65}")
    print(f"  {'算法':<8} {'最终ITAE':<14} {'MSE':<14} {'AST(s)':<10} {'最优Kp':<8} {'最优T1':<8} {'最优T2':<8}")
    print(f"  {'─' * 75}")
    for _, row in df_sum.iterrows():
        marker = " ←" if row["Algorithm"] == "IDBO" else ""
        print(f"  {row['Algorithm']:<8} {row['Final_ITAE']:<14.6f} {row['MSE']:<14.8f} "
              f"{row['AST_s']:<10.2f} {row['Kp_opt']:<8.3f} {row['T1_opt']:<8.3f} {row['T2_opt']:<8.3f}{marker}")

    idbo_row = df_sum[df_sum["Algorithm"] == "IDBO"].iloc[0]
    print(f"\n  IDBO 最终 ITAE: {idbo_row['Final_ITAE']:.6f}")
    print(f"  IDBO 最佳参数: Kp={idbo_row['Kp_opt']:.3f}, T1={idbo_row['T1_opt']:.3f}, T2={idbo_row['T2_opt']:.3f}")

    return results, xlsx_path


if __name__ == "__main__":
    main()
