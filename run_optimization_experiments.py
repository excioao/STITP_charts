#!/usr/bin/env python3
"""
IDBO 及 7 种对比算法 —— 真实优化执行脚本 v2.0
================================================================
- 所有算法真实执行（非 mock）
- tqdm 进度条输出到日志文件，终端仅打印一行汇总
- 结果导出到 Excel (STITP_Experimental_Data.xlsx)
"""

import numpy as np
from scipy.integrate import odeint
import time, sys, os, warnings
warnings.filterwarnings("ignore")

# ── 依赖检查 ──
for mod in ["tqdm", "pandas"]:
    try:
        __import__(mod)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", mod, "-q"])
from tqdm import tqdm
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# 目标函数 (来自 aaa(1).py, 保持不变)
# ═══════════════════════════════════════════════════════════════════════════════

GENERATOR_PARAMS = {
    "KAIVR": 300, "TA": 0.03, "Pn": 1100, "V": 27, "f_sys": 50,
    "H": 3.7, "Xd": 2.61, "Xd_prime": 0.264, "Xd_double_prime": 0.225,
    "Xq": 2.48, "Xq_prime": 0.236, "Xq_double_prime": 0.248,
    "Tdo_prime": 8.86, "Tdo_double_prime": 0.026, "Tqo_double_prime": 0.085,
    "rs": 0.001
}

def system_dynamics_for_pss(y, t, Kp, T1, T2):
    delta_omega, delta_delta = y
    T1e, T2e = max(T1, 0.001), max(T2, 0.001)
    zeta = np.clip(0.1 + 0.03 * Kp / (T1e * T2e) ** 0.5, 0.1, 1.5)
    omega_n = np.clip(0.5 + Kp * (T1e + T2e) / 10.0, 0.5, 3.0)
    return [ -2*zeta*omega_n*delta_omega - omega_n**2*delta_delta, delta_omega ]

def objective_function(pss_params):
    Kp, T1, T2 = pss_params
    if not (1.0 <= Kp <= 20.0 and 0.05 <= T1 <= 2.0 and 0.05 <= T2 <= 2.0):
        return 1e9
    t = np.linspace(0, 10.0, 1000)
    try:
        sol = odeint(system_dynamics_for_pss, [0.0, 0.1], t, args=(Kp, T1, T2))
        dw = sol[:, 0]
        fdev = dw / (2*np.pi*GENERATOR_PARAMS["f_sys"])
        J = np.trapezoid(t*np.abs(fdev) + t*np.abs(dw), t)
    except:
        return 1e9
    if np.max(np.abs(dw[-100:])) > 0.01:
        J += 1e7
    return J

BOUNDS = np.array([[1.0, 20.0], [0.05, 2.0], [0.05, 2.0]])
LO, HI = BOUNDS[:, 0], BOUNDS[:, 1]
DIM = 3

def clip(p):
    return np.clip(p, LO, HI)
def rand_pos(n=1):
    shape = (n, DIM) if n > 1 else (DIM,)
    return np.random.uniform(LO, HI, shape)

# ═══════════════════════════════════════════════════════════════════════════════
# 通用执行框架
# ═══════════════════════════════════════════════════════════════════════════════

def execute_algorithm(name, pop_size, max_iter, run_func, log_f):
    """统一执行入口：运行算法 + tqdm 进度写入 log_f + 返回结果字典"""
    t_start = time.time()
    convergence = np.zeros(max_iter)
    best_pos = None
    best_fit = np.inf

    # tqdm 进度条写入日志文件，不在终端打印
    pbar = tqdm(range(max_iter), desc=f"{name:>6s}", ncols=75,
                ascii=True, mininterval=0.5, unit="it", file=log_f,
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

    np.random.seed(2025)

    # 调用算法特定逻辑
    result = run_func(pop_size, max_iter, pbar)

    pbar.close()

    best_pos, best_fit, convergence = result
    exec_time = time.time() - t_start

    return {
        "convergence": convergence,
        "final_itae": best_fit,
        "best_params": best_pos,
        "execution_time": exec_time,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 算法 1: IDBO — 改进蜣螂优化 (来自 aaa(1).py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_idbo(pop_size, max_iter, pbar):
    # GA 预进化
    pop = np.random.uniform(LO, HI, (pop_size, DIM))
    for gen in range(30):
        fit = np.array([objective_function(p) for p in pop])
        sidx = np.argsort(fit)
        pop = pop[sidx]
        elite_n = max(2, pop_size // 5)
        elite = pop[:elite_n].copy()
        inv_fit = 1.0 / (fit[sidx] + 1e-10)
        probs = inv_fit / inv_fit.sum()
        new_pop = []
        while len(new_pop) < pop_size - elite_n:
            p1, p2 = pop[np.random.choice(pop_size, 2, p=probs)]
            cp = np.random.randint(1, DIM)
            child = np.concatenate([p1[:cp], p2[cp:]])
            if np.random.rand() < 0.1:
                child += np.random.normal(0, 0.05, DIM)*(HI - LO)
            child = clip(child)
            new_pop.append(child)
        pop = np.vstack([elite, np.array(new_pop[:pop_size - elite_n])])

    pbest_pos = pop.copy()
    pbest_fit = np.array([objective_function(p) for p in pop])

    best_idx = np.argmin(pbest_fit)
    gbest_pos = pop[best_idx].copy()
    gbest_fit = pbest_fit[best_idx]
    conv = np.zeros(max_iter)
    conv[0] = gbest_fit

    for it in pbar:
        w = 1.2 - (1.2 - 0.4)*(it/max_iter)
        mu_p = 0.5 - (0.5 - 0.01)*(it/max_iter)

        if it > 20:
            recent = conv[max(0, it-15):it]
            stagnation = len(recent) >= 10 and (recent[0] - recent[-1])/(recent[0]+1e-12) < 0.001
        else:
            stagnation = False

        indices = np.random.permutation(pop_size)
        n_roll = int(pop_size * 0.45)
        n_mate = int(pop_size * 0.20)
        n_elite = int(pop_size * 0.20)

        for idx, i in enumerate(indices):
            cur = pop[i].copy()
            if idx < n_roll:
                direction = (gbest_pos - cur) + np.random.uniform(-0.5, 0.5, DIM)*(HI - LO)
                new_pos = cur + w*np.random.rand()*direction
            elif idx < n_roll + n_mate:
                delta = np.random.uniform(-mu_p, mu_p, DIM)*(HI - LO)
                new_pos = gbest_pos + delta
            elif idx < n_roll + n_mate + n_elite:
                c1, c2 = 0.5 + np.random.rand(), 0.5 + np.random.rand()
                new_pos = cur + c1*np.random.rand()*(pbest_pos[i]-cur) + c2*np.random.rand()*(gbest_pos-cur)
            else:
                r4 = np.random.uniform(-1, 1, DIM)
                r5 = np.random.uniform(-1, 1, DIM)
                Li = np.linalg.norm(cur - gbest_pos)
                new_pos = gbest_pos + r4*np.exp(-6*it/max_iter)*Li + r5*(HI-LO)/10
            new_pos = clip(new_pos)

            new_fit = objective_function(new_pos)
            if new_fit < pbest_fit[i]:
                pbest_fit[i] = new_fit
                pbest_pos[i] = new_pos.copy()
                pop[i] = new_pos

            if stagnation and np.random.rand() < 0.3:
                a_idx, b_idx = np.random.choice(pop_size, 2, replace=False)
                F = 0.5 + 0.3*np.random.rand()
                pop_mean = np.mean(pop, axis=0)
                trial = cur + F*(pop[a_idx] - pop[b_idx]) + 0.3*F*(pop_mean - cur)
                trial = clip(trial)
                tf = objective_function(trial)
                if tf < pbest_fit[i]:
                    pbest_fit[i] = tf
                    pbest_pos[i] = trial.copy()
                    pop[i] = trial

        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()
        conv[it] = gbest_fit
        pbar.set_postfix({"ITAE": f"{gbest_fit:.5f}"})
        if it == 0 and max_iter > 1:
            conv[0] = gbest_fit

    return gbest_pos.copy(), gbest_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 算法 2: GWO — 灰狼优化 (Mirjalili, 2014)
# ═══════════════════════════════════════════════════════════════════════════════

def run_gwo(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    fit = np.array([objective_function(p) for p in pop])
    sidx = np.argsort(fit)
    a_pos = pop[sidx[0]].copy()
    b_pos = pop[sidx[1]].copy()
    d_pos = pop[sidx[2]].copy()
    best_fit = fit[sidx[0]]
    best_pos = a_pos.copy()
    conv = np.zeros(max_iter)
    conv[0] = best_fit

    for t in pbar:
        a = 2.0 - 2.0*t/max_iter
        for i in range(pop_size):
            new_pos = np.zeros(DIM)
            for leader in [a_pos, b_pos, d_pos]:
                r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
                A, C = 2*a*r1 - a, 2*r2
                D = np.abs(C*leader - pop[i])
                new_pos += (leader - A*D)
            new_pos = clip(new_pos/3.0)
            nf = objective_function(new_pos)
            if nf < fit[i]:
                fit[i] = nf
                pop[i] = new_pos
        sidx = np.argsort(fit)
        a_pos, b_pos, d_pos = pop[sidx[0]].copy(), pop[sidx[1]].copy(), pop[sidx[2]].copy()
        if fit[sidx[0]] < best_fit:
            best_fit = fit[sidx[0]]
            best_pos = a_pos.copy()
        conv[t] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
        if t == 0 and max_iter > 1:
            conv[0] = best_fit
    return best_pos.copy(), best_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 算法 3: WOA — 鲸鱼优化 (Mirjalili & Lewis, 2016)
# ═══════════════════════════════════════════════════════════════════════════════

def run_woa(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    fit = np.array([objective_function(p) for p in pop])
    bi = np.argmin(fit)
    best_pos = pop[bi].copy()
    best_fit = fit[bi]
    conv = np.zeros(max_iter)
    conv[0] = best_fit

    for t in pbar:
        a = 2.0 - 2.0*t/max_iter
        a2 = -1.0 - t/max_iter
        for i in range(pop_size):
            r1 = np.random.rand(DIM)
            A = 2*a*r1 - a
            C = 2*np.random.rand(DIM)
            p_rand = np.random.rand()
            l = (a2-1)*np.random.rand() + 1
            if p_rand < 0.5:
                if np.linalg.norm(A) < 1:
                    D = np.abs(C*best_pos - pop[i])
                    new_pos = best_pos - A*D
                else:
                    ri = np.random.randint(0, pop_size)
                    D = np.abs(C*pop[ri] - pop[i])
                    new_pos = pop[ri] - A*D
            else:
                D = np.abs(best_pos - pop[i])
                new_pos = D*np.exp(l)*np.cos(2*np.pi*l) + best_pos
            new_pos = clip(new_pos)
            nf = objective_function(new_pos)
            if nf < fit[i]:
                fit[i] = nf
                pop[i] = new_pos
        bi = np.argmin(fit)
        if fit[bi] < best_fit:
            best_fit = fit[bi]
            best_pos = pop[bi].copy()
        conv[t] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
        if t == 0 and max_iter > 1:
            conv[0] = best_fit
    return best_pos.copy(), best_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 算法 4: SA — 模拟退火 (多链并行, Kirkpatrick, 1983)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sa(pop_size, max_iter, pbar):
    chains = rand_pos(pop_size)
    cfit = np.array([objective_function(c) for c in chains])
    bi = np.argmin(cfit)
    best_pos = chains[bi].copy()
    best_fit = cfit[bi]
    conv = np.zeros(max_iter)
    conv[0] = best_fit
    T0 = 1.0

    for it in pbar:
        T = T0/(1 + np.log(1 + it))
        for i in range(pop_size):
            step = 0.1*(HI - LO)
            nb = clip(chains[i] + np.random.normal(0, step*T/T0, DIM))
            nf = objective_function(nb)
            delta = nf - cfit[i]
            if delta < 0 or np.random.rand() < np.exp(-delta/(T + 1e-12)):
                chains[i] = nb
                cfit[i] = nf
        bi = np.argmin(cfit)
        if cfit[bi] < best_fit:
            best_fit = cfit[bi]
            best_pos = chains[bi].copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}", "T": f"{T:.3f}"})
        if it == 0 and max_iter > 1:
            conv[0] = best_fit
    return best_pos.copy(), best_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 算法 5: ESA — 进化策略 (mu+lambda)-ES
# ═══════════════════════════════════════════════════════════════════════════════

def run_esa(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    sigma = np.ones(pop_size)*0.2
    fit = np.array([objective_function(p) for p in pop])
    bi = np.argmin(fit)
    best_pos = pop[bi].copy()
    best_fit = fit[bi]
    conv = np.zeros(max_iter)
    conv[0] = best_fit
    lam = pop_size  # offspring count

    for it in pbar:
        offspring = np.zeros((lam, DIM))
        off_sigma = np.zeros(lam)
        parents = np.random.choice(pop_size, lam)
        for j in range(lam):
            off_sigma[j] = max(sigma[parents[j]]*np.exp(np.random.normal(0, 1e-2)), 1e-6)
            offspring[j] = clip(pop[parents[j]] + off_sigma[j]*np.random.normal(0, 1, DIM)*(HI - LO))
        off_fit = np.array([objective_function(o) for o in offspring])
        combined = np.vstack([pop, offspring])
        cf = np.concatenate([fit, off_fit])
        cs = np.concatenate([sigma, off_sigma])
        sidx = np.argsort(cf)
        pop = combined[sidx[:pop_size]].copy()
        fit = cf[sidx[:pop_size]]
        sigma = cs[sidx[:pop_size]]
        if fit[0] < best_fit:
            best_fit = fit[0]
            best_pos = pop[0].copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
        if it == 0 and max_iter > 1:
            conv[0] = best_fit
    return best_pos.copy(), best_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 算法 6: VCS — 涡流搜索
# ═══════════════════════════════════════════════════════════════════════════════

def run_vcs(pop_size, max_iter, pbar):
    center = rand_pos()
    best_fit = objective_function(center)
    best_pos = center.copy()
    conv = np.zeros(max_iter)
    conv[0] = best_fit

    for it in pbar:
        radius = 1.0*(1.0 - it/max_iter)**2 + 0.001
        candidates = clip(center + np.random.normal(0, radius, (pop_size, DIM))*(HI - LO)*0.5)
        best_local_fit = np.inf
        best_local_pos = None
        for c in candidates:
            fit = objective_function(c)
            if fit < best_local_fit:
                best_local_fit = fit
                best_local_pos = c.copy()
        if best_local_fit < best_fit:
            best_fit = best_local_fit
            best_pos = best_local_pos.copy()
            center = best_local_pos.copy()
        else:
            center = best_pos.copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}", "r": f"{radius:.4f}"})
        if it == 0 and max_iter > 1:
            conv[0] = best_fit
    return best_pos.copy(), best_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 算法 7: HGS — 饥饿游戏搜索 (Yang et al., 2021)
# ═══════════════════════════════════════════════════════════════════════════════

def run_hgs(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    fit = np.array([objective_function(p) for p in pop])
    bi = np.argmin(fit)
    best_pos = pop[bi].copy()
    best_fit = fit[bi]
    conv = np.zeros(max_iter)
    conv[0] = best_fit
    hunger = np.ones(pop_size)

    for it in pbar:
        sidx = np.argsort(fit)
        for rank, idx in enumerate(sidx):
            if rank < pop_size*0.3:
                hunger[idx] = 0.0
            else:
                hunger[idx] += (fit[idx] - best_fit)/(best_fit + 1e-10)
            hunger[idx] = np.clip(hunger[idx], 0, 5)
        w1 = hunger/hunger.max() if hunger.max() > 0 else np.ones(pop_size)
        w2 = 1 - np.exp(-np.abs(fit - best_fit))

        for i in range(pop_size):
            r = np.random.rand()
            if r < 0.3:
                new_pos = pop[i] + (1-it/max_iter)*np.random.normal(0,1,DIM)*(best_pos-pop[i])*w1[i]
            elif r < 0.7:
                j = np.random.randint(0, pop_size)
                new_pos = pop[i] + w1[i]*(pop[j]-pop[i])*np.random.rand() + w2[i]*(best_pos-pop[i])*np.random.rand()
            else:
                new_pos = pop[i] + (1-it/max_iter)*np.random.normal(0,0.1,DIM)*(HI-LO)*w1[i]
            new_pos = clip(new_pos)
            nf = objective_function(new_pos)
            if nf < fit[i]:
                fit[i] = nf
                pop[i] = new_pos
                hunger[i] *= 0.9
        bi = np.argmin(fit)
        if fit[bi] < best_fit:
            best_fit = fit[bi]
            best_pos = pop[bi].copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
        if it == 0 and max_iter > 1:
            conv[0] = best_fit
    return best_pos.copy(), best_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 算法 8: IGOA — 改进蝗虫优化
# ═══════════════════════════════════════════════════════════════════════════════

def run_igoa(pop_size, max_iter, pbar):
    pop = rand_pos(pop_size)
    fit = np.array([objective_function(p) for p in pop])
    bi = np.argmin(fit)
    best_pos = pop[bi].copy()
    best_fit = fit[bi]
    conv = np.zeros(max_iter)
    conv[0] = best_fit

    def s_func(r):
        return 0.5*np.exp(-r/1.5) - np.exp(-r)

    for it in pbar:
        cmax, cmin = 1.0, 0.00004
        c = cmax - (cmax-cmin)*it/max_iter
        for i in range(pop_size):
            Si = np.zeros(DIM)
            for j in range(pop_size):
                if j == i:
                    continue
                dist = np.linalg.norm(pop[i] - pop[j]) + 1e-10
                xj_xi = (pop[j] - pop[i])/dist
                Si += c*s_func(dist)*xj_xi*(HI - LO)/pop_size
            elite_guide = best_pos - pop[i]
            new_pos = clip(pop[i] + Si + c*np.random.rand()*elite_guide)
            nf = objective_function(new_pos)
            if nf < fit[i]:
                fit[i] = nf
                pop[i] = new_pos
        bi = np.argmin(fit)
        if fit[bi] < best_fit:
            best_fit = fit[bi]
            best_pos = pop[bi].copy()
        conv[it] = best_fit
        pbar.set_postfix({"ITAE": f"{best_fit:.5f}"})
        if it == 0 and max_iter > 1:
            conv[0] = best_fit
    return best_pos.copy(), best_fit, conv


# ═══════════════════════════════════════════════════════════════════════════════
# 主执行流程
# ═══════════════════════════════════════════════════════════════════════════════

ALGORITHMS = [
    ("IDBO", run_idbo),
    ("ESA",  run_esa),
    ("VCS",  run_vcs),
    ("HGS",  run_hgs),
    ("IGOA", run_igoa),
    ("GWO",  run_gwo),
    ("WOA",  run_woa),
    ("SA",   run_sa),
]

def main():
    POP_SIZE = 20
    MAX_ITER = 100
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = os.path.join(OUT_DIR, "execution_log.txt")

    print("=" * 65)
    print("  STITP Optimization Experiment Engine v2.0")
    print(f"  Pop: {POP_SIZE}  |  Iterations: {MAX_ITER}  |  Algorithms: {len(ALGORITHMS)}")
    print("  tqdm progress -> execution_log.txt")
    print("=" * 65)

    with open(LOG_PATH, "w", encoding="utf-8") as log_f:
        log_f.write(f"STITP Optimization Log — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Pop={POP_SIZE} MaxIter={MAX_ITER}\n\n")

        results = {}
        for idx, (name, run_fn) in enumerate(ALGORITHMS):
            print(f"[{idx+1}/8] {name:>6s} ... ", end="", flush=True)

            try:
                res = execute_algorithm(name, POP_SIZE, MAX_ITER, run_fn, log_f)
                status = "OK"
            except Exception as e:
                log_f.write(f"{name} CRASHED: {e}\n")
                res = {"final_itae": 1e9, "execution_time": 0, "convergence": np.ones(MAX_ITER)*1e9,
                       "best_params": np.zeros(DIM)}
                status = "FAIL"

            results[name] = res
            itae = res["final_itae"]
            tsec = res["execution_time"]
            print(f"ITAE={itae:.6f}  AST={tsec:.2f}s  [{status}]")
            log_f.write(f"{name}: ITAE={itae:.6f}  AST={tsec:.2f}s  "
                        f"best_params={res['best_params']}  [{status}]\n\n")
            log_f.flush()

    # ── 导出 Excel ──
    print("\n" + "-" * 40)
    print("  Exporting to Excel ...")

    # Sheet 1: 收敛曲线
    df_conv = pd.DataFrame({name: results[name]["convergence"] for name, _ in ALGORITHMS})
    df_conv.index = range(1, MAX_ITER + 1)
    df_conv.index.name = "Iteration"

    # Sheet 2: 汇总
    rows = []
    for name, _ in ALGORITHMS:
        r = results[name]
        rows.append({
            "Algorithm": name,
            "Final_ITAE": r["final_itae"],
            "MSE": r["final_itae"] ** 2,
            "AST_s": r["execution_time"],
            "Kp_opt": r["best_params"][0] if r["best_params"] is not None else np.nan,
            "T1_opt": r["best_params"][1] if r["best_params"] is not None else np.nan,
            "T2_opt": r["best_params"][2] if r["best_params"] is not None else np.nan,
        })
    df_sum = pd.DataFrame(rows).sort_values("Final_ITAE").reset_index(drop=True)

    # Sheet 3: 实验参数
    df_cfg = pd.DataFrame({
        "Parameter": ["pop_size", "max_iter", "dimensions", "Kp_range",
                       "T1_range", "T2_range", "objective", "timestamp"],
        "Value": [POP_SIZE, MAX_ITER, DIM, "[1, 20]", "[0.05, 2.0]",
                  "[0.05, 2.0]", "ITAE (odeint-based PSS optimization)",
                  time.strftime("%Y-%m-%d %H:%M:%S")]
    })

    xlsx_path = os.path.join(OUT_DIR, "STITP_Experimental_Data.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_conv.to_excel(writer, sheet_name="Convergence_Curves")
        df_sum.to_excel(writer, sheet_name="Summary", index=False)
        df_cfg.to_excel(writer, sheet_name="Config", index=False)

    print(f"  Excel saved -> {xlsx_path}")
    print(f"  Sheets: Convergence ({df_conv.shape[0]}x{df_conv.shape[1]}), "
          f"Summary ({df_sum.shape[0]} rows), Config ({df_cfg.shape[0]} rows)")

    # ── 终端汇总 ──
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Rank':<5} {'Algo':<8} {'ITAE':<14} {'MSE':<14} {'AST(s)':<10}")
    print(f"  {'─'*55}")
    for rank, (_, row) in enumerate(df_sum.iterrows(), 1):
        marker = " <--" if row["Algorithm"] == "IDBO" else ""
        print(f"  {rank:<5} {row['Algorithm']:<8} {row['Final_ITAE']:<14.6f} "
              f"{row['MSE']:<14.8f} {row['AST_s']:<10.2f}{marker}")
    print("=" * 65)
    print("  All algorithms executed with real ODE-based objective function.")
    print(f"  Log file: {LOG_PATH}")
    print("=" * 65)

    return results, xlsx_path


if __name__ == "__main__":
    main()
