from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import norm, beta
import numpy as np
from scipy.stats import gaussian_kde
import pymc as pm
from fastapi import HTTPException
import time

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://abtest-calculator.vercel.app",
        "https://abtest-calculator.onrender.com"
    ],  # Разрешаем запросы с фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AB Test Calculator API"}

class ABTestInput(BaseModel):
    a_success: int
    a_total: int
    b_success: int
    b_total: int

class ABTestResult(BaseModel):
    freq_p_value: float
    freq_significant: bool
    bayes_prob_b_better: float
    a_distribution: list[float]
    b_distribution: list[float]
    x_values: list[float]
    a_mean: float
    b_mean: float
    a_prob_best: float
    b_prob_best: float
    a_expected_loss: float
    b_expected_loss: float
    diff_x: list[float]
    diff_distribution: list[float]
    a_hist: list[float]
    b_hist: list[float]
    x_hist: list[float]

@app.post("/calculate", response_model=ABTestResult)
def calculate_abtest(data: ABTestInput):
    func_start_time = time.time()
    print(f"[LOG] calculate_abtest: Start")

    # Проверка входных данных
    if data.a_total <= 0 or data.b_total <= 0:
        raise HTTPException(status_code=400, detail="Общее количество должно быть положительным числом")
    if data.a_success < 0 or data.b_success < 0:
        raise HTTPException(status_code=400, detail="Количество успехов не может быть отрицательным")
    if data.a_success > data.a_total or data.b_success > data.b_total:
        raise HTTPException(status_code=400, detail="Количество успехов не может быть больше общего количества")

    # --- Frequentist Part ---
    freq_calc_start_time = time.time()
    p1 = data.a_success / data.a_total
    p2 = data.b_success / data.b_total
    p_pool = (data.a_success + data.b_success) / (data.a_total + data.b_total)
    se_term = p_pool * (1 - p_pool) * (1/data.a_total + 1/data.b_total)
    se = np.sqrt(max(0, se_term)) if se_term > 0 else 0
    z = (p2 - p1) / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    significant = p_value < 0.05
    freq_calc_end_time = time.time()
    print(f"[LOG] Frequentist calculations: {freq_calc_end_time - freq_calc_start_time:.4f}s")

    # --- Bayesian Part (ADVI) ---
    bayesian_start_time = time.time()
    print(f"[LOG] Bayesian part: Start")

    model_block_start_time = time.time()
    with pm.Model() as model:
        p_a = pm.Beta('p_a', alpha=1, beta=1)
        p_b = pm.Beta('p_b', alpha=1, beta=1)
        obs_a = pm.Binomial('obs_a', n=data.a_total, p=p_a, observed=data.a_success)
        obs_b = pm.Binomial('obs_b', n=data.b_total, p=p_b, observed=data.b_success)
        delta = pm.Deterministic('delta', p_b - p_a)

        fit_start_time = time.time()
        approx = pm.fit(method='advi', n=30000, random_seed=42, progressbar=False) # Set progressbar=False for cleaner logs
        fit_end_time = time.time()
        print(f"[LOG] pm.fit(advi, n=30000): {fit_end_time - fit_start_time:.4f}s")

        sample_start_time = time.time()
        trace = approx.sample(draws=1000, random_seed=42)
        sample_end_time = time.time()
        print(f"[LOG] approx.sample(draws=1000): {sample_end_time - sample_start_time:.4f}s")
    model_block_end_time = time.time()
    print(f"[LOG] pm.Model block (fit+sample): {model_block_end_time - model_block_start_time:.4f}s")

    # --- Data Processing Part ---
    processing_start_time = time.time()
    print(f"[LOG] Data processing: Start")

    extract_samples_start_time = time.time()
    a_samples = trace.posterior['p_a'].values.flatten()
    b_samples = trace.posterior['p_b'].values.flatten()
    extract_samples_end_time = time.time()
    print(f"[LOG] Sample extraction: {extract_samples_end_time - extract_samples_start_time:.4f}s")

    kde_calc_start_time = time.time()
    x = np.linspace(0, 1, 300)
    a_kde = gaussian_kde(a_samples)
    b_kde = gaussian_kde(b_samples)
    a_dist = a_kde(x)
    b_dist = b_kde(x)
    if np.max(a_dist) > 0: a_dist = a_dist / np.max(a_dist)
    if np.max(b_dist) > 0: b_dist = b_dist / np.max(b_dist)
    kde_calc_end_time = time.time()
    print(f"[LOG] KDE calculations (a_dist, b_dist): {kde_calc_end_time - kde_calc_start_time:.4f}s")

    bayes_metrics_start_time = time.time()
    prob_b_better = float(np.mean(b_samples > a_samples))
    prob_a_better = float(np.mean(a_samples > b_samples))
    a_mean = float(np.mean(a_samples))
    b_mean = float(np.mean(b_samples))
    # added .any() for safety, though with Beta priors giving samples in (0,1) and non-zero success/total, samples should not be all zero.
    a_expected_loss = float(np.mean(np.where(b_samples > a_samples, (b_samples - a_samples) / b_samples if b_samples.any() else 0, 0)))
    b_expected_loss = float(np.mean(np.where(a_samples > b_samples, (a_samples - b_samples) / a_samples if a_samples.any() else 0, 0)))
    bayes_metrics_end_time = time.time()
    print(f"[LOG] Bayesian metrics (prob_b_better, means, expected_loss): {bayes_metrics_end_time - bayes_metrics_start_time:.4f}s")

    diff_dist_start_time = time.time()
    diff_samples = b_samples - a_samples
    diff_x = np.linspace(np.min(diff_samples), np.max(diff_samples), 100)
    kde_diff = gaussian_kde(diff_samples)
    diff_distribution = kde_diff(diff_x)
    if np.max(diff_distribution) > 0: diff_distribution = diff_distribution / np.max(diff_distribution) # Normalize
    diff_dist_end_time = time.time()
    print(f"[LOG] Difference distribution calculation: {diff_dist_end_time - diff_dist_start_time:.4f}s")

    hist_calc_start_time = time.time()
    bins = 300
    a_hist, bin_edges = np.histogram(a_samples, bins=bins, range=(0, 1), density=True)
    b_hist, _ = np.histogram(b_samples, bins=bins, range=(0, 1), density=True)
    x_hist = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_calc_end_time = time.time()
    print(f"[LOG] Histogram calculations: {hist_calc_end_time - hist_calc_start_time:.4f}s")

    processing_end_time = time.time()
    print(f"[LOG] Data processing: Total {processing_end_time - processing_start_time:.4f}s")

    # --- Return statement ---
    return_prep_start_time = time.time()
    result = ABTestResult(
        freq_p_value=round(p_value, 6),
        freq_significant=significant,
        bayes_prob_b_better=round(prob_b_better, 4),
        a_distribution=a_dist.tolist(),
        b_distribution=b_dist.tolist(),
        x_values=x.tolist(),
        a_mean=round(a_mean, 6),
        b_mean=round(b_mean, 6),
        a_prob_best=round(prob_a_better, 4),
        b_prob_best=round(prob_b_better, 4),
        a_expected_loss=round(a_expected_loss, 6),
        b_expected_loss=round(b_expected_loss, 6),
        diff_x=diff_x.tolist(),
        diff_distribution=diff_distribution.tolist(),
        a_hist=a_hist.tolist(),
        b_hist=b_hist.tolist(),
        x_hist=x_hist.tolist()
    )
    return_prep_end_time = time.time()
    print(f"[LOG] Result preparation: {return_prep_end_time - return_prep_start_time:.4f}s")

    func_end_time = time.time()
    print(f"[LOG] calculate_abtest: Total execution {func_end_time - func_start_time:.4f}s")
    return result