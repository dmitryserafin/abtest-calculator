from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import norm, beta
import numpy as np
from scipy.stats import gaussian_kde
# import pymc as pm # Removed
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

    # --- Bayesian Part (Analytic Beta Distribution) ---
    bayesian_start_time = time.time() # Используем существующий логгер или добавляем новый
    print(f"[LOG] Bayesian part (Analytic Beta): Start")

    # Априорные параметры (неинформативный prior Beta(1,1))
    prior_alpha = 1.0
    prior_beta = 1.0

    # Вычисление параметров заднего Beta-распределения для варианта A
    alpha_a_posterior = prior_alpha + data.a_success
    beta_a_posterior = prior_beta + data.a_total - data.a_success

    # Вычисление параметров заднего Beta-распределения для варианта B
    alpha_b_posterior = prior_alpha + data.b_success
    beta_b_posterior = prior_beta + data.b_total - data.b_success

    # Логирование вычисленных параметров (опционально, но полезно для отладки)
    print(f"[LOG] Posterior A: Beta({alpha_a_posterior}, {beta_a_posterior})")
    print(f"[LOG] Posterior B: Beta({alpha_b_posterior}, {beta_b_posterior})")

    # Заметка: переменные a_samples и b_samples будут созданы на следующем шаге плана.
    # На этом шаге мы только вычисляем alpha_a_posterior, beta_a_posterior, alpha_b_posterior, beta_b_posterior.

    sampling_start_time = time.time()
    N_SAMPLES = 20000 # Количество сэмплов для генерации
    a_samples = beta.rvs(alpha_a_posterior, beta_a_posterior, size=N_SAMPLES, random_state=42)
    b_samples = beta.rvs(alpha_b_posterior, beta_b_posterior, size=N_SAMPLES, random_state=42)
    sampling_end_time = time.time()
    print(f"[LOG] Sampling from posterior Beta distributions (N={N_SAMPLES}): {sampling_end_time - sampling_start_time:.4f}s")

    bayesian_end_time = time.time() # Завершение всего байесовского блока
    print(f"[LOG] Bayesian part (Analytic Beta): Total {bayesian_end_time - bayesian_start_time:.4f}s")

    # --- Bayesian Part (ADVI) ---
    # [LOGS AND PYMC CODE REMOVED AS PER TASK]
    # Variables like a_samples, b_samples, etc. will be redefined or removed in subsequent steps.
    # For now, the data processing part that uses them will likely error out if not handled.
    # The task is to remove the PyMC block first.

    # --- Data Processing Part ---
    # This part will be modified later to use new sample generation
    processing_start_time = time.time()
    print(f"[LOG] Data processing: Start")

    # extract_samples_start_time = time.time() # Removed
    # a_samples = trace.posterior['p_a'].values.flatten() # Removed
    # b_samples = trace.posterior['p_b'].values.flatten() # Removed
    # extract_samples_end_time = time.time() # Removed
    # print(f"[LOG] Sample extraction: {extract_samples_end_time - extract_samples_start_time:.4f}s") # Removed

    # The following lines will need new a_samples, b_samples or be removed/modified
    # For now, keeping them commented out or aware they will cause errors
    # x = np.linspace(0, 1, 300) # Placeholder, will be defined with new samples
    # a_samples = np.array([]) # Placeholder # REMOVED by new definition above
    # b_samples = np.array([]) # Placeholder # REMOVED by new definition above
    x = np.linspace(0, 1, 300) # x will be defined here based on typical [0,1] range for beta distributions
    # The following lines will need new a_samples, b_samples or be removed/modified
    # For now, providing dummy values or commenting out to prevent immediate errors
    # but expecting these to be handled by subsequent sample generation logic.
    a_dist = []
    b_dist = []
    kde_calc_start_time = time.time()
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
    # Ensure diff_samples is not empty before attempting np.min/np.max
    if diff_samples.size > 0:
        diff_x_min = np.min(diff_samples)
        diff_x_max = np.max(diff_samples)
        if diff_x_min == diff_x_max: # Avoids issue with linspace if min=max
            diff_x_min -= 0.01
            diff_x_max += 0.01
        diff_x = np.linspace(diff_x_min, diff_x_max, 100)
        kde_diff = gaussian_kde(diff_samples)
        diff_distribution = kde_diff(diff_x)
        if np.max(diff_distribution) > 0: diff_distribution = diff_distribution / np.max(diff_distribution) # Normalize
    else: # Handle empty diff_samples case
        diff_x = []
        diff_distribution = []
    diff_dist_end_time = time.time()
    print(f"[LOG] Difference distribution calculation: {diff_dist_end_time - diff_dist_start_time:.4f}s")

    hist_calc_start_time = time.time()
    bins = 300 # Kept for now, might be used by new sample generation
    # Ensure a_samples and b_samples are not empty before trying to create histograms
    if a_samples.size > 0:
        a_hist, bin_edges = np.histogram(a_samples, bins=bins, range=(0, 1), density=True)
        x_hist = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        a_hist = [] # Provide empty lists if no samples
        x_hist = [] # Ensure x_hist is also empty if a_samples is empty
        bin_edges = np.linspace(0,1,bins+1) # default bin_edges for x_hist if a_samples is empty

    if b_samples.size > 0:
        b_hist, _ = np.histogram(b_samples, bins=bins, range=(0, 1), density=True)
        if not x_hist.size > 0: # if a_samples was empty, x_hist would be empty
             _, bin_edges_b = np.histogram(b_samples, bins=bins, range=(0, 1), density=True) # need bin_edges for x_hist
             x_hist = 0.5 * (bin_edges_b[:-1] + bin_edges_b[1:])
    else:
        b_hist = []

    hist_calc_end_time = time.time()
    print(f"[LOG] Histogram calculations: {hist_calc_end_time - hist_calc_start_time:.4f}s")

    processing_end_time = time.time() # This specific log can remain if data processing part is still meaningful
    print(f"[LOG] Data processing: Total {processing_end_time - processing_start_time:.4f}s") # This specific log can remain

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