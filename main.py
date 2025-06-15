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

    N_SAMPLES_METRICS = 5000000
    N_SAMPLES_PLOTS = 16666

    sampling_metrics_start_time = time.time()
    a_samples_full = beta.rvs(alpha_a_posterior, beta_a_posterior, size=N_SAMPLES_METRICS, random_state=42)
    b_samples_full = beta.rvs(alpha_b_posterior, beta_b_posterior, size=N_SAMPLES_METRICS, random_state=42)
    sampling_metrics_end_time = time.time()
    print(f"[LOG] Sampling for metrics (N={N_SAMPLES_METRICS}): {sampling_metrics_end_time - sampling_metrics_start_time:.4f}s")

    # Создание подвыборок для графиков
    slicing_plot_samples_start_time = time.time()
    a_samples_plot = a_samples_full[:N_SAMPLES_PLOTS]
    b_samples_plot = b_samples_full[:N_SAMPLES_PLOTS]
    slicing_plot_samples_end_time = time.time()
    print(f"[LOG] Slicing samples for plots (N={N_SAMPLES_PLOTS}): {slicing_plot_samples_end_time - slicing_plot_samples_start_time:.4f}s")

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

    # extract_samples_start_time = time.time() # This block is removed as samples are now a_samples_full/plot, b_samples_full/plot
    # print(f"[LOG] Sample extraction: ...") # This log is removed

    # The following lines will need new a_samples, b_samples or be removed/modified
    # For now, keeping them commented out or aware they will cause errors
    # x = np.linspace(0, 1, 300) # Placeholder, will be defined with new samples
    # a_samples = np.array([]) # Placeholder # REMOVED by new definition above
    # b_samples = np.array([]) # Placeholder # REMOVED by new definition above
    x = np.linspace(0, 1, 300) # x will be defined here based on typical [0,1] range for beta distributions
    # The following lines will need new a_samples, b_samples or be removed/modified
    # For now, providing dummy values or commenting out to prevent immediate errors
    # but expecting these to be handled by subsequent sample generation logic.
    # a_dist = [] # These will be calculated below using _plot samples
    # b_dist = [] # These will be calculated below using _plot samples

    bayes_metrics_start_time = time.time()
    # Используем _full сэмплы для метрик
    prob_b_better = float(np.mean(b_samples_full > a_samples_full))
    prob_a_better = float(np.mean(a_samples_full > b_samples_full))
    a_mean = float(np.mean(a_samples_full))
    b_mean = float(np.mean(b_samples_full))
    # Для expected_loss также используем _full сэмплы
    a_expected_loss = float(np.mean(np.where(b_samples_full > a_samples_full, (b_samples_full - a_samples_full) / b_samples_full if b_samples_full.any() else 0, 0)))
    b_expected_loss = float(np.mean(np.where(a_samples_full > b_samples_full, (a_samples_full - b_samples_full) / a_samples_full if a_samples_full.any() else 0, 0)))
    bayes_metrics_end_time = time.time()
    print(f"[LOG] Bayesian metrics (from N={N_SAMPLES_METRICS} samples): {bayes_metrics_end_time - bayes_metrics_start_time:.4f}s")

    kde_calc_start_time = time.time()
    # Используем _plot сэмплы для KDE
    if a_samples_plot.size > 1: a_kde = gaussian_kde(a_samples_plot)
    else: a_kde = None # или какая-то заглушка, если 0 или 1 сэмпл
    if b_samples_plot.size > 1: b_kde = gaussian_kde(b_samples_plot)
    else: b_kde = None

    a_dist = a_kde(x) if a_kde else np.zeros_like(x)
    b_dist = b_kde(x) if b_kde else np.zeros_like(x)

    if np.max(a_dist, initial=0) > 0: a_dist = a_dist / np.max(a_dist) # Added initial=0 for safety with empty arrays
    if np.max(b_dist, initial=0) > 0: b_dist = b_dist / np.max(b_dist) # Added initial=0 for safety with empty arrays
    kde_calc_end_time = time.time()
    print(f"[LOG] KDE calculations (from N={N_SAMPLES_PLOTS} samples): {kde_calc_end_time - kde_calc_start_time:.4f}s")

    diff_dist_start_time = time.time()
    # Создаем diff_samples_plot из _plot сэмплов
    diff_samples_plot = b_samples_plot - a_samples_plot
    # Используем diff_samples_plot для KDE
    if diff_samples_plot.size > 1 :
        diff_x_min = np.min(diff_samples_plot)
        diff_x_max = np.max(diff_samples_plot)
        # Обеспечить, что diff_x_min < diff_x_max для linspace и KDE
        if diff_x_min >= diff_x_max:
             diff_x_min = diff_x_min - 0.01 # Пример простого отступа
             diff_x_max = diff_x_max + 0.01
        if diff_x_min == diff_x_max: # если все значения одинаковы после отступа (маловероятно, но для защиты)
             diff_x = np.array([diff_x_min])
             diff_distribution = np.array([1.0])
        else:
             diff_x = np.linspace(diff_x_min, diff_x_max, 100)
             kde_diff = gaussian_kde(diff_samples_plot)
             diff_distribution = kde_diff(diff_x)
    else: # если мало данных для KDE
        diff_x = np.array([-0.1, 0, 0.1]) # какая-то заглушка
        diff_distribution = np.array([0.0,1.0,0.0]) # какая-то заглушка

    if np.max(diff_distribution, initial=0) > 0: diff_distribution = diff_distribution / np.max(diff_distribution) # Added initial=0 for safety
    diff_dist_end_time = time.time()
    print(f"[LOG] Difference distribution calculation (from N={N_SAMPLES_PLOTS} samples): {diff_dist_end_time - diff_dist_start_time:.4f}s")

    hist_calc_start_time = time.time()
    bins = 300 # Kept for now, might be used by new sample generation
    # Используем _plot сэмплы для гистограмм
    if a_samples_plot.size > 0:
        a_hist, bin_edges = np.histogram(a_samples_plot, bins=bins, range=(0, 1), density=True)
        x_hist = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        a_hist = np.zeros(bins).tolist() # заглушка
        bin_edges = np.linspace(0,1,bins+1) # default bin_edges
        x_hist = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # x_hist based on default bins

    if b_samples_plot.size > 0:
        b_hist, _ = np.histogram(b_samples_plot, bins=bins, range=(0, 1), density=True)
    else:
        b_hist = np.zeros(bins).tolist() # заглушка

    # Ensure x_hist is tolist() if it's a numpy array from calculation
    if isinstance(x_hist, np.ndarray):
        x_hist = x_hist.tolist()
    if isinstance(a_hist, np.ndarray):
        a_hist = a_hist.tolist()
    if isinstance(b_hist, np.ndarray):
        b_hist = b_hist.tolist()

    hist_calc_end_time = time.time()
    print(f"[LOG] Histogram calculations (from N={N_SAMPLES_PLOTS} samples): {hist_calc_end_time - hist_calc_start_time:.4f}s")

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
        a_hist=a_hist, # Already a list
        b_hist=b_hist, # Already a list
        x_hist=x_hist  # Already a list
    )
    return_prep_end_time = time.time()
    print(f"[LOG] Result preparation: {return_prep_end_time - return_prep_start_time:.4f}s")

    func_end_time = time.time()
    print(f"[LOG] calculate_abtest: Total execution {func_end_time - func_start_time:.4f}s")
    return result