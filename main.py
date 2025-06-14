from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import norm, beta
import numpy as np
from scipy.stats import gaussian_kde
import pymc as pm
from fastapi import HTTPException

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
    # Проверка входных данных
    if data.a_total <= 0 or data.b_total <= 0:
        raise HTTPException(status_code=400, detail="Общее количество должно быть положительным числом")
    if data.a_success < 0 or data.b_success < 0:
        raise HTTPException(status_code=400, detail="Количество успехов не может быть отрицательным")
    if data.a_success > data.a_total or data.b_success > data.b_total:
        raise HTTPException(status_code=400, detail="Количество успехов не может быть больше общего количества")

    # Частотный подход (z-тест для пропорций)
    p1 = data.a_success / data.a_total
    p2 = data.b_success / data.b_total
    p_pool = (data.a_success + data.b_success) / (data.a_total + data.b_total)
    print(f"Debug values: p1={p1}, p2={p2}, p_pool={p_pool}")
    print(f"Input data: a_success={data.a_success}, a_total={data.a_total}, b_success={data.b_success}, b_total={data.b_total}")
    
    # Защита от отрицательных значений под корнем
    se_term = p_pool * (1 - p_pool) * (1/data.a_total + 1/data.b_total)
    print(f"SE term before sqrt: {se_term}")
    se = np.sqrt(max(0, se_term)) if se_term > 0 else 0
    print(f"Final SE: {se}")
    
    z = (p2 - p1) / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    significant = p_value < 0.05

    # Байесовский подход через PyMC с неинформативным prior (alpha=1, beta=1)
    with pm.Model() as model:
        p_a = pm.Beta('p_a', alpha=1, beta=1)
        p_b = pm.Beta('p_b', alpha=1, beta=1)
        obs_a = pm.Binomial('obs_a', n=data.a_total, p=p_a, observed=data.a_success)
        obs_b = pm.Binomial('obs_b', n=data.b_total, p=p_b, observed=data.b_success)
        delta = pm.Deterministic('delta', p_b - p_a)
        trace = pm.sample(2000, tune=1000, cores=1, random_seed=42, progressbar=True, return_inferencedata=False)

    a_samples = trace['p_a']
    b_samples = trace['p_b']
    # Для графика: KDE по сэмплам
    x = np.linspace(0, 1, 300)
    a_kde = gaussian_kde(a_samples)
    b_kde = gaussian_kde(b_samples)
    a_dist = a_kde(x)
    b_dist = b_kde(x)
    a_dist = a_dist / np.max(a_dist)
    b_dist = b_dist / np.max(b_dist)

    # Байесовские метрики
    prob_b_better = float(np.mean(b_samples > a_samples))
    prob_a_better = float(np.mean(a_samples > b_samples))
    a_mean = float(np.mean(a_samples))
    b_mean = float(np.mean(b_samples))
    a_expected_loss = float(np.mean(np.where(b_samples > a_samples, (b_samples - a_samples) / b_samples, 0)))
    b_expected_loss = float(np.mean(np.where(a_samples > b_samples, (a_samples - b_samples) / a_samples, 0)))
    # Распределение разности для графика
    diff_samples = b_samples - a_samples
    diff_x = np.linspace(np.min(diff_samples), np.max(diff_samples), 100)
    kde = gaussian_kde(diff_samples)
    diff_distribution = kde(diff_x)
    diff_distribution = diff_distribution / np.max(diff_distribution)

    # Для гистограммы: считаем частоты по бинам
    bins = 300
    a_hist, bin_edges = np.histogram(a_samples, bins=bins, range=(0, 1), density=True)
    b_hist, _ = np.histogram(b_samples, bins=bins, range=(0, 1), density=True)
    x_hist = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # центры бинов

    return ABTestResult(
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
        # Новые поля для гистограммы
        a_hist=a_hist.tolist(),
        b_hist=b_hist.tolist(),
        x_hist=x_hist.tolist()
    )