from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import norm, beta
import numpy as np
from scipy.stats import gaussian_kde

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://abtest-calculator.vercel.app"
    ],  # Разрешаем запросы с фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ABTestInput(BaseModel):
    a_success: int
    a_total: int
    b_success: int
    b_total: int
    a_prior_alpha: int = 1
    a_prior_beta: int = 1
    b_prior_alpha: int = 1
    b_prior_beta: int = 1

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

@app.post("/calculate", response_model=ABTestResult)
def calculate_abtest(data: ABTestInput):
    # Частотный подход (z-тест для пропорций)
    p1 = data.a_success / data.a_total
    p2 = data.b_success / data.b_total
    p_pool = (data.a_success + data.b_success) / (data.a_total + data.b_total)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/data.a_total + 1/data.b_total))
    z = (p2 - p1) / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    significant = p_value < 0.05

    # Байесовский подход (Beta posterior) с учетом априорных значений
    a_alpha = data.a_success + data.a_prior_alpha
    a_beta = data.a_total - data.a_success + data.a_prior_beta
    b_alpha = data.b_success + data.b_prior_alpha
    b_beta = data.b_total - data.b_success + data.b_prior_beta
    
    # Генерируем точки для построения графика
    x = np.linspace(0, 1, 100)
    # Сэмплируем из beta-распределения
    a_samples = np.random.beta(a_alpha, a_beta, 100_000)
    b_samples = np.random.beta(b_alpha, b_beta, 100_000)
    # KDE по сэмплам для гладких графиков
    a_kde = gaussian_kde(a_samples)
    b_kde = gaussian_kde(b_samples)
    a_dist = a_kde(x)
    b_dist = b_kde(x)
    a_dist = a_dist / np.max(a_dist)
    b_dist = b_dist / np.max(b_dist)

    # Байесовские метрики
    prob_b_better = float(np.mean(b_samples > a_samples))
    prob_a_better = float(np.mean(a_samples > b_samples))
    # Средние значения (Conversion Rate)
    a_mean = float(np.mean(a_samples))
    b_mean = float(np.mean(b_samples))
    # Expected Loss (DY-style)
    a_expected_loss = float(np.mean(np.maximum(b_samples - a_samples, 0)))
    b_expected_loss = float(np.mean(np.maximum(a_samples - b_samples, 0)))
    # Распределение разности для графика
    diff_samples = b_samples - a_samples
    diff_x = np.linspace(np.min(diff_samples), np.max(diff_samples), 100)
    kde = gaussian_kde(diff_samples)
    diff_distribution = kde(diff_x)
    diff_distribution = diff_distribution / np.max(diff_distribution)

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
        diff_distribution=diff_distribution.tolist()
    )