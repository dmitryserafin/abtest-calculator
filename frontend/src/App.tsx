import { useState } from 'react'
import { 
  Container, 
  TextField, 
  Button, 
  Typography, 
  Box, 
  Paper,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress
} from '@mui/material'
import axios from 'axios'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement
)

interface ABTestResult {
  freq_p_value: number
  freq_significant: boolean
  bayes_prob_b_better: number
  a_distribution: number[]
  b_distribution: number[]
  x_values: number[]
  a_mean: number
  a_prob_best: number
  a_expected_loss: number
  b_mean: number
  b_prob_best: number
  b_expected_loss: number
  diff_x: number[]
  diff_distribution: number[]
  x_hist?: number[]
  a_hist?: number[]
  b_hist?: number[]
}

function App() {
  const [controlSuccess, setControlSuccess] = useState('')
  const [controlTotal, setControlTotal] = useState('')
  const [variantSuccess, setVariantSuccess] = useState('')
  const [variantTotal, setVariantTotal] = useState('')
  const [result, setResult] = useState<ABTestResult | null>(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  // --- Калькулятор размера выборки ---
  const [sampleBaseline, setSampleBaseline] = useState(20)
  const [sampleLift, setSampleLift] = useState(3)
  const [sampleAlpha, setSampleAlpha] = useState(95)
  const [samplePower, setSamplePower] = useState(80)
  const [sampleResult, setSampleResult] = useState<number|null>(null)

  const handleCalculate = async () => {
    try {
      setError('')
      setIsLoading(true)
      const backendUrl = import.meta.env.VITE_BACKEND_URL || 'https://abtest-calculator.onrender.com'
      const response = await axios.post(`${backendUrl}/calculate`, {
        a_success: parseInt(controlSuccess),
        a_total: parseInt(controlTotal),
        b_success: parseInt(variantSuccess),
        b_total: parseInt(variantTotal)
      })
      setResult(response.data)
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.data?.detail) {
        setError(err.response.data.detail)
      } else {
        setError('Ошибка при расчете. Проверьте введенные данные.')
      }
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  // Формула для двух пропорций (частотный подход)
  function calcSampleSize() {
    const alpha = 1 - sampleAlpha / 100
    const power = samplePower / 100
    const p1 = sampleBaseline / 100
    const p2 = (sampleBaseline + sampleLift) / 100
    const z_alpha = normSInv(1 - alpha / 2)
    const z_beta = normSInv(power)
    const pooled = (p1 + p2) / 2
    const q1 = 1 - p1
    const q2 = 1 - p2
    const n = ((z_alpha * Math.sqrt(2 * pooled * (1 - pooled)) + z_beta * Math.sqrt(p1 * q1 + p2 * q2)) ** 2) / ((p2 - p1) ** 2)
    setSampleResult(Math.ceil(n))
  }

  // Обратная функция стандартного нормального распределения (approx)
  function normSInv(p: number) {
    // Abramowitz and Stegun formula 26.2.23
    if (p <= 0 || p >= 1) throw new Error('p must be in (0,1)')
    const a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969
    const a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924
    const b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887
    const b4 = 66.8013118877197, b5 = -13.2806815528857
    const c1 = -0.00778489400243029, c2 = -0.322396458041136
    const c3 = -2.40075827716184, c4 = -2.54973253934373
    const c5 = 4.37466414146497, c6 = 2.93816398269878
    const d1 = 0.00778469570904146, d2 = 0.32246712907004
    const d3 = 2.445134137143, d4 = 3.75440866190742
    const p_low = 0.02425
    const p_high = 1 - p_low
    let q, r;
    let ret = 0;
    if (p < p_low) {
      q = Math.sqrt(-2 * Math.log(p))
      ret = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    } else if (p <= p_high) {
      q = p - 0.5
      r = q * q
      ret = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p))
      ret = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    }
    return ret
  }

  // График: Распределение коэффициентов конверсии с учетом размера выборки (KDE)
  const conversionChartData = result ? {
    datasets: [
      {
        label: 'Контрольная группа',
        data: result.x_values.map((val, i) => ({ x: val * 100, y: result.a_distribution[i] })),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.3)',
        tension: 0.4,
        fill: false,
        pointRadius: 0,
        order: 1
      },
      {
        label: 'Тестовая группа',
        data: result.x_values.map((val, i) => ({ x: val * 100, y: result.b_distribution[i] })),
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.3)',
        tension: 0.4,
        fill: false,
        pointRadius: 0,
        order: 1
      }
    ]
  } : null;

  // Показываем оба "колокола" полностью (от роста до спада)
  let xMin = 0, xMax = 100;
  if (result) {
    const threshold = 0.01; // 1% от максимума любого распределения
    const allY = result.a_distribution.concat(result.b_distribution);
    const maxY = Math.max(...allY);
    const indices = result.x_values
      .map((_, i) => (result.a_distribution[i] > threshold * maxY || result.b_distribution[i] > threshold * maxY) ? i : -1)
      .filter(i => i !== -1);
    if (indices.length > 0) {
      // Добавим запас по краям
      xMin = Math.max(0, result.x_values[indices[0]] * 100 - 1);
      xMax = Math.min(100, result.x_values[indices[indices.length - 1]] * 100 + 1);
    }
  }

  const conversionChartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' as const },
      title: {
        display: true,
        text: 'Распределение коэффициентов конверсии с учетом размера выборки',
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false
      }
    },
    scales: {
      x: {
        type: 'linear' as const,
        title: { display: true, text: 'Конверсия, %' },
        min: xMin,
        max: xMax,
        ticks: {
          callback: function(tickValue: number | string) {
            return `${Number(tickValue).toFixed(2)}%`;
          }
        }
      },
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Плотность' }
      }
    }
  }

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Калькулятор A/B тестов
        </Typography>
        
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              <b>Априорные значения (α и β):</b> задают вашу изначальную уверенность в конверсии до эксперимента. Обычно используют значения 1 и 1 (нейтральный неинформативный приоритет). Если у вас есть исторические данные или экспертные ожидания, вы можете их отразить, увеличив α и β. Например, α=10, β=90 означает, что до эксперимента вы ожидали 10 успехов и 90 неудач.
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" gutterBottom>
                Контрольная группа
              </Typography>
              <TextField
                fullWidth
                label="Общее количество"
                type="number"
                value={controlTotal}
                onChange={(e) => setControlTotal(e.target.value)}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Количество успехов"
                type="number"
                value={controlSuccess}
                onChange={(e) => setControlSuccess(e.target.value)}
                margin="normal"
              />
            </Box>
            
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" gutterBottom>
                Тестовая группа
              </Typography>
              <TextField
                fullWidth
                label="Общее количество"
                type="number"
                value={variantTotal}
                onChange={(e) => setVariantTotal(e.target.value)}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Количество успехов"
                type="number"
                value={variantSuccess}
                onChange={(e) => setVariantSuccess(e.target.value)}
                margin="normal"
              />
            </Box>
          </Box>

          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button 
              variant="contained" 
              onClick={handleCalculate}
              disabled={!controlSuccess || !controlTotal || !variantSuccess || !variantTotal || isLoading}
            >
              {isLoading ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={20} color="inherit" />
                  <span>Расчет...</span>
                </Box>
              ) : (
                'Рассчитать'
              )}
            </Button>
          </Box>
        </Paper>

        {error && (
          <Typography color="error" align="center" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {result && !isLoading && (
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Результаты анализа
            </Typography>

            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                <b>Как интерпретировать результаты?</b>
              </Typography>
              <Typography variant="body2" gutterBottom>
                <b>Частотный подход:</b> P-value — вероятность получить такую или более экстремальную разницу между группами, если на самом деле разницы нет. Если p-value &lt; 0.05 — разница статистически значима, можно отвергнуть гипотезу о равенстве. Если p-value ≥ 0.05 — статистически значимой разницы не обнаружено.
              </Typography>
              <Typography variant="body2" gutterBottom>
                <b>Байесовский подход:</b> Probability to be Best — вероятность, что вариант действительно лучше. Expected Loss — средний процент, который вы теряете, если выберете не лучший вариант. Если вероятность быть лучшим &gt; 95% — очень сильные доказательства в пользу варианта. Если вероятность около 50% — недостаточно данных для уверенного выбора.
              </Typography>
            </Box>

            <Box sx={{ mb: 4 }}>
              <Typography variant="h6" gutterBottom>
                Частотный подход
              </Typography>
              <Typography>
                P-value: {result.freq_p_value}
              </Typography>
              <Typography>
                Статистическая значимость: {result.freq_significant ? 'Да' : 'Нет'}
              </Typography>
              <Typography>
                {result.freq_significant 
                  ? 'Есть статистически значимая разница между группами (p < 0.05)'
                  : 'Нет статистически значимой разницы между группами (p ≥ 0.05)'}
              </Typography>
            </Box>

            <Divider sx={{ my: 3 }} />

            <Box sx={{ mb: 4 }}>
              <Typography variant="h6" gutterBottom>
                Байесовский подход
              </Typography>
              <Typography>
                Вероятность того, что тестовая группа лучше: {(result.bayes_prob_b_better * 100).toFixed(1)}%
              </Typography>
              <Typography>
                {result.bayes_prob_b_better > 0.95 || result.bayes_prob_b_better < 0.05 
                  ? 'Очень сильные доказательства'
                  : result.bayes_prob_b_better > 0.9 || result.bayes_prob_b_better < 0.1
                  ? 'Сильные доказательства'
                  : result.bayes_prob_b_better > 0.8 || result.bayes_prob_b_better < 0.2
                  ? 'Умеренные доказательства'
                  : result.bayes_prob_b_better > 0.6 || result.bayes_prob_b_better < 0.4
                  ? 'Слабые доказательства'
                  : 'Недостаточно данных для принятия решения'}
              </Typography>
            </Box>

            {/* График распределения коэффициентов конверсии */}
            {conversionChartData && (
              <Box sx={{ height: { xs: 300, sm: 400 }, mt: 2 }}>
                <Line options={conversionChartOptions} data={conversionChartData} />
              </Box>
            )}

            {/* Таблица с метриками */}
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Байесовские метрики по группам
              </Typography>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Группа</TableCell>
                      <TableCell align="right">Conversion Rate</TableCell>
                      <TableCell align="right">Probability to be Best</TableCell>
                      <TableCell align="right">Expected Loss</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Контрольная</TableCell>
                      <TableCell align="right">{(result.a_mean * 100).toFixed(2)}%</TableCell>
                      <TableCell align="right">{(result.a_prob_best * 100).toFixed(2)}%</TableCell>
                      <TableCell align="right">{(result.a_expected_loss * 100).toFixed(4)}%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Тестовая</TableCell>
                      <TableCell align="right">{(result.b_mean * 100).toFixed(2)}%</TableCell>
                      <TableCell align="right">{(result.b_prob_best * 100).toFixed(2)}%</TableCell>
                      <TableCell align="right">{(result.b_expected_loss * 100).toFixed(4)}%</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          </Paper>
        )}

        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom align="center">
            Калькулятор размера выборки (частотный A/B тест)
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2, mb: 2 }}>
            <TextField
              label="Средний показатель, %"
              type="number"
              value={sampleBaseline}
              onChange={e => setSampleBaseline(Number(e.target.value))}
              fullWidth
              sx={{ flex: 1 }}
            />
            <TextField
              label="Ожидаемый прирост, %"
              type="number"
              value={sampleLift}
              onChange={e => setSampleLift(Number(e.target.value))}
              fullWidth
              sx={{ flex: 1 }}
            />
            <TextField
              label="Достоверность, %"
              type="number"
              value={sampleAlpha}
              onChange={e => setSampleAlpha(Number(e.target.value))}
              fullWidth
              sx={{ flex: 1 }}
            />
            <TextField
              label="Мощность, %"
              type="number"
              value={samplePower}
              onChange={e => setSamplePower(Number(e.target.value))}
              fullWidth
              sx={{ flex: 1 }}
            />
          </Box>
          <Box sx={{ textAlign: 'center', mb: 2 }}>
            <Button variant="contained" onClick={calcSampleSize}>
              Рассчитать размер выборки
            </Button>
          </Box>
          {sampleResult && (
            <Typography align="center" sx={{ fontWeight: 'bold' }}>
              Необходимый размер выборки на группу: {sampleResult}<br/>
              Всего: {sampleResult * 2}
            </Typography>
          )}
        </Paper>
      </Box>
    </Container>
  )
}

export default App
