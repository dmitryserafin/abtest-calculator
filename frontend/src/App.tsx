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
  TableRow
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
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
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
}

function App() {
  const [controlSuccess, setControlSuccess] = useState('')
  const [controlTotal, setControlTotal] = useState('')
  const [variantSuccess, setVariantSuccess] = useState('')
  const [variantTotal, setVariantTotal] = useState('')
  const [result, setResult] = useState<ABTestResult | null>(null)
  const [error, setError] = useState('')
  const [aPriorAlpha, setAPriorAlpha] = useState('1')
  const [aPriorBeta, setAPriorBeta] = useState('1')
  const [bPriorAlpha, setBPriorAlpha] = useState('1')
  const [bPriorBeta, setBPriorBeta] = useState('1')

  const handleCalculate = async () => {
    try {
      setError('')
      const response = await axios.post('https://abtest-calculator.onrender.com/calculate', {
        a_success: parseInt(controlSuccess),
        a_total: parseInt(controlTotal),
        b_success: parseInt(variantSuccess),
        b_total: parseInt(variantTotal),
        a_prior_alpha: parseInt(aPriorAlpha),
        a_prior_beta: parseInt(aPriorBeta),
        b_prior_alpha: parseInt(bPriorAlpha),
        b_prior_beta: parseInt(bPriorBeta)
      })
      setResult(response.data)
    } catch (err) {
      setError('Ошибка при расчете. Проверьте введенные данные.')
      console.error(err)
    }
  }

  const chartData = result ? {
    labels: result.x_values.map(x => (x * 100).toFixed(1) + '%'),
    datasets: [
      {
        label: 'Контрольная группа',
        data: result.a_distribution,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.4
      },
      {
        label: 'Тестовая группа',
        data: result.b_distribution,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.4
      }
    ]
  } : null

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Байесовские распределения конверсий',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Нормализованная плотность вероятности'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Конверсия'
        },
        ticks: {
          callback: function(tickValue: number | string) {
            return `${(Number(tickValue) * 100).toFixed(1)}%`
          }
        }
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
                label="Количество успехов"
                type="number"
                value={controlSuccess}
                onChange={(e) => setControlSuccess(e.target.value)}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Общее количество"
                type="number"
                value={controlTotal}
                onChange={(e) => setControlTotal(e.target.value)}
                margin="normal"
              />
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                <TextField
                  label="Априорное α"
                  type="number"
                  value={aPriorAlpha}
                  onChange={(e) => setAPriorAlpha(e.target.value)}
                  size="small"
                />
                <TextField
                  label="Априорное β"
                  type="number"
                  value={aPriorBeta}
                  onChange={(e) => setAPriorBeta(e.target.value)}
                  size="small"
                />
              </Box>
            </Box>
            
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" gutterBottom>
                Тестовая группа
              </Typography>
              <TextField
                fullWidth
                label="Количество успехов"
                type="number"
                value={variantSuccess}
                onChange={(e) => setVariantSuccess(e.target.value)}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Общее количество"
                type="number"
                value={variantTotal}
                onChange={(e) => setVariantTotal(e.target.value)}
                margin="normal"
              />
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                <TextField
                  label="Априорное α"
                  type="number"
                  value={bPriorAlpha}
                  onChange={(e) => setBPriorAlpha(e.target.value)}
                  size="small"
                />
                <TextField
                  label="Априорное β"
                  type="number"
                  value={bPriorBeta}
                  onChange={(e) => setBPriorBeta(e.target.value)}
                  size="small"
                />
              </Box>
            </Box>
          </Box>

          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button 
              variant="contained" 
              onClick={handleCalculate}
              disabled={!controlSuccess || !controlTotal || !variantSuccess || !variantTotal}
            >
              Рассчитать
            </Button>
          </Box>
        </Paper>

        {error && (
          <Typography color="error" align="center" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {result && (
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

            <Box sx={{ height: 400 }}>
              {chartData && <Line options={chartOptions} data={chartData} />}
            </Box>

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
      </Box>
    </Container>
  )
}

export default App
