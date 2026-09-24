# Hyperspectral RWW — модульный проект

Проект обрабатывает RWW-файл с RAW-данными гиперспектральной матрицы.

## Геометрия данных

В математических обозначениях используется

\[
P_{x,y,k},
\]

где:

- `X = 696` — первая пространственная координата;
- `Y = количество кадров` — вторая пространственная координата;
- `K = 64` — количество спектральных каналов.

Для предоставленного файла `Y = 865`.

В памяти Python куб хранится как `cube[y, x, k]` с формой `(Y, X, K)`.

## Архитектура

`main.py` ничего не вычисляет сам. Он последовательно получает результат одной функции и передаёт его следующей.

```text
RWW
 ↓
read_rww()
 ↓
split_frames()
 ↓
frames_to_cube()
 ↓
raw cube [Y, X, K]
 ├────────────────────────────────────┐
 │                                    │
 │ визуальный поток                   │ математический поток
 │                                    │
 ▼                                    ▼
build_channel_images()          calculate_mean_spectrum()
 ↓                                    ↓
robust_normalize_channels()     calculate_covariance_matrix()
 ↓                                    ↓
PNG каналов                      calculate_mahalanobis_distance_map()
                                     ↓
                                stretch_distance_map()
                                     ↓
                                create_anomaly_mask()
                                     ↓
                                highlight_anomalies()
```

## Почему потоки разделены

Robust normalization используется только для визуализации. Она не меняет данные, которые поступают в расчёт расстояния Махаланобиса.

Математическая ветка работает с исходными значениями RAW после преобразования в куб.

## Махаланобис

Для каждого пространственного пикселя формируется спектральный вектор размерности `K=64`.

Средний спектр:

\[
\mu = [\mu_1, \mu_2, ..., \mu_K].
\]

Ковариационная матрица:

\[
\Sigma.
\]

Для каждого `(x,y)` рассчитывается квадрат расстояния Махаланобиса:

\[
D^2_M(x,y) =
(P_{x,y}-\mu)^T\Sigma^{-1}(P_{x,y}-\mu).
\]

В коде используется псевдообратная матрица `Σ`, что устойчивее при высокой корреляции спектральных каналов.

## Визуализации

В `data/output/visualization/`:

- `channels/channel_01.png ... channel_64.png` — robust-normalized спектральные каналы;
- `mean_spectrum.png` — средняя визуализация по 64 каналам.

В `data/output/mahalanobis/`:

- `mean_spectrum.npy` — `μ`;
- `covariance_matrix.npy` — `Σ`;
- `distance_map.npy` — карта `D²_M`;
- `mahalanobis_distance.png` — карта расстояний, растянутая для отображения;
- `anomaly_mask.npy` — бинарная маска наиболее отклонённых пикселей;
- `mahalanobis_anomalies_red.png` — красное выделение отклонений на фоновой визуализации.

## Запуск через Poetry

Установить зависимости:

```bash
poetry install
```

Запуск проекта:

```bash
poetry run hyperspectral
```

Либо напрямую:

```bash
poetry run python main.py
```

Тесты:

```bash
poetry run pytest
```

## Основные модули

```text
main.py
cube_builder.py
frame_splitter.py
rww_reader.py

visualization/
├── robust_normalizer.py
├── composite.py
└── image_saver.py

mahalanobis/
├── mean_spectrum.py
├── covariance.py
├── distance.py
├── distance_visualization.py
├── anomaly_mask.py
├── overlay.py
└── image_saver.py
```

Каждый модуль отвечает за одну отдельную операцию, а `main.py` используется только для навигации по pipeline.

### Важное замечание о пороге аномалии

Сейчас `ANOMALY_PERCENTILE = 99.0` используется только для визуального выделения верхних 1% расстояний. Это не является окончательной статистической постановкой порога детектирования. В дальнейшем этот модуль можно заменить отдельным правилом, не меняя расчёт Махаланобиса.
