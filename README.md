# AI Legends 2026 — Sales Forecasting (2nd Place)

🏆 **2-р байр** — AI Legends 2026 ML Model Development тэмцээн

Энэ репо нь 3 өөр түвшний борлуулалтын урьдчилсан таамаглалын асуудлыг шийдсэн **hierarchical forecasting** pipeline-ийг агуулна:
- **Store Daily** — дэлгүүр тус бүрийн өдрийн нийт борлуулалт
- **Category Daily** — ангилал тус бүрийн өдрийн борлуулалт
- **Product Weekly** — бүтээгдэхүүн тус бүрийн долоо хоногийн борлуулалт (sparse)

## Арга барил (Approach)

### Үндсэн загвар
- **LightGBM Regressor** log-transformed target дээр, multi-task forecasting
- Task бүрд тохирсон тусдаа feature set болон hyperparameter

### Feature Engineering
| Feature төрөл | Тайлбар |
|---|---|
| **Date features** | year, month, day, dayofweek, weekofyear, quarter, is_weekend, sin/cos cyclic encoding |
| **Lag features** | store/category: lag-1, 7, 14, 28; product: lag-1, 2, 4, 8, 13, 26, 52 |
| **Rolling stats** | mean, std, max өөр өөр window-той (7/14/28 өдөр, эсвэл 2/4/8/13 7 хоногтой) |
| **Hierarchical features** | category_mean, store_mean, product_mean — global anchor байдлаар |
| **Learned week seasonality** | Train-only WeekOfYear multiplier (leakage-free) |

### Product Weekly-ийн онцгой анхаарал
Бүтээгдэхүүний 70%+ нь долоо хоногт 1-ээс бага зардаг (sparse) байсан тул:
- **Zero-aware baseline:** sparse бүтээгдэхүүнд LightGBM-ийн оронд log-mean window forecast
- **Sparse threshold tuning:** дундаж борлуулалтын босгоор stratified routing
- **P99 capping:** outlier бүтээгдэхүүний нөлөөг хязгаарлах
- **Learned week-of-year multiplier:** улирлын тренд train-from data-аар сурсан

### Validation стратеги
- Time-based split (сүүлийн 1 сар нь validation)
- Validation үе шатанд параметр сонголт хийсэн grid search
- Final submit үед FULL train data дээр дахин сургасан

## Үр дүн

| Task | RMSLE (validation) |
|---|---|
| Store Daily | SMAPE: 17.7256%  |  RMSLE: 0.26600  |  RMSE: 170.9450 |
| Category Daily | SMAPE: 15.6404%  |  RMSLE: 0.22819  |  RMSE: 158.8425 |
| Product Weekly | SMAPE: 80.1315%  |  RMSLE: 0.65479  |  RMSE: 18.7616 |

**Public Leaderboard:** 2-р байр

## Файлын бүтэц

```
.
├── README.md
├── requirements.txt
├── notebook.ipynb       # Үндсэн шийдэл (Kaggle Notebook)
└── .gitignore
```

## Notebook-ийн агуулга

| Cell | Зорилго |
|---|---|
| 1 | Data ачаалал |
| 2 | Product weekly sparsity шинжилгээ |
| 3 | Helper функцууд (date features, RMSLE, SMAPE) |
| 4 | Hierarchical mean feature |
| 5 | LightGBM forecast pipeline (store/category) |
| 5b | Product weekly sparse-aware baseline |
| 6 | Local validation |
| 6b | Product weekly grid search |
| 7 | Final submission |

## Ажиллуулах

Энэхүү notebook нь Kaggle орчинд бичигдсэн. Локал ажиллуулахын тулд:

```bash
pip install -r requirements.txt
```

Дараа нь `notebook.ipynb` дотор `train.csv`, `test.csv`-ийн зам тохируулна.

## Ашигласан технологи

- Python 3.10+
- pandas, numpy
- LightGBM
- Jupyter

## Зохиогч

[@Detsu123](https://github.com/Detsu123)

AI Legends 2026 — Mongolian AI Competition
