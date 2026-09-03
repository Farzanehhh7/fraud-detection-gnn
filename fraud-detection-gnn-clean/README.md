# تشخیص تقلب مالی با Graph Neural Networks

سیستم دومرحله‌ای تشخیص تقلب: GraphSAGE ساختاری برای تشخیص illicit/licit، به‌علاوه‌ی طبقه‌بندهای جدا برای نوع و خانواده‌ی تراکنش روی SAML-D. جزئیات کامل معماری و یافته‌های آماری در [docs/architecture.md](docs/architecture.md)، فهرست همه‌ی آزمایش‌ها در [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## نصب

```powershell
git clone https://github.com/Farzanehhh7/detection.git
cd fraud-detection-gnn-clean
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## دیتاست

فایل‌های خام (Elliptic Bitcoin + SAML-D) به‌دلیل حجم روی گیت‌هاب نیستن:

https://huggingface.co/datasets/farzanehwr/fraud-detection-artifacts

دانلود کن و توی `datasets/` بذار:
```
datasets/
├── elliptic_txs_classes.csv
├── elliptic_txs_edgelist.csv
├── elliptic_txs_features.csv
└── SAML-D.csv
```

## بازتولید نتایج اصلی

```powershell
python experiments/samld/02_graph_construction.py
python experiments/samld/03_binary_training_multiseed.py
python experiments/samld/06_typology_two_stage.py
python experiments/samld/08_graph_ablation_typology.py
python experiments/samld/09_coarse_family_classification.py

python experiments/elliptic/06_final_checkpoint.py
python experiments/elliptic/07_significance_test_15seed.py
```

هر اجرا یک ردیف به `outputs/run_registry.csv` اضافه می‌کنه (کامیت گیت، seed، معیار، مسیر چک‌پوینت) — منبع مستقیم جدول‌های پایان‌نامه.

## ساختار

```
src/fraud_gnn/       کتابخونه‌ی مشترک — مدل‌ها، لود دیتا، معیارها
experiments/         اسکریپت‌های اجرا، به‌ترتیب شماره
experiments/*/archived/   معماری‌های کنارگذاشته‌شده (GAT, Triple Attention, FiLM, Hybrid, Skip-GCN, Multi-task)
outputs/             چک‌پوینت‌ها، لاگ‌ها، run_registry.csv
dashboard/           داشبورد Django، import از src/fraud_gnn
docs/                architecture.md + EXPERIMENTS.md
```

## داشبورد

```powershell
cd dashboard
python manage.py migrate
python manage.py runserver
```
