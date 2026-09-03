# فهرست آزمایش‌ها

هر ردیف = یک اسکریپت experiment. وضعیت‌ها: ⭐ نهایی/تولید، 🗄 آرشیو
(کنارگذاشته‌شده، برای فصل تجربه‌های ناموفق نگه داشته شده)، ⏳ تأیید نشده
(ادعا هست، شواهد خام هنوز چک نشده).

## Elliptic

| فایل قدیم | چی تست شد | نتیجه | وضعیت | مسیر جدید |
|---|---|---|---|---|
| step2_simple_gcn.py | GCN ساده | baseline | 🗄 | experiments/elliptic/00_baseline_gcn.py |
| step3_professional_sage.py | GraphSAGE اولیه | زمینه‌ساز، بدون StructuralOnlyGraphSAGE | 🗄 | experiments/elliptic/archived/professional_sage_v1.py |
| step4_gat_attention.py | GATv2 attention | ⏳ داده‌ی ۵-seed جدا برای GAT در Step12 نبود؛ هنوز تأیید نشده | 🗄 | experiments/elliptic/archived/gat_attention.py |
| step5_hybrid_model.py | مدل ترکیبی | ⏳ تأیید نشده | 🗄 | experiments/elliptic/archived/hybrid_model.py |
| step9_triple_attention.py | سه‌جریانه با gating | **تأیید شده (Step12)**: نسخه‌ی خام سه‌جریانی p=0.0105 (GraphSAGE به‌طور معنادار بهتر)؛ نسخه‌ی +زمانی‌فقط p=0.0270 (GraphSAGE بهتر)؛ نسخه‌ی +کلی‌اصلاح‌شده p=0.103 (معنادار نیست) | 🗄 | experiments/elliptic/archived/triple_attention.py |
| step9c_film_modulation.py | FiLM modulation | **تأیید شده (Step12)**: p=0.1412، معنادار نیست | 🗄 | experiments/elliptic/archived/film_modulation.py |
| step10_skip_gcn.py | Skip-GCN | **تأیید شده (Step12)**: p=0.0456، GraphSAGE به‌طور معنادار بهتر | 🗄 | experiments/elliptic/archived/skip_gcn.py |
| step11_tie_baseline.py | بدون گراف (embedding زمانی+MLP) | baseline مرجع | ⭐ | src/fraud_gnn/models/baselines.py |
| Step12_significance_summary.py | GraphSAGE vs TIE، ۵ seed | p=0.5576 | 🗄 (superseded by step32) | experiments/elliptic/02_significance_test_5seed.py |
| Step13_heterophily_check.py | چرا گراف کمک نمی‌کنه | تشخیصی | 🗄 | experiments/elliptic/03_diagnostics_heterophily.py |
| Step14_edge_leakage_check.py | سلامت روش‌شناسی | تشخیصی | 🗄 | experiments/elliptic/04_diagnostics_edge_leakage.py |
| Step15_xavier_graphnorm.py | Xavier init + GraphNorm | داخل SAGEBlock نهایی شد | 🗄 | (داخل graphsage.py) |
| Step16_hyperparameter_search.py | جست‌وجوی هایپرپارامتر | — | 🗄 | experiments/elliptic/05_hyperparameter_search.py |
| Step17_diagnostics_and_checkpoint.py | چک‌پوینت نهایی Elliptic | مرجع (نسخه‌ی dropout مضاعف) | ⭐ | experiments/elliptic/06_final_checkpoint.py |
| Step32_elliptic_graph_vs_nograph_more_seeds.py | GraphSAGE vs TIE، ۱۵ seed | **تأیید شده مستقل**: F1 p=0.062، MCC p=0.199، AUC p<0.0001 (TIE بهتر)، PR-AUC p=0.0017 (TIE بهتر) | ⭐ | experiments/elliptic/07_significance_test_15seed.py |
| Step34_elliptic_binary_explainability.py | Explainability باینری | — | ⭐ | experiments/elliptic/08_explainability.py |

## SAML-D

| فایل قدیم | چی تست شد | نتیجه | وضعیت | مسیر جدید |
|---|---|---|---|---|
| step6-8 | اکتشاف، feature engineering، baseline جدولی | — | 🗄 | experiments/samld/00-01 |
| step18/18v3/19/19b/20 | ساخت گراف، sampling، رفع imbalance | — | ⭐ (v3 فقط) | src/fraud_gnn/data/graph_utils.py |
| step21_samld_tabular_baseline.py | baseline جدولی نوع | — | 🗄 | experiments/samld/01_tabular_baseline.py |
| step22_samld_multiseed*.py | آموزش GraphSAGE باینری چند-seed | چک‌پوینت‌های تولید | ⭐ (فقط ۱۵seed) | experiments/samld/03_binary_training_multiseed.py |
| step25_samld_multitask.py | مدل واحد multi-task | جایگزین شد با step28 | 🗄 | experiments/samld/archived/multitask_variant.py |
| step28_samld_multitask_fixed_15seed.py | multi-task اصلاح‌شده | **تأیید شده از لاگ واقعی**: type macro_f1=0.1515±0.0328، p=0.0001 در برابر no-graph. باینری: p90 Prec=0.0180±0.0030 Rec=0.3044±0.0509 (مقایسه با step29 هنوز ناقص) | 🗄 (کنار گذاشته شد، هزینه‌ی باینری) | experiments/samld/archived/multitask_variant.py |
| step25_samld_type_classification_15seed.py | GraphSAGE embedding فقط، طبقه‌بند جدا | **تأیید شده از لاگ واقعی**: macro_f1=0.1123±0.0144، p=0.161 (معنادار نیست) | ⭐ (پایه‌ی مسیر برنده) | src/fraud_gnn/models/graphsage.py + experiments/samld/06_typology_two_stage.py |
| step26_samld_structural_features_15seed.py | + ویژگی ساختاری | **تأیید شده از لاگ واقعی**: macro_f1=0.1296±0.0180، p=0.0002 (معنادار) | ⭐ | داخل src/fraud_gnn/data/samld.py |
| step29_binary_only_fair_reference.py | باینری خالص، بدون هزینه‌ی multi-task | ⏳ لاگ کامل ۱۵-seed هنوز لازمه | ⭐ | experiments/samld/07_fair_reference_baseline.py |
| step30_type_graph_ablation_15seed.py | نردبان کامل: no-graph تا multi-task | **تأیید شده از لاگ واقعی** (جدول بالا در architecture.md) | ⭐ | experiments/samld/08_graph_ablation_typology.py |
| step31_samld_coarse_family_classification.py | طبقه‌بندی خانواده (۳ کلاس) | از step25 import می‌کنه، نه multitask | ⭐ | experiments/samld/09_coarse_family_classification.py |
| step33/35/36 | Explainability، ویژگی‌های رفتاری | — | ⭐ | experiments/samld/10-12 |

## Learnable Adjacency (فرا-پروژه)

| فایل قدیم | چی تست شد | نتیجه | وضعیت |
|---|---|---|---|
| learnable_adjacency.py + experiment_*.py (۴ فایل) | یادگیری adjacency به‌جای گراف ثابت | نتیجه‌ی منفی، cheap pre-check | 🗄 archived/negative_results/ |

---

## کارهای باز قبل از این‌که این سند «کامل» بشه

- [ ] لاگ کامل ۱۵-seed step29 رو پیدا/اجرا کن، paired t-test seed-matched با step28 بزن
- [x] ادعای فاز Elliptic («GCN/GAT/Hybrid/... هیچ‌کدوم بهتر نبودن») با CSV/داده‌ی خام Step12 تأیید شد — به‌جز GAT (step4) که هنوز داده‌ی جدا نداره
- [ ] تصمیم نهایی درباره‌ی تفاوت dropout (Step17 در برابر step25) رو اینجا و در architecture.md ثبت کن
