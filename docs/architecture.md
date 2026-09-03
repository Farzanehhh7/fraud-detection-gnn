# معماری پروژه — نسخه‌ی تأییدشده

این سند تنها منبع مرجع معماریه. هیچ کامنت یا docstring داخل کدهای `src/` و
`experiments/` نیست — هر توضیحی که لازم باشه، اینجاست.

هر عدد این سند یا مستقیم از کد گیت‌هاب استخراج شده، یا از رو لاگ خام اجرای
واقعی محاسبه شده — نه از حافظه یا حدس. منبع هرکدوم مشخص شده.

---

## معماری نهایی

```
مرحله ۱ (باینری illicit/licit، هر دو دیتاست):
  StructuralOnlyGraphSAGE(in_channels=8, hidden_channels=64, out_channels=2)
  GraphSAGE خالص، بدون attention
  چک‌پوینت‌ها: samld_seed_<42/1/7/123/2024/...>.pt (از step22)

مرحله ۲ (فقط SAML-D — نوع + خانواده‌ی تراکنش):
  embeddings = extract_embeddings(model_stage1, ...)
  structural = [in_degree, out_degree, gather_scatter_score,
                degree_ratio, reciprocal_count, is_pass_through]
  X_combined = concat(embeddings, structural)
  type_clf   = LogisticRegression روی X_combined  -> ۱۰ نوع
  family_clf = LogisticRegression روی X_combined  -> ۳ خانواده
       0 = ساختاری/چندجهشی (Fan_In, Fan_Out, Cycle, ...)
       1 = حجمی/آستانه‌ای (Structuring, Smurfing, ...)
       2 = رفتاری/زمانی (Behavioural_Change_1/2, Deposit-Send)
```

منبع: `fraud_dashboard/detector/pipeline.py` — تأیید مستقیم با خوندن کد،
شامل import صریح `StructuralOnlyGraphSAGE` از `step25_samld_type_classification`
و ترکیب `X_combined = np.concatenate([h2, struct_scaled], axis=1)`.

Elliptic فقط از مرحله‌ی ۱ استفاده می‌کنه (لیبل نوع نداره).

---

## معماری‌های رد شده (آرشیو شده، نه حذف)

| معماری | فایل قدیم | نتیجه |
|---|---|---|
| GCN ساده | step2 | baseline اولیه |
| GAT / GATv2 attention | step4 | بهبود آماری‌معنادار نداشت |
| Hybrid model | step5 | کنار گذاشته شد |
| Triple Attention | step9 | کنار گذاشته شد |
| FiLM modulation | step9c | کنار گذاشته شد |
| Skip-GCN | step10 | کنار گذاشته شد |
| Multi-task backbone (باینری+نوع با loss ترکیبی) | step25_multitask, step28 | نوع را بهتر می‌کرد، ولی هزینه‌ی باینری داشت (پایین‌تر ببین) |
| Learnable Adjacency | learnable_adjacency.py | نتیجه‌ی منفی، cheap pre-check |

**به‌روزرسانی: تأیید شد.** منبع: `Step12_significance_summary.py`، ۵ seed
یکسان (۴۲, ۱, ۷, ۱۲۳, ۲۰۲۴) روی همه‌ی نسخه‌ها، paired t-test مستقل توسط
Claude محاسبه شد (نه فقط خوندن کد):

| مقایسه (در برابر GraphSAGE ساختاری، mean F1=0.4427) | mean F1 نسخه‌ی مقابل | p-value | نتیجه |
|---|---|---|---|
| در برابر TIE (بدون گراف) | 0.4229 | 0.5576 | معنادار نیست |
| در برابر Skip-GCN | 0.3863 | **0.0456** | GraphSAGE به‌طور معنادار بهتر |
| در برابر ساختاری+کلی (نسخه‌ی اصلاح‌شده) | 0.3826 | 0.1030 | معنادار نیست |
| در برابر FiLM + کلی | 0.4021 | 0.1412 | معنادار نیست |
| در برابر سه‌جریانی خام (Triple، gate خام) | 0.3346 | **0.0105** | GraphSAGE به‌طور معنادار بهتر |
| در برابر ساختاری+زمانی | 0.3829 | **0.0270** | GraphSAGE به‌طور معنادار بهتر |

نتیجه‌ی دقیق‌تر از ادعای قبلی: GraphSAGE ساختاری در برابر Skip-GCN، نسخه‌ی
سه‌جریانی خام، و نسخه‌ی ساختاری+زمانی **به‌طور معنادار بهتره** — نه فقط
«بهتر بود، رد شد»، بلکه آماری تأییدشده. در برابر TIE، نسخه‌ی global اصلاح‌شده،
و FiLM، تفاوت معنادار نیست (نه GraphSAGE قطعاً بهتره، نه بدتر).

---

## یافته‌ی آماری — Elliptic (ارزش گراف، وابسته به معیار)

منبع: `Step32_elliptic_graph_vs_nograph_more_seeds.py`، ۱۵ seed، paired
t-test مستقیم روی `step32_graphsage_15seed_results.csv` و
`step32_tie_15seed_results.csv` (محاسبه‌ی مستقل انجام شد، نه فقط خوندن کد).

| معیار | میانگین GraphSAGE | میانگین TIE (بدون گراف) | p-value | نتیجه |
|---|---|---|---|---|
| F1 | 0.4447 | 0.4092 | 0.062 | معنادار نیست (GraphSAGE ۱۰/۱۵ seed برنده) |
| MCC | 0.4191 | 0.3999 | 0.199 | معنادار نیست |
| AUC | 0.8699 | 0.8903 | <0.0001 | TIE به‌طور معنادار بهتر |
| PR-AUC | 0.3552 | 0.4585 | 0.0017 | TIE به‌طور معنادار بهتر |

نتیجه: روی Elliptic، گراف روی معیارهای رتبه‌بندی (AUC، PR-AUC) ضرر معنادار
می‌زنه؛ روی معیارهای threshold-based (F1، MCC) تفاوت معنادار نیست.
**«گراف کمکی نمی‌کنه» یا «گراف بی‌فایده‌ست» ادعای نادرستیه — ادعای درست
متریک‌محوره.**

---

## یافته‌ی آماری — SAML-D (گراف معنادار کمک می‌کنه)

منبع: `step30_type_graph_ablation_15seed.py`، لاگ واقعی اجرا (۱۵ seed).

| نسخه | macro F1 | p (در برابر no-graph) | معنادار؟ |
|---|---|---|---|
| بدون گراف (۸ ویژگی خام) | 0.1068 ± 0.0000 | — | — |
| GNN embedding فقط (step25) | 0.1123 ± 0.0144 | 0.161 | ❌ نه |
| embedding + structural (step26) | 0.1296 ± 0.0180 | 0.0002 | ✅ بله |
| multi-task backbone (step28) | 0.1515 ± 0.0328 | 0.0001 | ✅ بله |

نتیجه: روی SAML-D، اضافه‌کردن ساختار گراف (چه به‌صورت ویژگی ساختاری، چه
multi-task) به‌طور معنادار طبقه‌بندی نوع را بهبود می‌ده — درست برعکس Elliptic.

**نتیجه‌ی کلی پروژه: ارزش گراف به دیتاست/تسک وابسته‌ست، نه یک حکم universal.**

---

## چرا two-stage به‌جای multi-task نهایی شد (در حال تکمیل)

Multi-task (step28) روی نوع‌طبقه‌بندی بهتر از two-stage عمل کرد
(macro F1 = 0.1515 در برابر 0.1296). فرضیه: این بهبود به قیمت افت عملکرد
باینری تموم شده (پدیده‌ی شناخته‌شده در multi-task learning).

میانگین محاسبه‌شده از لاگ خام ۱۵-seed خود step28 (باینری):

| صدک | Precision | Recall |
|---|---|---|
| ۹۰ | 0.0180 ± 0.0030 | 0.3044 ± 0.0509 |
| ۹۵ | 0.0173 ± 0.0046 | 0.1469 ± 0.0382 |
| ۹۹ | 0.0215 ± 0.0101 | 0.0364 ± 0.0170 |

**باز مونده:** مقایسه‌ی paired و seed-matched با `step29_binary_only_fair_reference.py`
(باینری خالص) هنوز کامل نشده — نیاز به لاگ کامل ۱۵-seed step29 داره، نه فقط
یه خط خلاصه. تا اون موقع، این بخش به‌عنوان «در حال تأیید» علامت‌گذاری می‌مونه،
نه نتیجه‌ی قطعی.

---

## ناسازگاری شناخته‌شده: دو پیاده‌سازی مستقل از یک اسم

`StructuralOnlyGraphSAGE` در `Step17_diagnostics_and_checkpoint.py` (مسیر
Elliptic) و `step25_samld_type_classification.py` (مسیر SAML-D) دو کلاس
**مستقل** هستن، نه یکی که از دیگری import شده باشه — تأیید شده با دیف مستقیم
کد:

- **Step17**: `SAGEBlock` واسط با `dropout(p=0.2)` داخلی + `dropout(p=0.3)`
  بیرونی روی هر لایه (دو dropout پشت‌سرهم).
- **step25**: `SAGEConv` خام با فقط `dropout(p=0.3)` بعد از ReLU.

هیچ کامنتی در کد دلیل این تفاوت رو مستند نکرده بود. تصمیم گرفته شد: [اینجا
تصمیم نهایی رو بنویس — مستندسازی صادقانه‌ی تفاوت، یا تست تجربی که مشخص کرد
تفاوت معنادار بود/نبود].

---

## محدودیت شناخته‌شده: تایپولوژی «Deposit-Send»

طبق docstring اصلی `step31`، تعریف دقیق «Deposit-Send» در مقاله‌ی منبع پیدا
نشد؛ صرفاً بر اساس اسمش در خانواده‌ی ۲ (رفتاری/زمانی) دسته‌بندی شده. این یک
inference است، نه citation تأییدشده — باید در بخش محدودیت‌های پایان‌نامه بیاد.
