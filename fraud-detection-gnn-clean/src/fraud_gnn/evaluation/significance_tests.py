import numpy as np
from scipy import stats


def paired_significance_test(f1_a, f1_b, name_a="A", name_b="B", verbose=True):
    f1_a = np.array(f1_a, dtype=float)
    f1_b = np.array(f1_b, dtype=float)
    diff = f1_a - f1_b
    t_stat, p_value = stats.ttest_rel(f1_a, f1_b)

    if verbose:
        print(f"\n--- {name_a} در برابر {name_b} ---")
        print(f"میانگین {name_a}: {f1_a.mean():.4f}   میانگین {name_b}: {f1_b.mean():.4f}")
        print(f"میانگین تفاوت: {diff.mean():.4f}   t = {t_stat:.3f}   p = {p_value:.4f}")
        if p_value < 0.05:
            print("تفاوت از نظر آماری معنادار است، p کمتر از 0.05.")
        else:
            print("تفاوت از نظر آماری معنادار نیست، ممکنه فقط نویز seed باشه.")

    return t_stat, p_value
