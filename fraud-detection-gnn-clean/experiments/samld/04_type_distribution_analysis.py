import torch
from collections import Counter
import yaml

CONFIG_PATH = "configs/samld_binary.yaml"


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dict = torch.load(f"{cfg['output_prefix']}.pt", weights_only=False)
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    val_mask = data_dict["val_mask"]
    test_mask = data_dict["test_mask"]
    num_types = data_dict["num_types"]

    print(f"تعداد کل انواع پول‌شویی: {num_types}\n")

    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        types_in_split = y_type[mask]
        illicit_types = types_in_split[types_in_split != -1].tolist()
        counts = Counter(illicit_types)
        print(f"=== {name}، تعداد کل illicit با نوع مشخص: {len(illicit_types)} ===")
        for type_id in range(num_types):
            c = counts.get(type_id, 0)
            flag = "  ⚠ کمتر از ۵ نمونه" if c < 5 else ""
            print(f"  نوع {type_id:2d}: {c:4d} نمونه{flag}")
        print()


if __name__ == "__main__":
    main()
