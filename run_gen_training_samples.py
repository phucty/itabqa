from itabqa import loader, objs
from itabqa.qa import QA_MODEL
from run_exp import run_eval_split

if __name__ == "__main__":
    loader.generate_fine_tune_data(
        output_name="all_raw",  # f"{config.DATA_ROOT}/train_{output_name}.json"
        split=objs.DataSplit.TRAIN,  # SPLIT_DIR = f"{config.DATA_VQA}/{split.value}/ground_truth/"
        n_samples=10_000_000,  # 10_000_000 = all samples, max # sample ~ 1.2M
        train_ratio=1,  # 100% training data
        n_qtype_per_table=100,  # number of samples in 1 category: 100 --> get all
        ago_headers=False,  # not detect headers
    )
