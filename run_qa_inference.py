from itabqa import loader, objs
from itabqa.qa import QA_MODEL
from run_exp import run_eval_split

if __name__ == "__main__":
    run_eval_split(
        objs.DataSplit.TEST,
        from_ground_truth=False,  # From ground truth or inference models
        qa_models=[QA_MODEL.OMNITAB_FT_RAW_3],
        device_id=3,
        head=0,
        ago_html=False,
        ago_headers=False,
        setting="raw_3_all",
    )
