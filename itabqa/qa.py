import math
import warnings
from enum import Enum
from numbers import Number
from typing import Any, List, Union

import numpy as np
import pandas as pd
import strsimpy
from regex import R
from sklearn.metrics import mean_absolute_percentage_error
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from transformers import Pipeline, pipeline

from itabqa.objs import AnswerType, Category
from itabqa.parse_number import convert_num

warnings.filterwarnings("ignore")


class QA_MODEL(str, Enum):
    TAPAS_WTQ = "google/tapas-large-finetuned-wtq"
    TAPAS_WIKISQL = "google/tapas-large-finetuned-wikisql-supervised"
    TAPAS_SQA = "google/tapas-large-finetuned-sqa"
    TAPEX_WTQ = "microsoft/tapex-large-finetuned-wtq"
    OMNITAB_WTQ = "neulab/omnitab-large-finetuned-wtq"
    OMNITAB = "neulab/omnitab-large"
    OMNITAB_FT_RAW_2 = (
        "/disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-100k-raw-2"
    )
    OMNITAB_FT_RAW_1 = (
        "/disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-100k-raw-1"
    )
    OMNITAB_FT_RAW_3 = (
        "/disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-all-raw"
    )

    OMNITAB_FT_HEADER_1 = (
        "/disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-100k-header-1"
    )
    OMNITAB_FT_HEADER_2 = (
        "/disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-100k-header-2"
    )
    OMNITAB_FT_HEADER_3 = (
        "/disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-200k-header-2"
    )
    OMNITAB_FT_HEADER_4 = "/disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-all-header"
    # omnitab-large-1024shot-finetuned-wtq-1024shot
    # omnitab-large-finetuned-wtq
    # ZERO = "neulab/omnitab-large-finetuned-wtq"


Norm_Lev_distance = NormalizedLevenshtein()


def get_qa_results(
    model_name: QA_MODEL,
    pipeline: Pipeline,
    table: pd.DataFrame,
    questions: List[str],
    sample_id: str = None,
) -> List[Union[str, int]]:
    results = []
    reload_pipeline = False

    # if model_name is QA_MODEL.ZERO:
    #     results.append(0)
    #     continue

    try:
        tokenizer_kwargs = {"truncation": True}
        answers = pipeline(table=table, query=questions, **tokenizer_kwargs)
    except Exception:
        print(f"Error: {sample_id}")
        results.append(0)
        reload_pipeline = True
    if isinstance(answers, dict) and len(questions) == 1:
        answers = [answers]
    for tmp in answers:
        # print(tmp)
        if model_name in [QA_MODEL.TAPAS_WTQ, QA_MODEL.TAPAS_WIKISQL]:
            if tmp["aggregator"] == "COUNT":
                result = len(tmp["cells"])
            elif tmp["aggregator"] == "SUM":
                sum_answer = 0
                for i in tmp["cells"]:
                    to_num = convert_num(i)
                    if to_num is not None:
                        sum_answer += to_num
                result = sum_answer
            elif tmp["aggregator"] == "AVERAGE":
                avg_answer = []
                for i in tmp["cells"]:
                    to_num = convert_num(i)
                    if to_num is not None:
                        avg_answer.append(to_num)
                if avg_answer:
                    result = np.mean(avg_answer)
                else:
                    result = 0
            else:  # NONE
                result = tmp["answer"]
        else:
            if "answer" not in tmp:
                result = 0
            else:
                tmp = tmp["answer"]
                result = convert_num(tmp)
                if result is None:
                    result = tmp

        # elif model is MODEL.TAPEX:
        #     pass
        # elif model is MODEL.OMNITAB:
        #     pass
        # elif model is MODEL.INSURERS:
        #     pass

        to_num = convert_num(result)
        if to_num is not None:
            result = to_num
        else:
            result = 0

        # assign no prediciton to 0
        if (
            result is None
            or not result
            or result in [" nan", "nan"]
            or (isinstance(result, Number) and math.isnan(result))
        ):
            result = 0
        results.append(result)
    return results, reload_pipeline


def cal_anls(predictions, references):
    return Norm_Lev_distance.similarity(str(predictions), str(references))


def cal_mape(predict: float, answer: float) -> float:
    result = mean_absolute_percentage_error([predict], [answer])
    if result > 1:
        result = 1
    return 1 - result


def cal_metric(answer_type: AnswerType, answer: Any, predict: Any) -> float:
    if answer_type == AnswerType.NUMERIC:
        answer_str = str(convert_num(answer))
    else:
        # Error logic:
        # convert
        answer_is_num = convert_num(answer)
        if answer_is_num is not None:
            answer_str = str(answer_is_num)
        else:
            answer_str = answer

    predict_str = str(predict)
    score_anls = cal_anls(predict_str, answer_str)
    score_num = 0
    if answer_type == AnswerType.NUMERIC:
        answer_num = convert_num(answer_str)
        predict_num = convert_num(predict_str)

        if answer_num is not None and predict_num is not None:
            score_num = cal_mape(predict_num, answer_num)

        score = np.linalg.norm(
            np.array([score_anls, score_num]) * np.array([0.5, 0.5])
        ) / np.linalg.norm(np.array([0.5, 0.5]))
    else:
        score = score_anls
    return score


def cal_metric_competition(answer_type: AnswerType, answer: Any, predict: Any):
    if isinstance(answer, Number) and math.isnan(answer):
        answer = 0
    pred_type_numeric = True
    if answer_type == AnswerType.NUMERIC:
        answer = round(float(answer), 2)
        try:
            predict = round(float(predict), 2)
        except:
            pred_type_numeric = False
    else:
        pred_type_numeric = False
    anld = Norm_Lev_distance.distance(str(answer), str(predict))
    anls = 1.0 - anld
    # print("ANLS : ", round(anls, 2))
    pct_closeness = 0
    if answer_type == AnswerType.NUMERIC and pred_type_numeric:
        if abs(answer) != 0:
            pct_deviation = float(abs(abs(predict) - abs(answer))) / abs(answer)
        elif abs(predict) != 0:
            pct_deviation = 1.0
        else:
            pct_deviation = 0.0
        pct_deviation = min(pct_deviation, 1.0)
        pct_closeness = 1.0 - pct_deviation
        # print("Percentage absolute closeness score : ", round(pct_closeness, 2))
    score = 0.0
    if answer_type == AnswerType.NUMERIC and pred_type_numeric:
        score = math.sqrt((anls * anls) + (pct_closeness * pct_closeness)) / math.sqrt(
            2
        )
    elif answer_type == AnswerType.NUMERIC and not pred_type_numeric:
        score = anls / math.sqrt(2)
    else:
        score = anls
    # print("Final aggregated score : ", round(score, 2))
    return score, anls, pct_closeness


def eval_results(
    predicts: List[Any],
    answers: List[Any],
    q_types: List[AnswerType],
    q_categories: List[Category],
):
    scores_sum = 0
    scores_weight = 0
    cat_weights = {
        Category.CATEGORY_1: 0.25,
        Category.CATEGORY_2: 0.4,
        Category.CATEGORY_3: 0.5,
        Category.CATEGORY_4: 0.75,
        Category.CATEGORY_5: 1,
    }
    for predict, answer, q_type, q_category in zip(
        predicts, answers, q_types, q_categories
    ):
        score, anls, pct_closeness = cal_metric_competition(q_type, answer, predict)
        # print(
        #     f"{score:.4f}|{predict}|{anls:.4f}|{pct_closeness:.4f}|{q_type.value}|{cat_weights[q_category]}|{answer}"
        # )
        scores_sum += score * cat_weights[q_category]
        scores_weight += cat_weights[q_category]

    return scores_sum / scores_weight


def run_example():
    data = {
        "year": [1896, 1900, 1904, 2004, 2008, 2012],
        "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"],
    }
    table = pd.DataFrame.from_dict(data)
    table = table.astype(str)
    questions = [
        "In which year did beijing host the Olympic Games?",
        "which city organized olympic in 2012?",
        "In which year did beijing and paris host the Olympic Games?",
        "which city organized olympic in 2012, and 1896?",
        "How many years between athens and paris?",
    ]
    answers = [2008, "london", "2008, 1900", "london, athens", 4]
    q_types = [
        AnswerType.NUMERIC,
        AnswerType.TEXT,
        AnswerType.TEXT,
        AnswerType.TEXT,
        AnswerType.NUMERIC,
    ]
    q_categories = [
        Category.CATEGORY_1,
        Category.CATEGORY_1,
        Category.CATEGORY_1,
        Category.CATEGORY_1,
        Category.CATEGORY_1,
    ]

    qa_models = {}
    for m in QA_MODEL:
        qa_models[m] = pipeline(task="table-question-answering", model=m.value)
    for model in QA_MODEL:
        print(model.value)
        results = get_qa_results(model, qa_models[model], table, questions)
        eval_results(results, answers, q_types, q_categories)


if __name__ == "__main__":
    convert_num(6418.0)
    run_example()
