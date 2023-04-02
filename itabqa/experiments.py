import json
import math
import multiprocessing
import os
from collections import Counter, defaultdict
from contextlib import closing
from multiprocessing.pool import Pool
from typing import List

import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from transformers import Pipeline, pipeline

from itabqa import config, io_func, loader, objs, qa
from itabqa.parse_number import convert_num


def init_worker(model_name, qa_pipeline, device_id):
    global _model_name
    global _device_id
    global _qa_pipeline
    _model_name = model_name
    _qa_pipeline = qa_pipeline
    _device_id = device_id


def run_pool_qa(sample):
    # global _qa_pipeline
    predicts, answers, q_types, q_categories, q_id = [], [], [], [], []
    table = pd.DataFrame.from_dict(sample.to_json())
    table = table.astype(str)
    questions = []

    for q in sample.questions_answers.values():
        questions.append(q.question)
        answers.append(q.answer)
        q_types.append(q.answer_type)
        q_categories.append(q.category)
        q_id.append(q.question_id)

    if sample.table is None:
        results = [0] * len(questions)  # heuristic
        reload_pipeline = False
    else:

        results, reload_pipeline = qa.get_qa_results(
            _model_name, _qa_pipeline, table, questions, sample.sample_id
        )

    predicts.extend(results)
    return predicts, answers, q_types, q_categories, q_id, reload_pipeline


def load_qa_pipeline(model_name: qa.QA_MODEL, device_id: int = -1):
    if device_id >= 0 and torch.cuda.is_available():
        qa_pipeline = pipeline(
            task="table-question-answering", model=model_name.value, device=device_id
        )
    else:
        qa_pipeline = pipeline(task="table-question-answering", model=model_name.value)
    return qa_pipeline


def run_qa_pipeline(
    split: objs.DataSplit,
    model_name: qa.QA_MODEL,
    head=0,
    n_process: int = 1,
    device_id: int = -1,
    from_ground_truth: bool = True,
    ago_headers: bool = True,
    ago_html: bool = True,
    setting: str = "temp",
):
    qa_pipeline = load_qa_pipeline(model_name, device_id)
    if n_process == 0:
        n_process = multiprocessing.cpu_count()

    samples = loader.load_samples(
        split,
        head=head,
        from_ground_truth=from_ground_truth,
        ago_headers=ago_headers,
        ago_html=ago_html,
    )
    predicts, answers, q_types, q_categories = [], [], [], []
    print(model_name.value)
    n_questions, n_num, n_text, n_tables = 0, 0, 0, 0
    p_bar = tqdm(total=len(samples))
    init_worker(model_name, qa_pipeline, device_id)
    for i, sample in enumerate(samples):
        # if i < 220:
        #     continue
        (
            tmp_predicts,
            tmp_answers,
            tmp_q_types,
            tmp_q_categories,
            tmp_q_id,
            reload_pipeline,
        ) = run_pool_qa(sample)
        if reload_pipeline:
            qa_pipeline = load_qa_pipeline(model_name, device_id)
            init_worker(model_name, qa_pipeline, device_id)
        n_tables += 1
        # save json file
        if split == objs.DataSplit.TEST:
            output_file = f"{config.HOME_ROOT}/answers/{setting}/answers"
            if not os.path.exists(output_file):
                os.makedirs(output_file)
            output_file = f"{output_file}/{sample.sample_id}.json"
            json_obj = defaultdict(dict)
            for i in range(len(tmp_q_id)):
                n_questions += 1
                if tmp_q_types[i] == objs.AnswerType.NUMERIC:
                    n_num += 1
                    if isinstance(tmp_predicts[i], str):
                        value = convert_num(tmp_predicts[i])
                    else:
                        value = tmp_predicts[i]
                else:
                    n_text += 1
                    value = str(tmp_predicts[i])
                json_obj[tmp_q_categories[i].value][tmp_q_id[i]] = value

            with open(output_file, "w") as f:
                json.dump(json_obj, f)

        predicts.extend(tmp_predicts)
        answers.extend(tmp_answers)
        q_types.extend(tmp_q_types)
        q_categories.extend(tmp_q_categories)
        p_bar.update()
        p_bar.set_description(f"Answered: {len(predicts)}")
    p_bar.close()

    if split != objs.DataSplit.TEST:
        score = qa.eval_results(predicts, answers, q_types, q_categories)
    else:
        print(f"{n_tables:,} tables|{n_questions:,} questions|{n_num} num|{n_text} txt")
        score = 0
    return score


def run_qa_one_sample(
    table_list,
    model_name: qa.QA_MODEL,
    device_id: int = -1,
    from_ground_truth=True,
    ago_headers: bool = True,
    ago_html: bool = True,
):
    print(model_name)
    predicts, answers, q_types, q_categories = [], [], [], []
    qa_pipeline = load_qa_pipeline(model_name, device_id)
    init_worker(model_name, qa_pipeline, device_id)
    p_bar = tqdm(total=len(table_list))
    for split, table_id, headers in table_list:
        sample = loader.load_sample_from_id(
            split,
            table_id,
            from_ground_truth=from_ground_truth,
            ago_headers=ago_headers,
            ago_html=ago_html,
        )
        sample.headers = headers
        (
            tmp_predicts,
            tmp_answers,
            tmp_q_types,
            tmp_q_categories,
            tmp_q_id,
            reload_pipeline,
        ) = run_pool_qa(sample)
        if reload_pipeline:
            qa_pipeline = load_qa_pipeline(model_name, device_id)
            init_worker(model_name, qa_pipeline, device_id)
        predicts.extend(tmp_predicts)
        answers.extend(tmp_answers)
        q_types.extend(tmp_q_types)
        q_categories.extend(tmp_q_categories)
        p_bar.update()
        p_bar.set_description(f"Answered: {len(predicts)}")
    p_bar.close()
    score = qa.eval_results(predicts, answers, q_types, q_categories)
    return score


def analysis_string(input_text, spacy_model):
    dims_text = {
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "LOC",
        "GPE",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
    }
    dims_num = {"CARDINAL", "PERCENT", "MONEY", "ORDINAL"}
    dims_time = {"QUANTITY", "TIME", "DATE"}
    doc = spacy_model(input_text)
    n_words, n_nouns, r_digits = 0, 0, 0
    r_ners, r_pos, count_words, count_nouns = Counter(), Counter(), Counter(), Counter()
    ner_dims = Counter({"txt": 0, "num": 0, "time": 0, "ent": 0})

    # NER
    for ent in doc.ents:
        r_ners[ent.label_] += 1

        ner_dims["ent"] += 1
        if ent.label_ in dims_text:
            ner_dims["txt"] += 1
        elif ent.label_ in dims_num:
            ner_dims["num"] += 1
        else:
            ner_dims["time"] += 1

    # Count
    ratio_digit, n_tokens = 0, 0
    for token in doc:
        r_pos[token.pos_] += 1
        if token.is_stop or token.is_punct or token.is_space:
            continue
        n_tokens += 1
        if token.is_digit:
            ratio_digit += 1
        count_words[token.text] += 1
        n_words += 1
        if token.pos_ == "NOUN":
            n_nouns += 1
            count_nouns[token.text] += 1
    if n_tokens:
        r_digits += ratio_digit / n_tokens
    else:
        r_digits = 0
    return (
        n_words,
        n_nouns,
        r_digits,
        r_ners,
        r_pos,
        ner_dims,
        count_words,
        count_nouns,
    )


def eda_data(split: objs.DataSplit, head: int = 0):
    spacy_model = spacy.load("en_core_web_trf")

    n_tables, n_i_tables, n_questions, non_answers = 0, 0, 0, 0
    n_q_words, n_q_nouns, n_a_words, n_a_nouns = 0, 0, 0, 0
    len_questions, len_answers = [], []
    q_ners, a_ners, q_pos, a_pos = Counter(), Counter(), Counter(), Counter()
    q_ner_dims = Counter({"txt": 0, "num": 0, "time": 0, "ent": 0})
    a_ner_dims = Counter({"txt": 0, "num": 0, "time": 0, "ent": 0})
    q_digits, a_digits = 0, 0
    count_q_words, count_q_nouns = Counter(), Counter()
    count_a_words, count_a_nouns = Counter(), Counter()

    gt_rows, gt_cols, p_rows, p_cols = [], [], [], []
    gt_cell_lens, p_headers, p_data = [], [], []
    SPLIT_DIR = f"{config.DATA_VQA}/{split.value}/ground_truth/"
    for i, file in enumerate(tqdm(io_func.get_files_from_dir(SPLIT_DIR))):
        if head and i >= head:
            break
        sample_id = os.path.splitext(os.path.basename(file))[0]
        sample = loader.load_sample_from_id(split, sample_id, from_ground_truth=True)
        n_tables += 1
        sample_infer = (
            f"{config.DATA_PUBTAB}/{split.value}/infer_html_2/{sample_id}.txt"
        )

        for q in sample.questions_answers.values():
            n_questions += 1

            len_questions.append(len(q.question))

            # Questions analysis
            (
                tmp_n_words,
                tmp_n_nouns,
                tmp_r_digits,
                tmp_r_ners,
                tmp_r_pos,
                tmp_ner_dims,
                tmp_count_words,
                tmp_count_nouns,
            ) = analysis_string(q.question, spacy_model)
            n_q_words += tmp_n_words
            n_q_nouns += tmp_n_nouns
            q_digits += tmp_r_digits
            q_ners.update(tmp_r_ners)
            q_pos.update(tmp_r_pos)
            q_ner_dims.update(tmp_ner_dims)
            count_q_words.update(tmp_count_words)
            count_q_nouns.update(tmp_count_nouns)

            # Answer analysis
            if q.answer is math.nan:
                non_answers += 1

            (
                tmp_n_words,
                tmp_n_nouns,
                tmp_r_digits,
                tmp_r_ners,
                tmp_r_pos,
                tmp_ner_dims,
                tmp_count_words,
                tmp_count_nouns,
            ) = analysis_string(str(q.answer), spacy_model)

            len_answers.append(len(str(q.answer)))
            n_a_words += tmp_n_words
            n_a_nouns += tmp_n_nouns
            a_digits += tmp_r_digits
            a_ners.update(tmp_r_ners)
            a_pos.update(tmp_r_pos)
            a_ner_dims.update(tmp_ner_dims)
            count_a_words.update(tmp_count_words)
            count_a_nouns.update(tmp_count_nouns)

        # Table analysis
        # from ground truth
        gt_rows.append(sample.n_rows)
        gt_cols.append(sample.n_cols)
        if sample.table is None:
            avg_len = 0
        else:
            table = sample.table.astype(str)
            avg_len = table.applymap(len).mean().mean()

        gt_cell_lens.append(avg_len)
        if os.path.exists(sample_infer):
            n_i_tables += 1
            sample_i = loader.load_sample_from_id(
                split, sample_id, from_ground_truth=False, ago_headers=True
            )
            table_json = sample_i.to_json()
            len_headers, len_data, n_data, n_row = 0, 0, 0, 0
            for k, v in table_json.items():
                len_headers += len(k)
                n_data += len(v)
                n_row = len(v)
                len_data += sum(len(v_i) for v_i in v)
            p_headers.append(len_headers / len(table_json))
            if n_data:
                p_data.append(len_data / n_data)
            else:
                p_data.append(0)
            p_cols.append(len(table_json))
            p_rows.append(n_row + 1)

    print(f"#Tables(GT)|{n_tables:,}")
    print(f"#Tables(Inference)|{n_i_tables:,}")
    print(f"#Questions|{n_questions}")
    print(
        f"  Len|{np.mean(len_questions):.2f}|{np.median(len_questions):,.0f}|{np.min(len_questions):,.0f}|{np.max(len_questions):,.0f}"
    )
    print(f"  #Words|{n_q_words / n_questions:.2f}")
    print(f"  #Nouns|{n_q_nouns / n_questions:.2f}")
    q_digits = q_digits / n_questions * 100
    print(f"  digits/text|{q_digits:.2f}%|{100-q_digits:.2f}%")

    print(f"  Dims")
    q_ner_dims = sorted(q_ner_dims.items(), key=lambda x: x[1], reverse=True)
    for dim, v in q_ner_dims:
        print(f"    {dim}|{v / n_questions:.2f}")

    print(f"  NER")
    q_ners = sorted(q_ners.items(), key=lambda x: x[1], reverse=True)
    for dim, v in q_ners:
        print(f"    {dim}|{v / n_questions:.2f}")

    print(f"  POS")
    q_pos = sorted(q_pos.items(), key=lambda x: x[1], reverse=True)
    for dim, v in q_pos:
        print(f"    {dim}|{v / n_questions:.2f}")

    print(f"  Most 30 common words")
    for k, v in count_q_words.most_common(30):
        print(f"    {k}|{v / n_questions:.2f}")

    print(f"  Most 30 common nouns")
    for k, v in count_q_nouns.most_common(30):
        print(f"    {k}|{v / n_questions:.2f}")

    print(f"#Answer")
    print(f"  Non_Answer:|{non_answers:,}|({non_answers / n_questions*100:.2f}%)")
    print(
        f"  Len_Avg|{np.mean(len_answers):.2f}|{np.median(len_answers):,.0f}|{np.min(len_answers):,.0f}|{np.max(len_answers):,.0f}"
    )
    print(f"  #Words|{n_a_words / n_questions:.2f}")
    print(f"  #Nouns|{n_a_nouns / n_questions:.2f}")
    a_digits = a_digits / n_questions * 100
    print(f"  digits/text|{a_digits:.2f}%|{100-a_digits:.2f}%")

    print(f"  Dims")
    a_ner_dims = sorted(a_ner_dims.items(), key=lambda x: x[1], reverse=True)
    for dim, v in a_ner_dims:
        print(f"    {dim}|{v / n_questions:.2f}")

    print(f"  NER")
    a_ners = sorted(a_ners.items(), key=lambda x: x[1], reverse=True)
    for dim, v in a_ners:
        print(f"    {dim}|{v / n_questions:.2f}")

    print(f"  POS")
    a_pos = sorted(a_pos.items(), key=lambda x: x[1], reverse=True)
    for dim, v in a_pos:
        print(f"    {dim}|{v / n_questions:.2f}")

    print(f"  Most 30 common words")
    for k, v in count_a_words.most_common(30):
        print(f"    {k}|{v / n_questions:.2f}")

    print(f"  Most 30 common nouns")
    for k, v in count_a_nouns.most_common(30):
        print(f"    {k}|{v / n_questions:.2f}")

    print("Table(GT)")
    print(
        f"  Row|{np.mean(gt_rows):.2f}|{np.median(gt_rows):.2f}|{np.min(gt_rows):.2f}|{np.max(gt_rows):.2f}"
    )
    print(
        f"  Col|{np.mean(gt_cols):.2f}|{np.median(gt_cols):.2f}|{np.min(gt_cols):.2f}|{np.max(gt_cols):.2f}"
    )
    print(
        f"  Cell_len|{np.mean(gt_cell_lens):.2f}|{np.median(gt_cell_lens):.2f}|{np.min(gt_cell_lens):.2f}|{np.max(gt_cell_lens):.2f}"
    )

    if not p_rows or not p_cols:
        return
    print("Table(Predict)")
    print(
        f"  Row|{np.mean(p_rows):.2f}|{np.median(p_rows):.2f}|{np.min(p_rows):.2f}|{np.max(p_rows):.2f}"
    )
    print(
        f"  Col|{np.mean(p_cols):.2f}|{np.median(p_cols):.2f}|{np.min(p_cols):.2f}|{np.max(p_cols):.2f}"
    )
    print(
        f"  Headers_len|{np.mean(p_headers):.2f}|{np.median(p_headers):.2f}|{np.min(p_headers):.2f}|{np.max(p_headers):.2f}"
    )
    print(
        f"  Data_len|{np.mean(p_data):.2f}|{np.median(p_data):.2f}|{np.min(p_data):.2f}|{np.max(p_data):.2f}"
    )


if __name__ == "__main__":
    # run_qa_pipeline(QA_MODEL.OMNITAB, head=100, n_process=14)

    run_qa_one_sample(
        split=objs.DataSplit.VAL,
        table_id="val_table_image_7517__GPC__2013__page_55_split_0",
        headers=[0, 1, 2],
        model_name=qa.QA_MODEL.TAPAS_WTQ,
    )
    """
    #Tables(GT)  1
#Tables(Inference)  1
#Questions  31
  Len_Avg  126.32
  Len_Mid  109
  #Words  11.94
  #Nouns  7.52
  digits/text  6.45% / 93.55%
  Dims
    ent  1.52
    time  1.00
    num  0.52
    txt  0.00
  NER
    DATE  1.00
    CARDINAL  0.52
  POS
    NOUN  7.52
    ADP  4.00
    PUNCT  3.29
    ADJ  2.45
    DET  1.71
    NUM  1.52
    VERB  1.39
    PRON  0.87
    AUX  0.87
    SPACE  0.42
    CCONJ  0.26
    SCONJ  0.16
  Most 30 common words
    income  1.03
    value  1.00
    2011  1.00
    year  0.87
    accumulated  0.65
    comprehensive  0.65
    loss  0.65
    row  0.52
    investment  0.32
    securities  0.32
    related  0.32
    tax  0.32
    balance  0.29
    interest  0.26
    rate  0.26
    swap  0.26
    cap  0.26
    agreements  0.26
    6  0.26
    9  0.26
    reclassification  0.23
    net  0.23
    dollar  0.16
    worth  0.16
    help  0.16
    benefit  0.16
    ending  0.16
    contributing  0.16
    beginning  0.13
    unrealized  0.13
  Most 30 common nouns
    income  1.03
    value  1.00
    year  0.87
    loss  0.65
    row  0.52
    investment  0.32
    securities  0.32
    tax  0.32
    balance  0.29
    interest  0.26
    rate  0.26
    swap  0.26
    cap  0.26
    agreements  0.26
    reclassification  0.23
    dollar  0.16
    benefit  0.16
    losses  0.13
    beginning  0.10
    ratio  0.10
    worth  0.03
#Answer
  Non_Answer: 0 (0.00%)
  Len_Avg  4.74
  Len_Mid  5
    """
