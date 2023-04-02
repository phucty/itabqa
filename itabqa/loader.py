import fnmatch
import json
import os  # folder directory navigation
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import ujson
from cv2 import Mat  # opencv
from IPython.core.display import HTML, display
from tqdm import tqdm

from itabqa import config, io_func, objs


def load_sample_from_id(
    split: objs.DataSplit,
    sample_id: str,
    from_ground_truth: bool = True,
    ago_headers: bool = False,
    ago_html: bool = True,
):
    if split is objs.DataSplit.TEST:
        return objs.SampleTest(
            split=split, sample_id=sample_id, ago_headers=ago_headers, ago_html=ago_html
        )

    if from_ground_truth:
        return objs.Sample(split=split, sample_id=sample_id, ago_headers=ago_headers)

    return objs.SampleTR(
        split=split, sample_id=sample_id, ago_headers=ago_headers, ago_html=ago_html
    )


def load_samples(
    split: objs.DataSplit,
    head: int = 0,
    from_ground_truth: bool = True,
    check_images: bool = False,
    ago_headers: bool = False,
    ago_html: bool = True,
    check_missing_inference: bool = False,
) -> List[objs.Sample]:
    samples = []
    count_miss = 0
    SPLIT_DIR = f"{config.DATA_VQA}/{split.value}/ground_truth/"
    count_images, count_groundtruths = 0, 0
    for i, file in enumerate(io_func.get_files_from_dir(SPLIT_DIR)):
        if head and i >= head:
            break

        sample_id = os.path.splitext(os.path.basename(file))[0]
        if check_missing_inference:
            sample_file = (
                f"{config.DATA_PUBTAB}/{split.value}/infer_html_2/{sample_id}.txt"
            )
            if not os.path.exists(sample_file):
                count_miss += 1
                print(f"{count_miss}. Missing inference files: {sample_file}")
            # continue

        sample = load_sample_from_id(
            split=split,
            sample_id=sample_id,
            from_ground_truth=from_ground_truth,
            ago_headers=ago_headers,
            ago_html=ago_html,
        )

        if check_images:
            has_image = os.path.exists(sample.image_dir)
            if has_image:
                count_images += 1

        has_groundtruth = os.path.exists(sample.ground_truth_dir)
        if has_groundtruth:
            count_groundtruths += 1

        # if has_image and has_groundtruth:
        if sample:
            samples.append(sample)

    print(
        f"{split.value}: {len(samples)} - {count_images}  / {count_groundtruths} (images / groundtruth)"
    )
    return samples


def generate_fine_tune_data(
    output_name: str,
    split: objs.DataSplit,
    n_samples: int = 0,
    train_ratio: float = 0.8,
    ago_html: bool = False,
    ago_headers: bool = False,
    n_qtype_per_table: int = 1,
):
    # 5 categories, sampling # questions based on 5 categories
    limit_samples = n_samples // len(objs.Category)
    print(limit_samples)
    samples = {i: [] for i in objs.Category}

    SPLIT_DIR = f"{config.DATA_VQA}/{split.value}/ground_truth/"
    q_id, count_miss = 0, 0
    for file in tqdm(io_func.get_files_from_dir(SPLIT_DIR)):
        if all(len(v) >= limit_samples for v in samples.values()):
            break

        sample_id = os.path.splitext(os.path.basename(file))[0]

        sample_file = f"{config.DATA_PUBTAB}/{split.value}/infer_html_2/{sample_id}.txt"
        if not os.path.exists(sample_file):
            count_miss += 1
            # print(f"{count_miss}. Missing inference files: {sample_file}")
            continue

        sample = load_sample_from_id(
            split=split,
            sample_id=sample_id,
            from_ground_truth=False,
            ago_html=ago_html,
            ago_headers=ago_headers,
        )
        table_json = sample.to_json()
        headers = list(table_json.keys())
        rows = [list(row) for row in zip(*table_json.values())]
        if not headers or not rows:
            continue

        questions_categories = defaultdict(list)

        for q in sample.questions_answers.values():
            questions_categories[q.category].append(q)

        for category, questions in questions_categories.items():
            if len(samples[category]) >= limit_samples:
                continue
            if len(questions) > n_qtype_per_table:
                qs = random.choices(questions, k=n_qtype_per_table)
            else:
                qs = questions
            for q in qs:
                if len(samples[category]) >= limit_samples:
                    continue
                json_sample = {
                    "id": q.question_id,
                    "question": q.question,
                    "answers": [str(q.answer)],
                    "table": {
                        "header": headers,
                        "rows": rows,
                    },
                }
                samples[q.category].append(json_sample)
            q_id += 1

    train_file = f"{config.DATA_ROOT}/train_{output_name}.json"
    val_file = f"{config.DATA_ROOT}/val_{output_name}.json"
    f_train = open(train_file, "w")
    if train_ratio < 1:
        f_val = open(val_file, "w")

    n_train, n_val = 0, 0
    for key, values in samples.items():
        print(f"{key}: {len(values)}")
        n_train_samples = len(values) * train_ratio
        for i, value in enumerate(values):
            if i < n_train_samples:
                f = f_train
                n_train += 1
            elif train_ratio < 1:
                f = f_val
                n_val += 1
            json.dump(value, f)
            f.write("\n")
    f_train.close()
    print(f"{train_file}: {n_train}")
    if train_ratio < 1:
        f_val.close()
        print(f"{val_file}: {n_val}")


def print_sample(
    sample: objs.Sample,
    show_bbox: bool = False,
    show_text: bool = False,
    show_question: bool = True,
    n_questions: int = 3,
    canonical: bool = True,
):
    display(HTML(sample.to_html(canonical=canonical)))
    for i, q in enumerate(sample.questions_answers.values()):
        if n_questions and i and i > n_questions:
            break
        print(f"?({q.category}): {q.question}")
        print(f"!({q.answer_type}): {q.answer}")
    sample.draw_image(show_bbox, show_text, show_question)
    # print(sample.to_html(canoxnical=canonical))
