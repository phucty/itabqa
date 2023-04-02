import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Counter, Dict, List

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import ujson
from bs4 import BeautifulSoup
from cv2 import Mat  # opencv

from itabqa import config


class DataSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AnswerType(str, Enum):
    NUMERIC = "numeric"
    TEXT = "text"


class Category(str, Enum):
    CATEGORY_1 = "category_1"
    CATEGORY_2 = "category_2"
    CATEGORY_3 = "category_3"
    CATEGORY_4 = "category_4"
    CATEGORY_5 = "category_5"


@dataclass
class BBox:
    bbox: List[int]
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    content: str


@dataclass
class Question:
    question: str
    question_id: str
    answer: str
    answer_type: AnswerType
    category: Category


@dataclass
class Sample:
    split: DataSplit
    sample_id: str
    table: Any = None
    table_structure: Dict[str, BBox] = None
    questions_answers: Dict[str, Question] = None
    n_rows: int = 0
    n_cols: int = 0
    headers: List[int] = None
    ago_html: bool = False
    ago_headers: bool = True

    def __post_init__(self):
        self.parse_groundtruth()
        self.cal_table_stats()
        self.table = self.to_pandas()

    @property
    def image_dir(self):
        output_path = (
            f"{config.DATA_VQA}/{self.split.value}/table_images/{self.sample_id}.png"
        )
        if not os.path.exists(output_path):
            raise IOError(f"Missing {output_path}")
        return output_path

    @property
    def ground_truth_dir(self):
        output_path = (
            f"{config.DATA_VQA}/{self.split.value}/ground_truth/{self.sample_id}.json"
        )
        if not os.path.exists(output_path):
            raise IOError(f"Missing {output_path}")
        return output_path

    def parse_groundtruth(self):
        self.table_structure = dict()
        self.questions_answers = dict()
        json_obj = None
        if not os.path.exists(self.ground_truth_dir):
            raise FileExistsError

        with open(self.ground_truth_dir, "r") as f:
            json_obj = ujson.load(f)

        if "table_structure" in json_obj:
            for bbox_i, bbox_obj in json_obj["table_structure"].items():
                self.table_structure[bbox_i] = BBox(**bbox_obj)

        if "questions_answers" in json_obj:
            question_i = 0
            for category_id, questions in json_obj["questions_answers"].items():
                for question_id, question_obj in questions.items():
                    self.questions_answers[str(question_i)] = Question(
                        question=question_obj["question"],
                        question_id=question_id,
                        answer=question_obj.get("answer"),
                        answer_type=AnswerType(question_obj["answer_type"]),
                        category=Category(category_id),
                    )
                    question_i += 1

    def cal_table_stats(self):
        for bbox in self.table_structure.values():
            self.n_cols = max(self.n_cols, bbox.end_col)
            self.n_rows = max(self.n_rows, bbox.end_row)
        self.n_cols += 1
        self.n_rows += 1

    def to_pandas(self) -> pd.DataFrame:
        pd_array = [[None for _ in range(self.n_cols)] for _ in range(self.n_rows)]

        for item in self.table_structure.values():
            for row in range(item.start_row, item.end_row + 1):
                for col in range(item.start_col, item.end_col + 1):
                    pd_array[row][col] = item.content
        df = pd.DataFrame(pd_array)
        df = df.fillna(value=np.nan)
        return df

    def to_html(self, canonical: bool = True) -> str:
        if canonical:
            pd_obj = self.to_pandas()
            return pd_obj.to_html(na_rep="")

        raise NotImplementedError
        html = "<table>\n"
        first_item = next(iter(self.table_structure.values()))
        cur_row = None
        cur_col = 0
        row_text = ""
        col_text = ""
        for item in self.table_structure.values():
            rowspan = item.end_row - item.start_row + 1
            colspan = item.end_col - item.start_col + 1

            if cur_row != item.end_row:
                # save catch
                if cur_row is not None:
                    html += row_text + col_text + "\t</tr>\n"
                row_text = f"\t</tr>\n"
                cur_row = item.end_row

            html += (
                f"\t\t<td rowspan='{rowspan}' colspan='{colspan}'>{item.content}</td>\n"
            )

        html = html[:-6] + "\n</table>"
        return html

    def to_image(
        self,
        show_bbox: bool = False,
        show_text: bool = False,
        show_question: bool = True,
    ) -> Mat:
        img = cv2.imread(self.image_dir)
        height, width, _ = img.shape
        if show_bbox or show_text:
            for bbox in self.table_structure.values():
                if show_bbox:
                    cv2.rectangle(img, bbox.bbox, color=config.COLOR_RED, thickness=4)
                if show_text:
                    loc_i = (bbox.bbox[0], bbox.bbox[3])
                    cv2.putText(
                        img,
                        bbox.content,
                        loc_i,
                        config.FONT,
                        1,
                        config.COLOR_BLUE,
                        2,
                        config.LINE,
                    )

        if show_question:
            q = next(iter(self.questions_answers.values()))

            text_width_q, text_height_q = cv2.getTextSize(
                str(q.question), config.FONT, 1, 2
            )[0]
            text_width_a, text_height_a = cv2.getTextSize(
                str(q.answer), config.FONT, 1, 2
            )[0]

            pad_top, pad_bottom, pad_text = 20, 20, 20
            n = 2  #  (question + answer)
            background_h = text_height_q + text_height_a + pad_top + pad_text * n
            background_w = width
            background = np.zeros((background_h, background_w, 3), np.uint8)
            background.fill(210)

            # Question
            x = (width - text_width_q - text_width_a) // 2  # center
            y = pad_bottom + text_height_q
            cv2.putText(
                background,
                str(q.question),
                (x, y),
                config.FONT,
                1,
                config.COLOR_RED,
                2,
                config.LINE,
            )

            # Answer
            y += pad_text + text_height_a
            cv2.putText(
                background,
                str(q.answer),
                (x, y),
                config.FONT,
                1,
                config.COLOR_BLUE,
                2,
                config.LINE,
            )
            img = np.vstack((background, img))
        return img

    def draw_image(
        self,
        show_bbox: bool = False,
        show_text: bool = False,
        show_question: bool = True,
    ):
        img = self.to_image(show_bbox, show_text, show_question)
        fig = px.imshow(img, title=f"{self.sample_id}")
        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.show()

    def to_json(self) -> dict:
        table = self.to_pandas()
        headers = self.headers
        if self.ago_headers:
            if self.headers is None:
                # Heuristic generate headers
                # If row contain nan
                headers = []
                for row_i, is_nan in enumerate(table.isna().any(axis=1)):
                    if is_nan:
                        headers.append(row_i)
                    else:
                        break
                # If not first row
                if not headers:
                    headers = [0]
            else:
                headers = self.headers.copy()

        if not headers:
            headers = [f"Column {i}" for i in range(self.n_cols)]
        else:
            headers.sort()
            df = table.loc[headers]
            table = table.loc[~table.index.isin(headers)]
            df = df.agg(lambda x: ", ".join(x.dropna())).to_frame().T
            headers = df.to_numpy()[0]
            headers = [
                header[2:] if header[:2] == ", " else header for header in headers
            ]

        table = table.astype(str)
        table = table.replace(np.nan, "", regex=True)
        table = table.replace("nan", "", regex=True)
        json_obj = {}
        for i in range(self.n_cols):
            column_name = headers[i]
            if column_name in json_obj:
                column_name += f" {i}"
            json_obj[column_name] = list(table[i])
        return json_obj


@dataclass
class SampleTR:
    split: DataSplit
    sample_id: str
    table: Any = None
    questions_answers: Dict[str, Question] = None
    n_cols: int = 0
    n_rows: int = 0
    headers: List[int] = None
    ago_html: bool = False
    ago_headers: bool = True

    def __post_init__(self):
        self.parse_infer_html()

    @property
    def image_dir(self):
        output_path = (
            f"{config.DATA_PUBTAB}/{self.split.value}/table_images/{self.sample_id}.png"
        )
        return output_path

    @property
    def ground_truth_dir(self):
        output_path = (
            f"{config.DATA_PUBTAB}/{self.split.value}/infer_html_2/{self.sample_id}.txt"
        )
        return output_path

    def post_process_table_html(self, html_table):
        # detect hierarchical
        hierarchical_flag = False
        hierarchical_td_store = ""

        table_ = BeautifulSoup(html_table, "lxml")
        tr_list = table_.find_all("tr")
        for tr in tr_list:
            th_td_list = tr.find_all("td") + tr.find_all("th")
            # check the first <td> in each row
            if (
                th_td_list
                and th_td_list[0].contents
                and (
                    (
                        th_td_list[0].has_attr("colspan")
                        and not th_td_list[0].has_attr("rowspan")
                    )
                    or (
                        th_td_list
                        and len(th_td_list) > 1
                        and len(th_td_list[1].contents) == 0
                    )
                )
            ):
                hierarchical_flag = True
                hierarchical_td_store = th_td_list[0].contents[0]
                continue

            # if the first td is rowspan, then reset hierarchical_flag
            if th_td_list and (
                th_td_list[0].has_attr("rowspan") or (len(th_td_list[0].contents) == 0)
            ):
                hierarchical_flag = False
                hierarchical_td_store = ""
                continue

            if hierarchical_flag and th_td_list and th_td_list[0].contents:
                tmp = ", ".join(
                    [th_td_list[0].contents[0].text, hierarchical_td_store.text]
                )
                th_td_list[0].string = tmp

        return table_.prettify(formatter=None)  # table_.find_all('table')[0]

    def parse_infer_html(self):
        self.table = None
        html_obj = None
        groundtruth_sample = Sample(self.split, self.sample_id)
        self.questions_answers = groundtruth_sample.questions_answers

        if not os.path.exists(self.ground_truth_dir):
            self.n_rows, self.n_cols = 0, 0
            return

        with open(self.ground_truth_dir, "r") as f:
            html_obj = f.read()

        if self.ago_html:
            html_obj = self.post_process_table_html(html_obj)

        self.table = pd.read_html(html_obj, thousands=None)[0]
        self.n_rows, self.n_cols = self.table.shape

    def to_html(self) -> str:
        return self.table.to_html(na_rep="")

    def to_json(self) -> dict:
        if not os.path.exists(self.ground_truth_dir):
            return {}

        headers = self.headers
        if self.ago_headers:
            if self.headers is None:
                # Heuristic generate headers
                # If row contain nan
                headers = []
                pre_cell_spans = False
                has_value_header = [False for _ in range(self.n_cols)]
                for row_i, is_nan in enumerate(self.table.isna().any(axis=1)):
                    if len(headers) > 3 or len(headers) >= self.n_rows - 1:
                        break
                    row_cells = self.table.iloc[row_i].replace(np.nan, "", regex=True)
                    for cell_i, cell in enumerate(row_cells):
                        if cell:
                            has_value_header[cell_i] = True

                    counter = Counter(row_cells)
                    n_duplicates = sum(
                        1 for k, v in counter.items() if v > 1 and k and len(str(k)) > 1
                    )
                    if n_duplicates:
                        if n_duplicates == 1:
                            pre_cell_spans = True
                        headers.append(row_i)
                    else:
                        if pre_cell_spans:
                            pre_cell_spans = False
                            headers.append(row_i)
                        else:
                            if is_nan and not all(has_value_header):
                                headers.append(row_i)
                            else:
                                break
                # If length of cells < 0.05 and cells > 0.95 cells
                # headers_len = []
                # row_lengths = self.table.apply(
                #     lambda x: sum([len(str(i)) for i in x]), axis=1
                # )
                # q1, q3 = np.quantile(row_lengths, [0.005, 0.995])
                # filtered_rows = (row_lengths > q3) | (row_lengths < q1)

                # for i, is_header in enumerate(filtered_rows.to_list()):
                #     if not is_header:
                #         break
                #     headers_len.append(i)
                # if headers_len > headers and len(filtered_rows) > 3:
                #     headers = headers_len

                # If not first row
                if not headers:
                    headers = [0]
            else:
                headers = self.headers.copy()

        table = self.table
        table = table.astype(str)
        table = table.replace(np.nan, "", regex=True)
        table = table.replace("nan", "", regex=True)
        if not headers:
            headers = [f"Column {i}" for i in range(self.n_cols)]
        else:
            headers.sort()
            df = table.loc[headers]
            df = df.loc[::-1]
            table = table.loc[~table.index.isin(headers)]

            df = df.agg(lambda x: ", ".join(x.dropna())).to_frame().T
            headers = df.to_numpy()[0]
            headers = [
                header[2:] if header[:2] == ", " else header for header in headers
            ]
        json_obj = {}
        for i in range(self.n_cols):
            column_name = headers[i]
            if column_name in json_obj:
                column_name += f" "
            json_obj[column_name] = list(table[i])
        return json_obj


@dataclass
class SampleTest(SampleTR):
    def parse_infer_html(self):
        self.table = None
        html_obj = None

        self.questions_answers = dict()
        json_obj = None
        # if not os.path.exists(self.ground_truth_dir):
        #     raise FileExistsError

        ground_truth_dir = (
            f"{config.DATA_VQA}/{self.split.value}/ground_truth/{self.sample_id}.json"
        )
        with open(ground_truth_dir, "r") as f:
            json_obj = ujson.load(f)

            question_i = 0
            for category_id, questions in json_obj.items():
                for question_id, question_obj in questions.items():
                    self.questions_answers[str(question_i)] = Question(
                        question=question_obj["question"],
                        answer=question_obj.get("answer"),
                        question_id=question_id,
                        answer_type=AnswerType(question_obj["answer_type"]),
                        category=Category(category_id),
                    )
                    question_i += 1

        if not os.path.exists(self.ground_truth_dir):
            self.n_rows, self.n_cols = 0, 0
            return

        with open(self.ground_truth_dir, "r") as f:
            html_obj = f.read()

        if self.ago_html:
            html_obj = self.post_process_table_html(html_obj)

        self.table = pd.read_html(html_obj, thousands=None)[0]
        self.n_rows, self.n_cols = self.table.shape
