from cProfile import run
from typing import List

import click

import run_exp
from itabqa import objs, qa


@click.group()
def cli():
    pass


@click.command()
@click.option("--split", default="val")
@click.option("--head", default=0)
@click.option("--from_ground_truth", default=False)
@click.option("--qa_models", default=qa.QA_MODEL.OMNITAB_FT_RAW_1)
@click.option("--device_id", default=0)
@click.option("--ago_headers", default=False)
@click.option("--ago_html", default=False)
@click.option("--setting", default="temp")
def run_qa(
    split: objs.DataSplit,
    head: int = 0,
    from_ground_truth: bool = True,
    qa_models: List[qa.QA_MODEL] = None,
    device_id: int = 0,
    ago_headers: bool = False,
    ago_html: bool = True,
    setting: str = "temp",
):
    return run_exp.run_eval_split(
        split,
        head,
        from_ground_truth,
        qa_models,
        device_id,
        ago_headers,
        ago_html,
        setting,
    )


cli.add_command(run_qa)

if __name__ == "__main__":
    cli()
