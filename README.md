# TabIQA: Table Questions Answering on Business Document Images

This is the instruction on how to reproduce TabIQA experiments on [VQAonBD 2023](https://ilocr.iiit.ac.in/vqabd/index.html).

- Preliminary technical report: [https://arxiv.org/abs/2303.14935](https://arxiv.org/abs/2303.14935)
- Ranking in Test Leaderboard: [https://ilocr.iiit.ac.in/vqabd/leaderboard.html](https://ilocr.iiit.ac.in/vqabd/leaderboard.html)

    2023/04/02: the results for NII-TabIQA have been ranked second, achieving a Weighted Categories Average score of 0.901.

***


## Install
### Install itabqa
```bash
git clone https://github.com/phucty/itabqa.git
cd itabqa

conda create -n itabqa python=3.8
conda activate itabqa
pip install poetry
poetry shell
poetry install
```
### Install MTL-TabNet
```bash
git clone https://github.com/phucty/MTL-TabNet.git
```
Please follow the instruction of MTL-TabNet to install the module

### Install OmniTab
```bash
git clone https://github.com/phucty/OmniTab.git
```
Please follow the instruction of OmniTab to install the tool. You might need to install omnitab in a different conda env and different pytorch version.
***

## Config itabqa

Please setup working directory to your setting in [itabqa/config.py](itabqa/config.py) file:
- `HOME_ROOT`: itabqa project directory

    e.g,: `/home/phuc/itabqa`
- `DATA_ROOT`: store models, and dataset
    
    e.g,: `/disks/strg16-176/VQAonBD2023/data`
- `DATA_VQA`: VQAonBD 2023 dataset
    
    e.g, `{DATA_ROOT}/vqaondb2023`
- `DATA_PUBTAB`: HTML tables infered from table structure extraction, 
    
    e.g., `{DATA_ROOT}/TR_output`

***

## Table structure extraction

Please run the file [MTL-TabNet/table_inference_VQAonBD2023_inference.py](MTL-TabNet/table_inference_VQAonBD2023_inference.py)
to gerate HTML tables from document images

***
## Generate training samples for QA model
```
python run_gen_training_samples.py
```
***
## Fine-tune with OmniTab
Note: We fine-tune OmniTab on 4 A100 40GB. If you have V100 please change `per_device_train_batch_size`, and `per_device_eval_batch_size` to 6
```bash
cd OmniTab
conda activate ominitab

python -m torch.distributed.launch --nproc_per_node=4 run.py \
    --do_train \
    --train_file /disks/strg16-176/VQAonBD2023/data/train_all_rawjson \
    --validation_file /disks/strg16-176/VQAonBD2023/data/train_100_raw.json \
    --model_name_or_path neulab/omnitab-large \
    --output_dir /disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-all-raw \
    --max_source_length 1024 \
    --max_target_length 128 \
    --val_max_target_length 128 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 12 \
    --num_train_epochs 50.0 \
    --warmup_ratio 0.1 \
    --learning_rate 2e-5 \
    --fp16 \
    --logging_steps 100 \
    --eval_steps 1000000 \
    --save_steps 50000 \
    --evaluation_strategy steps \
    --predict_with_generate \
    --num_beams 5 \
    --generation_max_length 128 \
    --overwrite_output_dir
```

***
## Run QA:
The qa models are in [itabqa/qa.py](itabqa/qa.py).
After fine-tuning, like previous example, the model is in `/disks/strg16-176/VQAonBD2023/models/omnitab-large-finetuned-qa-all-raw`.
The pretrained model is [here](https://drive.google.com/file/d/1Shch5gdtjv5IGWsY0uXAJ4BNOXnJqcjS/view).
We can run QA inference as
```
cd ..
python run_qa_inference.py
```
The answers will be saved in [answers/raw_3_all](answers/raw_3_all)
***

# Cite

If you find TabIQA tool useful in your work, and you want to cite our work, please use the following referencee:
```
@article{nguyen2023tabiqa,
  title={TabIQA: Table Questions Answering on Business Document Images},
  author={Nguyen, Phuc and Ly, Nam Tuan and Takeda, Hideaki and Takasu, Atsuhiro},
  journal={arXiv preprint arXiv:2303.14935},
  year={2023}
}
```