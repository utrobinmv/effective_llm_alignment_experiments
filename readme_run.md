su app

pip install poetry

pytho3 -m poetry install

python3 -m poetry run pip install -r requirements_add.txt

PYTHONPATH="${PYTHONPATH}:src/" python3 -m poetry run accelerate launch --config_file accelerate/single_gpu_config.yaml scripts/model_training/sft.py training_configs/sft/sft-yandex-lora-GrandmasterRAG_jf.yaml

PYTHONPATH="${PYTHONPATH}:src/" accelerate launch --config_file accelerate/single_gpu_config.yaml scripts/model_training/sft.py training_configs/sft/sft-yandex-lora-GrandmasterRAG_jf.yaml

