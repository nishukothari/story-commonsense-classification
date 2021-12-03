mkdir processed
mkdir processed/emotion
mkdir processed/motivation

python3 scripts/make_dataloader.py memory
python3 scripts/make_dataloader.py neural

python3 scripts/make_gendataloader.py memory
python3 scripts/make_gendataloader.py neural