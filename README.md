# story-commonsense-classification

Steps to Reproduce our Results:
1. Set up your virtual environment on your great-lakes machine and install the requisite packages:
```
git clone https://github.com/nishukothari/story-commonsense-classification
cd story-commonsense-classification
virtualenv venv
module load python/3.9.7
source venv/bin/activate
pip3 install pandas numpy tensorflow tensorflow_hub tensorflow_text sklearn bert-tensorflow==1.0.1 sentencepiece torch tqdm pickle tabulate
```
2. Decide what your task is and submit the requisite SLURM Job:  
a) If you want to use our pretrained models, then first download the trained models from:  
https://drive.google.com/file/d/13XHTlN4E8Gd9A9I1LXel1xxytLsRKWTi/view?usp=sharing  
Then unzip the file and move the contents of the unzipped models folder. That would look something like this:
```
cd story-commonsense-classification
tar -xvzf models.tar.gz
rm models.tar.gz
mv models/* .
rm -rf models
```
Lastly, submit a Job as follows:
```
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0-00:50:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=16GB

pushd ~/story-commonsense-classification


module load python/3.9.7 cuda/11.2.1 cudnn/11.2-v8.1.0
source venv/bin/activate


python3 test.py
```

b) If you want to train and test your own models, then submit a Job as follows:
```
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0-03:45:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=16GB

pushd ~/story-commonsense-classification


module load python/3.9.7 cuda/11.2.1 cudnn/11.2-v8.1.0
source venv/bin/activate

python3 train.py
python3 test.py
```
