# PAME

## 1. Make the enriroment
Our code is based on  `pytorch==2.1.0+cu118` `cuda==11.8` `mamba-ssm==1.2.0.post1`. Other environments can refer to requirements.txt
```
conda create -n PAME python=3.10
git clone https://github.com/CQC-gogopro/PAME.git
cd PAME
pip install requirements.txt
```

## 2. Get data 
You can download the PASCAL-Context and NYUD-v2 from ATRC's repository:
### PASCAL-Context
```
wget https://data.vision.ee.ethz.ch/brdavid/atrc/NYUDv2.tar.gz
tar xfvz PASCALContext.tar.gz
```
### NYUD-v2
```
wget https://data.vision.ee.ethz.ch/brdavid/atrc/PASCALContext.tar.gz
tar xfvz NYUDv2.tar.gz
```

**Attention**: You need to set the ```db_root``` variable in ```./configs/mypath.py``` to the root path of all your datasets, with the DATASET_ROOT folder formatted as follows:
```
DATASET_ROOT
├── NYUDv2
└── PASCALContext
```



## 3. Evaluate the model
Our trained PAME model can be obtained [here](https://drive.google.com/file/d/1mb07S3Ox85fF0_dUwleDiN7fkjqUmdIN/view?usp=drive_link)
```
bash ./script/test_PAME_pascal.sh
```

## 4. Train the model
```
bash ./script/train_PAME_pascal.sh
```
