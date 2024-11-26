# NMT For Maghrebi dialects and MSA bidirectional translation

### Dataset
- The dataset is sourced from: [Maghrebi dialects](https://github.com/laith85/Transformer_NMT_AD/blob/main/North_Africa%20_Dialect.txt) and
  [MSA](https://github.com/laith85/Transformer_NMT_AD/blob/main/MSA_For_North_Africa_Dialects.txt). The dataset should be split randomly into training, validation, and test sets.

### Installing requirements:
1) Create an anaconda environment with Python version 3.8.13

```python

conda create -n nmt_ar python=3.8.13
```
2) Activate the environment using:
```python
conda activate nmt_ar
```
3) Once your environment is activated, install pip version 22.1.12 using:
```python
pip install pip==22.1.12
```
4) Install the requirements:
```python
pip install -r requirements.txt
```
### Training:
- Training the Model (our proposed approach for Maghrebi dialects to MSA)

```python
python src/train.py
```
- Training the Model (our proposed approach for MSA to Maghrebi dialects)

```python
python src/train_inverse.py
```
**Same for src/train_enc.py, src/train_enc_inverse.py, src/train_dec.py, src/train_dec_inverse.py.**

### Generation:

Generate translation based on the saved models using generate.py file using greedy decoding methodology

- For Maghrebi dialects to MSA
```python
python src/generate.py
```

- For MSA to Maghrebi dialects
```python
python src/generate_inverse.py
```

