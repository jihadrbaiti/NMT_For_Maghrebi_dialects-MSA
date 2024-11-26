# NMT For Maghrebi dialects and MSA bidirectional translation

### Dataset
- The dataset is sourced from: [Maghrebi dialects](https://github.com/laith85/Transformer_NMT_AD/blob/main/North_Africa%20_Dialect.txt) and [MSA](https://github.com/laith85/Transformer_NMT_AD/blob/main/MSA_For_North_Africa_Dialects.txt

### Installing requirements:
-->
Create an anaconda environment with Python version 3.8.13

```python

conda create -n nmt_ar python=3.8.13
```
Activate the environment using:
```python
conda activate nmt_ar
```
Once your environment is activated, install pip version 22.1.12 using:
```python
pip install pip==22.1.12
```
Install the requirements:
```python
pip install -r requirements.txt
```
### Training:
Training the Model (our proposed approach for Maghrebi dialects to MSA)

```python
python train.py
```
Training the Model (our proposed approach for MSA to Maghrebi dialects)

```python
python train_inverse.py
```
**Same for train_enc.py, train_enc_inverse.py, train_dec.py, train_dec_inverse.py.**

### Generation:

Generate translation based on the saved models using generate.py file using greedy decoding methodology

- For Maghrebi dialects to MSA
```python
python generate.py
```

- For MSA to Maghrebi dialects
```python
python generate.py
```

