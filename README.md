# NMT For Maghrebi dialects and MSA
<h1> Requirements </h1>
<!--
<ol>
<ul> python 3.8.13 <br/></ul>
<ul>pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html <br/></ul>
<ul>pip install torchtext==0.10.0 <br/></ul>
<ul>pip install spacy==3.3.1 <br/></ul>
<ul>pip install pyrsistent <br/></ul>
<ul>pip install arabert <br/></ul>
<ul>pip install nltk <br/></ul>
<ul>python -m spacy download de_core_news_sm <br/></ul>
<ul>python -m spacy download en_core_web_sm <br/>
</ul>
<ul>pip install transformers==4.29.2</ul>
<ul>pip install safetensors==0.3.0</ul>
</ol>
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
Training the Model (our proposed approach for Maghrebi dialects to MSA)

```python
python train.py
```
### Training
Training the Model (our proposed approach for MSA to Maghrebi dialects)

```python
python train_inverse.py
```
Same for train_enc.py, train_enc_inverse.py, train_dec.py, train_dec_inverse.py.

### Generation

Generate translation based on the saved models using generate.py file using greedy decoding methodology

- For Maghrebi dialects to MSA
```python
python generate.py
```

- For MSA to Maghrebi dialects
```python
python generate.py
```

