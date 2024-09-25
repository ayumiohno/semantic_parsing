実装
=

### 全体: 
- seq2seq_geo: Attention機構無しのgeoquery
- seq2seq_attn_geo: Attention機構ありのgeoquery
- seq2seq_attn_spider: spider
- seq2seq_attn_spider_skeleton: spiderのSQL構造だけを抽出して学習・推論
- seq2seq_attn_Resdsql_like: Resdsqlの工夫を取り入れようとした

### 各ディレクトリ内部: 
- (*_)main.py: メイン
- data.py or spider.py: データローダー
- model.py: モデル
- eval(*_).py: Accuracy評価
- result.txt: 推論結果

### データ
- data/geoquery: 
    - このコードを参考にlang2logic用にparse済みのデータを使用した. 
    - https://github.com/donglixp/lang2logic/blob/master/pull_data.py
- data/spider:
    - https://yale-lily.github.io/spider
    - train_spider.json: これをtrain, validation, testに分割した. 
    - tables.json: Resdsql_likeで使用

### 実行環境
- cscサーバを使用した.
- 以下の環境で行った. 
``` toml
[tool.poetry]
name = "seq2seq"
version = "0.1.0"
description = ""
authors = ["ayumiohno <ayumi0130ohno@gmail.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
torch = {version = "^2.0.1+cu117", source = "torch_cu117"}
matplotlib = "^3.9.0"

[[tool.poetry.source]]
name = "torch_cu117"
url = "https://download.pytorch.org/whl/cu117"
priority = "explicit"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```