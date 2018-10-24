# doc2vec sample

## Overview
Doc2Vec で記事とwikipediaページを比較するサンプル

## Requirements
- macOS or Ubuntu
- Python 3.5
- pyenv
- pipenv
  
## Preparation
- `pipenv install` を実行
- [jhlau/doc2vec](https://github.com/jhlau/doc2vec) から学習済のDoc2Vecモデルを取得して `model` フォルダに配置する。
    - Pre-Trained Doc2Vec Models > English Wikipedia DBOW
    - `model/enwiki_dbow/doc2vec.bin` に配置する。

## Run

```sh
$ pipenv shell
$ python main.py
# ----------------------------------- #
Apple_Inc vs Google = 0.4756675958633423
Apple_Inc vs Renault = 0.41303306818008423
Alphabet Inc. vs Alphabet = 0.2532535493373871
Apple_Inc vs CNN - iPhone XR review = 0.1879728138446808
Apple_Inc dict vs CNN - iPhone XR review = 0.46645450592041016
Google vs CNN - Google should buy Twitter and Square. But it won't = 0.15044787526130676
Google dict vs CNN - Google should buy Twitter and Square. But it won't = 0.7282728552818298
Twitter vs CNN - Google should buy Twitter and Square. But it won't = 0.1996014267206192
Twitter dict vs CNN - Google should buy Twitter and Square. But it won't = 0.5841634273529053
# ----------------------------------- #
```

## refs

- [Word2Vecの進化形Doc2Vecで文章と文章の類似度を算出する](https://qiita.com/okappy/items/32a7ba7eddf8203c9fa1)

