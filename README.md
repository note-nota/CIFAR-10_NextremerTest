# CIFAR-10 classification test

## Environment
* MacBook Pro (13-inch, 2018)
  - 2.7 GHz クアッドコアIntel Core i7
  - メモリ 16 GB
  - Python 3.7.6
  - pip 20.1.1
  - tensorflow 2.2.0
  - mlflow 1.9.1
  - keras 2.4.3
* Ubuntu 18.04 LTS
  - 4.2 GHz x 8 Intel Core i7
  - GeForce GTX 1070
  - メモリ 16 GB
  - Python 3.6.9
  - pip 20.1.1
  - tensorflow 2.2.0
  - mlflow 1.9.1
  - keras 2.4.3

## Dataset
* [The CIFAR-10 dataset](https://www.cs.toronto.edu/%7Ekriz/cifar.html)
  - 80 million tiny imagesのサブセット
  - クラスラベル:10 クラス
    - [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
    - クラスラベルは排他的
  - 全部で 60000 枚
    - 訓練画像:50000 枚（各クラス 5000 枚）
    - テスト画像:10000 枚（各クラス 1000 枚）
  - 画像サイズは 32 ピクセル x 32 ピクセル
  - RGB の 3 チャンネルカラー画像
  - Python の cPickle 形式で提供

ここでは、tensorflow.keras を用いて実装したため、`tf.keras.datasets.cifar10` からデータセットを取得した。

## Files
* dataset_cifar10.py
  - detaset の取得・加工
* check_cifar10.py
  - データ確認用
  - cifar10_image_train.png:画像出力
* model_cifar10.py
  - ML Model の定義
* train_cifar10.py
  - 学習
  - checkpoints/:学習済みモデル出力先
* test_cifar10.py
  - 学習済みモデルによるテスト
* mlruns/
  - Tracking 用ファイル群
* checkpoints/
  - 学習済みモデルの格納フォルダ
  - ここでは、精度を問わないので学習の最終結果のみを保存


## Requirement
* python
* tensorflow
* keras
* mlflow
*

## Check
```
$ python3 check_cifar10.py
```

画像ファイル (cifar10_image_train.png) の出力を行う。Train の画像データとラベルの確認用。

## Train
```
$ python3 train_cifar10.py

[ options ]
--epochs=10 (default:5)
--batch_size=32 (default:16)
```

学習後、Train と Validation による Loss と Accuracy のそれぞれのプロットを画面出力する。学習済みモデルとして `checkpoints` 配下にモデルの保存する。また、パラメータ及びエポックごとのメトリックス、モデルデータなどは MLFlow Tracking にて `mlruns/1/` 配下に保存している。

## Test
```
$ python3 test_cifar10.py
```

`checkpoint_path = "./checkpoints/ckp"` 配下の Train で保存した学習事みモデルをロードしてテストデータによる精度検証を行う。

## Model comparison
```
$ mlflow ui
```

MLFlow Server (default: http://127.0.0.1:5000) が立つ。これまでのモデル及び結果の比較検討を行う。

動作確認段階で Experiments/test/ 配下に実験結果を格納している。

## References
* [[Doc] tensorflow](https://www.tensorflow.org/install)
* [[Doc] MLFlow](https://mlflow.org/docs/latest/index.html)
