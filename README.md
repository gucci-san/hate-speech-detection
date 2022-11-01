## NISHIKA, hate-speech-detection

### To Do
<スクリプト化>
* CVとLBが完全一致という感じもしないのでAdversarialしてみてもいいかも
    * 意味なかったから多分データ少ないです
        * それか文章的には本当に一緒

* pseudo-labelling + corpusから追加データセット作成
    * pseudoがsoftだとstratified-kfoldのとこ実装変えないと通せないから注意

* mdebertaは使ってみたほうがいい
    * 確かにシングルだと高めに見えます(baseサイズ比)

* lr上げると全部0になるので、思ってるより過学習気味かも

* 外部データ学習
    * Validateどうするの？
        * train_df全部を対象としてHold-Outでいい？

### 学び
#### コーディング面
* dask, pandarallelは使い得
    * 特にSeries.mapは結構使いがちだけど、pandaralellでだいぶ早くなる
* 強い人はみんなtransformerのgithubからコード引っ張って手元で書いてた
* amp早すぎ
    * roberta-largeのfold(1/5), ampなしで8min, ampありで4minくらい, 理論上も2倍くらい早いらしいので妥当
        * CV結構変わってますけど、LBは0.727->0.723, まあ誤差の範囲かなと
            * ampするためにlossを変えてる。「numerically more stable」ってことは計算結構変わってるはず
    * mdeberta系はfp32でfixしないと動かないっぽい
        * with autocast(enabled=False)にしてもダメっぽいので、思ってる実装になってない説あるな...
            * 素直に分岐させたほうがいい説あります

* pseudo-labelで、hard-labelならアンサンブルしなくてよくない？という観点
    * そもそもアンサンブルの有無で結果変わるようなやつを使わんでほしいという観点
* 中間モデルを使わないようにすれば容量節約＋再現性担保できる
    * corpusデータを使う時、どのデータを使うかのindexリストだけ保管しておく、とか

#### 解析面
* train_batch, fold数, NLPならpretrained-modelどれがいいかとかは最初に機械的に検証してデータ取っておくべき

* Trust Public-LBのケースもある
    * https://blog.recruit.co.jp/data/articles/kaggle-chaii/
        * 「訓練データ数が少なく、手元で検証に用いることができるデータが少ないため」
        * まさに今では？

### 確認したい事項
* 結局BERTの中身わかってないからなあ......
* AutoModelとAutoModelForMaskedLMの違い

### Papers --
* https://arxiv.org/pdf/1910.12574.pdf
    * Mozafari et al.,(2019)
    * 英語の3クラス分類、ヘイトスピーチ系
    * BERT+4つのFine-Tuning構造
        * 1, last_hidden_stateの[CLS]直下だけを直接Softmax
        * 2, last_hidden_stateの[CLS]直下 -> nn.Linear(768, 768)-LeakyReLU(negative_slope=0.01) を2回 -> Softmax
        * 3, RNN -> Softmax
        * 4, 