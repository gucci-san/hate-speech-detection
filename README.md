## NISHIKA, hate-speech-detection

### To Do
<スクリプト化>
* CVとLBが完全一致という感じもしないのでAdversarialしてみてもいいかも
    * 意味なかったから多分データ少ないです

* pseudo-labelling + corpusから追加データセット作成
    * pseudoがsoftだとstratified-kfoldのとこ実装変えないと通せないから注意

* mdebertaは使ってみたほうがいい
    * 確かにシングルだと高めに見えます(baseサイズ比)

* lr上げると全部0になるので、思ってるより過学習気味かも

* 外部データ学習
    * Validateどうするの？
        * train_df全部を対象としてHold-Outでいい？

### 学び
* dask, pandarallelは使い得
    * 特にSeries.mapは結構使いがちだけど、pandaralellでだいぶ早くなる
* 強い人はみんなtransformerのgithubからコード引っ張って手元で書いてた
* amp早すぎ
    * roberta-largeのfold(1/5), ampなしで8min, ampありで4minくらい, 理論上も2倍くらい早いらしいので妥当
        * CV結構変わってますけど、LBは0.727->0.723, まあ誤差の範囲かなと

### 確認したい事項
* 結局BERTの中身わかってないからなあ......