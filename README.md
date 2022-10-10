## NISHIKA, hate-speech-detection

### やりたいこと
<スクリプト化>
* CVとLBが完全一致という感じもしないのでAdversarialしてみてもいいかも
* pseudo-labelling + corpusから追加データセット作成
    * pseudoがsoftだとstratified-kfoldのとこ実装変えないと通せない --

### To Do
* mdebertaは使ってみたほうがいい
* lr上げると全部0になるので、思ってるより過学習気味かも

### 学び
* dask, pandarallelは使い得
    * 特にSeries.mapは結構使いがちだけど、pandaralellでだいぶ早くなる
* 強い人はみんなtransformerのgithubからコード引っ張って手元で書いてた

### 確認したい事項
* 結局BERTの中身わかってないからなあ......