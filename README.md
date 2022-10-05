## NISHIKA, hate-speech-detection

### やりたいこと
* ログ出力の機能を追加
* 実験管理（csvにsettings, remark ,scoreを残していく感じのやつ）を追加 --
* 最終的にはコマンドライン引数で勝手に実験を進めていく感じにしたい


### メモ
* ルートのconfig.pyはこの問題の間基本的に変わらなそうな要素を入れる
    * rawデータのパス
    * グラフのカラー指定 etc...

### 確認したい事項
* BERTの##の意味 : 例えばid:5「の」とid:28444「##の」で、"##の"はどのくらいの意味感を持つのか
(格まで持つのか、助詞ですよ程度の意味なのか的な)

* このwarningsの意味 : Some weights of the model checkpoint at cl-tohoku/bert-base-japanese-whole-word-masking were not used when initializing BertModel: ['cls.predictions.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.bias']
    * 使う前に学習させろよ、ってだけかもしれん
