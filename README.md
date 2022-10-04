## NISHIKA, hate-speech-detection

###

### 確認したい事項
* BERTの##の意味 : 例えばid:5「の」とid:28444「##の」で、"##の"はどのくらいの意味感を持つのか
(格まで持つのか、助詞ですよ程度の意味なのか的な)

* このwarningsの意味 : Some weights of the model checkpoint at cl-tohoku/bert-base-japanese-whole-word-masking were not used when initializing BertModel: ['cls.predictions.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.bias']
    * 使う前に学習させろよ、ってだけかもしれん
