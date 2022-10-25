### AutoModel実装してからRoberta行こうとしてハマったのでメモ --

#### AutoModel
* out = model(ids, masks, **kwargs)
    * type(out) -> BaseModelOutputWithPoolingAndCrossAttentions
    * out.keys() -> odict keys(["last_hidden_state", "pooler_output", "attentions"])
        * サンプルではout[1]を次に回してたから、out["pooler_output"]を次に回してたことになる
    * out["pooler_output"].shape -> torch.Size([32, 768]), means torch.Size([batch_size, hidden_layer_size])
    * out["last_hidden_state"].shape -> torch.Size([32, 76, 768]), means torch.Size([batch_size, max_length, hidden_layer_size])

    * out["pooler_output"] != out["hidden_states"][-1].max(axis=1)[0]
        * pooler_outputが何してるのか問題

    * len(out["hidden_states"]) = 13, type(out["hidden_states"]) -> tuple,
        * hidden_layer_numをタプルで持って、各hidden_layerが[batch, max_length, hidden_size]を持つイメージ

#### RobertaForMaskedLM
* out = model(ids, masks, **kwargs)
    * type(out) -> MaskedLMOutput
    * out.keys() -> odict keys(["logits", "attentions"])
        * output_hidden_states=Trueで、last_hidden_statesのmax_poolingでとりあえず通せる

#### そもそも論
* AutoModel, とかがBare model
    * AutoModelFor...は特定の目的に合わせてヘッダ付きで学習してあるという感じ
    * Fine-TuneしたいときはBareModelでいいんじゃない？多分

* 学習済みモデルは大体max length=512を想定
    * reshape自体はできるらしいけど、当然未学習のパラメータ増えるから精度は落ちる(https://github.com/huggingface/transformers/issues/18506)
    * なので、512超えるならどこ取ってくるかというところに観点が一つ増える

#### tokenizer
* encode_plusは文頭の[SEP], 文末の[CLS]は勝手に足してくれる
    * add_special_token=Trueが必要