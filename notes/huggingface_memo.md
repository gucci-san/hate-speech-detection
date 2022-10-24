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

#### tokenizer
* encode_plusは文頭の[SEP], 文末の[CLS]は勝手に足してくれる