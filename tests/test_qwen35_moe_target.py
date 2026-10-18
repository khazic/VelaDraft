

def test_turn_start_positions_follow_im_start_markers():
    from vlm_dflash.target.sample_capture import _turn_start_positions

    class _Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return 7 if token == "<|im_start|>" else 0

    processor = SimpleNamespace(tokenizer=_Tok())
    ids = [7, 1, 2, 7, 3, 7, 4, 5, 7, 6]
    assert _turn_start_positions(processor, ids, 4) == [0, 3, 5, 8]
    assert _turn_start_positions(processor, ids, 3) is None  # marker count must match the turns
    assert _turn_start_positions(SimpleNamespace(tokenizer=None), ids, 4) is None
