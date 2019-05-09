def recover_label(
    pred_variable, gold_variable, mask_variable, label_alphabet, word_recover
):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]

    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [
            label_alphabet.get_instance(pred_tag[idx][idy])
            for idy in range(seq_len)
            if mask[idx][idy] != 0
        ]
        gold = [
            label_alphabet.get_instance(gold_tag[idx][idy])
            for idy in range(seq_len)
            if mask[idx][idy] != 0
        ]
        assert len(pred) == len(gold)
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [
                label_alphabet.get_instance(pred_tag[idx][idy][idz])
                for idy in range(seq_len)
                if mask[idx][idy] != 0
            ]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label
