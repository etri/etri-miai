from espnet.nets.pytorch_backend.nets_utils import th_accuracy

from sklearn.metrics import f1_score


def get_f1_score(pad_outputs, pad_targets, ignore_label, label_list):
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)
    ).argmax(2)
    mask = pad_targets != ignore_label

    pad_pred_flatten = pad_pred.masked_select(mask)
    pad_target_flatten = pad_targets.masked_select(mask)

    # tensor to numpy
    if pad_pred_flatten.is_cuda:
        pad_pred_flatten = pad_pred_flatten.cpu().numpy()
        pad_target_flatten = pad_target_flatten.cpu().numpy()
    else:
        pad_pred_flatten = pad_pred_flatten.numpy()
        pad_target_flatten = pad_target_flatten.numpy()

    each_f1 = f1_score(
        pad_target_flatten,
        pad_pred_flatten,
        average=None,
        labels=label_list,
        zero_division=0.0,
    )
    macro_f1 = f1_score(
        pad_target_flatten,
        pad_pred_flatten,
        average="macro",
        labels=label_list,
        zero_division=0.0,
    )
    weighted_f1 = f1_score(
        pad_target_flatten,
        pad_pred_flatten,
        average="weighted",
        labels=label_list,
        zero_division=0.0,
    )

    return each_f1, weighted_f1, macro_f1


def initialize_stats():
    return {
        "acc_scores": 0.0,
        "weighted_f1_scores": 0.0,
        "macro_f1_scores": 0.0,
    }


def update_stats(stats, f1_scores, additional_keys):
    for index, key in enumerate(additional_keys):
        stats[key] = f1_scores[index]


def process_labels(flatten_logits, full_targets, label_list, ignore_id):
    stats = initialize_stats()
    num_labels = len(label_list)

    additional_keys = [f"{key.lower()}_f1_scores" for key in label_list]

    acc_scores = th_accuracy(flatten_logits, full_targets, ignore_label=ignore_id)
    each_f1_scores, weighted_f1_scores, macro_f1_scores = get_f1_score(
        flatten_logits,
        full_targets,
        ignore_label=ignore_id,
        label_list=list(range(num_labels)),
    )

    stats["acc_scores"] = acc_scores
    stats["weighted_f1_scores"] = weighted_f1_scores
    stats["macro_f1_scores"] = macro_f1_scores

    update_stats(stats, each_f1_scores, additional_keys)

    return stats


def get_stats(flatten_logits, full_targets, label_list, ignore_id):
    stats = process_labels(flatten_logits, full_targets, label_list, ignore_id)

    return stats
