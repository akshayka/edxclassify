def compress_likert(score, binary=False, bin_threshold=4):
    if binary:
        if score <= bin_threshold:
            return 0
        else:
            return 1

    if score <= 3:
        return 0
    elif score == 4:
        return 1
    else:
        return 2
