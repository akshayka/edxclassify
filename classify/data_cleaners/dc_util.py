def compress_likert(score):
    if score <= 3:
        return 0
    elif score == 4:
        return 1
    else:
        return 2

