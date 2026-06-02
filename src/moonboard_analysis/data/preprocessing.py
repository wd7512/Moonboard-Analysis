import pandas as pd

NECESSARY_COLUMNS = ["Method", "Grade", "Name", "Rating", "Repeats", "Moves"]

GRADES_TO_DROP = ["8A+", "8B", "8B+"]


def get_sorted_descriptions_from_dict(moves: list, grade: str) -> list[list[str]]:
    start = [item["Description"] for item in moves if item["IsStart"]]
    middle = [
        item["Description"]
        for item in moves
        if not item["IsStart"] and not item["IsEnd"]
    ]
    middle.sort(key=lambda x: int(x[1:]))
    end = [item["Description"] for item in moves if item["IsEnd"]]

    if grade == "6B" or len(middle) < 4:
        return [
            start
            + ["START_END"]
            + middle
            + ["MIDDLE_END"]
            + end
            + ["END_ROUTE"]
            + [grade]
            + ["GRADE_END"]
        ]

    out = [
        start
        + ["START_END"]
        + middle
        + ["MIDDLE_END"]
        + end
        + ["END_ROUTE"]
        + [grade]
        + ["GRADE_END"]
    ]
    middle[0], middle[1] = middle[1], middle[0]
    out.append(
        start
        + ["START_END"]
        + middle
        + ["MIDDLE_END"]
        + end
        + ["END_ROUTE"]
        + [grade]
        + ["GRADE_END"]
    )
    middle[0], middle[1] = middle[1], middle[0]
    middle[-1], middle[-2] = middle[-2], middle[-1]
    out.append(
        start
        + ["START_END"]
        + middle
        + ["MIDDLE_END"]
        + end
        + ["END_ROUTE"]
        + [grade]
        + ["GRADE_END"]
    )
    middle[0], middle[1] = middle[1], middle[0]
    out.append(
        start
        + ["START_END"]
        + middle
        + ["MIDDLE_END"]
        + end
        + ["END_ROUTE"]
        + [grade]
        + ["GRADE_END"]
    )
    return out


def preprocess_lstm_data(df: pd.DataFrame, augment: bool = True) -> list:
    """Convert raw DataFrame into tokenised route sequences.

    Args:
        df: Raw DataFrame with route data.
        augment: If True (default), apply hold-swap augmentation to create up to 4
            sequences per route. If False, only return the original (unswapped)
            sequence to prevent data leakage across CV folds.
    """
    df = df[NECESSARY_COLUMNS]
    for grade in GRADES_TO_DROP:
        df = df[df["Grade"] != grade]
    df = df.reset_index(drop=True)

    X, y = df["Moves"], df["Grade"]
    data: list = []
    for i in range(len(X)):
        seqs = get_sorted_descriptions_from_dict(X[i], y[i])
        if augment:
            data += seqs
        else:
            data.append(seqs[0])
    return data


def drop_duplicate_sequences(sequences: list[list]) -> list[list]:
    seen: set = set()
    unique: list[list] = []
    for seq in sequences:
        t = tuple(seq)
        if t not in seen:
            seen.add(t)
            unique.append(seq)
    return unique
