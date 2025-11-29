def compute_global_average(ratings):
    # avoid div by zero when we have no ratings
    if not ratings:
        return 0.0
    return sum(rating for _, _, rating in ratings) / len(ratings)


def global_average_mse(ratings_train, ratings_valid):
    # return mse when predicting the global average everywhere
    global_avg = compute_global_average(ratings_train)

    if not ratings_valid:
        return 0.0, global_avg

    mse = sum((global_avg - rating) ** 2 for _, _, rating in ratings_valid) / len(
        ratings_valid
    )

    return mse, global_avg
