def globalAverage(ratings):
    return sum(rating for _, _, rating in ratings) / len(ratings)


def baselineMSE(ratings_train, ratings_valid):
    # return mse when predicting the global average everywhere
    global_avg = globalAverage(ratings_train)

    mse = sum((global_avg - rating) ** 2 for _, _, rating in ratings_valid) / len(
        ratings_valid
    )

    return mse, global_avg
