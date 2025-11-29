def getGlobalAverage(trainRatings):
    # return the average rating in the training set
    return sum(trainRatings) / len(trainRatings)


def trivialValidMSE(ratingsValid, globalAverage):
    return sum((rating[2] - globalAverage) ** 2 for rating in ratingsValid) / len(
        ratingsValid
    )


def baseline():
    # a baseline model which considers global avg; user-bias (professor) and item-bias(course code)
    pass
