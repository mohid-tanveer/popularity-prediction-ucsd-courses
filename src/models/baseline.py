from collections import defaultdict


def globalAverage(ratings):
    return sum(rating for _, _, rating in ratings) / len(ratings)


def globalAverageBaselineMSE(ratings_train, ratings_valid):
    # return mse when predicting the global average everywhere
    global_avg = globalAverage(ratings_train)

    mse = sum((global_avg - rating) ** 2 for _, _, rating in ratings_valid) / len(
        ratings_valid
    )

    return mse, global_avg


def getDepartmentAverages(ratingsTrain, itemToDept):
    deptRatings = defaultdict(list)
    for _, item, rating in ratingsTrain:
        dept = itemToDept.get(item, "UNKNOWN")
        deptRatings[dept].append(rating)

    deptAvgs = {}
    for dept, ratings in deptRatings.items():
        deptAvgs[dept] = sum(ratings) / len(ratings)

    return deptAvgs


def departmentAverageBaselineMSE(ratings_train, ratings_valid, itemToDept):
    department_avgs = getDepartmentAverages(ratings_train, itemToDept)
    mse = sum(
        (department_avgs[itemToDept.get(item, "UNKNOWN")] - rating) ** 2
        for _, item, rating in ratings_valid
    ) / len(ratings_valid)
    return mse
