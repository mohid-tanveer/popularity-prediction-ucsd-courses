from typing import Any, Dict, List, Tuple
from collections import defaultdict

##################################################
# bias-only model                                #
##################################################


def getDepartmentAverages(ratingsTrain, itemToDept):
    deptRatings = defaultdict(list)
    globalSum = 0
    globalCount = 0
    for _, item, rating in ratingsTrain:
        dept = itemToDept.get(item, "UNKNOWN")
        deptRatings[dept].append(rating)
        globalSum += rating
        globalCount += 1

    deptAvgs = {}
    for dept, ratings in deptRatings.items():
        deptAvgs[dept] = sum(ratings) / len(ratings)

    globalAvg = globalSum / globalCount if globalCount > 0 else 0
    return deptAvgs, globalAvg


def alphaUpdate(ratingsTrain, alpha, betaU, betaI, itemToDept):
    # update equation for alpha (per department)
    deptResiduals = defaultdict(float)
    deptCounts = defaultdict(int)

    for user, item, rating in ratingsTrain:
        dept = itemToDept.get(item, "UNKNOWN")
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        deptResiduals[dept] += rating - (bu + bi)
        deptCounts[dept] += 1

    newAlpha = {}
    for dept in deptResiduals:
        newAlpha[dept] = deptResiduals[dept] / deptCounts[dept]

    return newAlpha


def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lambU, itemToDept, globalAlpha):
    # update equation for betaU
    newBetaU = defaultdict(float)
    for user in ratingsPerUser:
        res = 0
        for item, rating in ratingsPerUser[user]:
            dept = itemToDept.get(item, "UNKNOWN")
            a = alpha.get(dept, globalAlpha)
            bi = betaI.get(item, 0)
            res += rating - (a + bi)
        newBetaU[user] = res / (lambU + len(ratingsPerUser[user]))
    return newBetaU


def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lambI, itemToDept, globalAlpha):
    # update equation for betaI
    newBetaI = defaultdict(float)
    for item in ratingsPerItem:
        res = 0
        for user, rating in ratingsPerItem[item]:
            dept = itemToDept.get(item, "UNKNOWN")
            bu = betaU.get(user, 0)
            a = alpha.get(dept, globalAlpha)
            res += rating - (a + bu)
        newBetaI[item] = res / (lambI + len(ratingsPerItem[item]))
    return newBetaI


def msePlusReg(
    ratingsTrain, alpha, betaU, betaI, lambU, lambI, itemToDept, globalAlpha
):
    # compute the mse and the mse+regularization term
    mse = 0
    for user, item, rating in ratingsTrain:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        dept = itemToDept.get(item, "UNKNOWN")
        a = alpha.get(dept, globalAlpha)

        pred = a + bu + bi
        residual = pred - rating

        mse += residual**2

    mse /= len(ratingsTrain)

    # regularization terms
    regularizerU = sum(betaU[user] ** 2 for user in betaU)
    regularizerI = sum(betaI[item] ** 2 for item in betaI)
    regularizer = lambU * regularizerU + lambI * regularizerI

    return mse, mse + regularizer


def validMSE(ratingsValid, alpha, betaU, betaI, itemToDept, globalAlpha):
    # compute the MSE on the validation set
    mse = 0
    for user, item, rating in ratingsValid:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        dept = itemToDept.get(item, "UNKNOWN")
        a = alpha.get(dept, globalAlpha)

        pred = a + bu + bi
        mse += (pred - rating) ** 2
    mse /= len(ratingsValid)
    return mse


def myRatingModel(
    ratingsTrain,
    ratingsValid,
    ratingsPerUser,
    ratingsPerItem,
    lambU,
    lambI,
    itemToDept,
    verbose=False,
):
    # hyperparameters
    maxIter = 100
    patience = 5
    earlyStopTolerance = 5e-5

    # initialize parameters
    alpha, globalAlpha = getDepartmentAverages(ratingsTrain, itemToDept)
    betaU = defaultdict(float)
    betaI = defaultdict(float)

    bestValidMSE = float("inf")
    bestParams = None
    noImprovementCount = 0

    for i in range(maxIter):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, itemToDept)
        betaU = betaUUpdate(
            ratingsPerUser, alpha, betaU, betaI, lambU, itemToDept, globalAlpha
        )
        betaI = betaIUpdate(
            ratingsPerItem, alpha, betaU, betaI, lambI, itemToDept, globalAlpha
        )

        trainMSE, trainMSEReg = msePlusReg(
            ratingsTrain, alpha, betaU, betaI, lambU, lambI, itemToDept, globalAlpha
        )
        vMSE = validMSE(ratingsValid, alpha, betaU, betaI, itemToDept, globalAlpha)

        if verbose:
            print(
                f"Iteration {i + 1}: Training MSE = {trainMSE:.4f}, MSE+Reg = {trainMSEReg:.4f}, Valid MSE = {vMSE:.4f}"
            )

        # early stopping check
        if vMSE < bestValidMSE - earlyStopTolerance:
            bestValidMSE = vMSE
            bestParams = (alpha, dict(betaU), dict(betaI), globalAlpha)
            noImprovementCount = 0
        else:
            noImprovementCount += 1
            if noImprovementCount >= patience:
                if verbose:
                    print(f"early stopping at iteration {i + 1}")
                break

    # restore best parameters
    if bestParams:
        alpha, betaU, betaI, globalAlpha = bestParams

    return alpha, betaU, betaI, bestValidMSE, globalAlpha
