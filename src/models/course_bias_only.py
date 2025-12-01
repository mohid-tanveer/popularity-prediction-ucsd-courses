from typing import Any, Dict, List, Tuple
from collections import defaultdict

##################################################
# course bias-only model                                #
##################################################


def clip(x, min_val, max_val):
    return max(min_val, min(x, max_val))


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


def alphaUpdate(ratingsTrain, alpha, betaI, itemToDept):
    # update equation for alpha (per department)
    deptResiduals = defaultdict(float)
    deptCounts = defaultdict(int)

    for _, item, rating in ratingsTrain:
        dept = itemToDept.get(item, "UNKNOWN")
        bi = betaI.get(item, 0)
        deptResiduals[dept] += rating - (bi)
        deptCounts[dept] += 1

    newAlpha = {}
    for dept in deptResiduals:
        newAlpha[dept] = deptResiduals[dept] / deptCounts[dept]

    return newAlpha


def betaIUpdate(ratingsPerItem, alpha, betaI, lambI, itemToDept, globalAlpha):
    # update equation for betaI
    newBetaI = defaultdict(float)
    for item in ratingsPerItem:
        res = 0
        for _, rating in ratingsPerItem[item]:
            dept = itemToDept.get(item, "UNKNOWN")
            a = alpha.get(dept, globalAlpha)
            res += rating - (a)
        newBetaI[item] = res / (lambI + len(ratingsPerItem[item]))
    return newBetaI


def msePlusReg(ratingsTrain, alpha, betaI, lambI, itemToDept, globalAlpha):
    # compute the mse and the mse+regularization term
    mse = 0
    for user, item, rating in ratingsTrain:
        bi = betaI.get(item, 0)
        dept = itemToDept.get(item, "UNKNOWN")
        a = alpha.get(dept, globalAlpha)

        pred = a + bi
        residual = pred - rating

        mse += residual**2

    mse /= len(ratingsTrain)

    # regularization terms
    regularizerI = sum(betaI[item] ** 2 for item in betaI)
    regularizer = lambI * regularizerI

    return mse, mse + regularizer


def validMSE(ratingsValid, alpha, betaI, itemToDept, globalAlpha):
    # compute the MSE on the validation set
    mse = 0
    for user, item, rating in ratingsValid:
        bi = betaI.get(item, 0)
        dept = itemToDept.get(item, "UNKNOWN")
        a = alpha.get(dept, globalAlpha)

        pred = a + bi
        mse += (pred - rating) ** 2
    mse /= len(ratingsValid)
    return mse


def getCourseBiasOnlyPreds(ratingsTrain, alpha, betaI, itemToDept, globalAlpha):
    biasonly_preds = []
    for _, item, _ in ratingsTrain:
        bi = betaI.get(item, 0)
        dept = itemToDept.get(item, "UNKNOWN")
        a = alpha.get(dept, globalAlpha)

        pred = a + bi
        pred = clip(pred, 0, 100)
        biasonly_preds.append(pred)
    return biasonly_preds


def courseBiasOnlyModel(
    ratingsTrain,
    ratingsValid,
    ratingsPerItem,
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
    betaI = defaultdict(float)

    bestValidMSE = float("inf")
    bestParams = None
    noImprovementCount = 0

    for i in range(maxIter):
        alpha = alphaUpdate(ratingsTrain, alpha, betaI, itemToDept)
        betaI = betaIUpdate(
            ratingsPerItem, alpha, betaI, lambI, itemToDept, globalAlpha
        )

        trainMSE, trainMSEReg = msePlusReg(
            ratingsTrain, alpha, betaI, lambI, itemToDept, globalAlpha
        )
        vMSE = validMSE(ratingsValid, alpha, betaI, itemToDept, globalAlpha)

        if verbose:
            print(
                f"Iteration {i + 1}: Training MSE = {trainMSE:.4f}, MSE+Reg = {trainMSEReg:.4f}, Valid MSE = {vMSE:.4f}"
            )

        # early stopping check
        if vMSE < bestValidMSE - earlyStopTolerance:
            bestValidMSE = vMSE
            bestParams = (alpha, dict(betaI), globalAlpha)
            noImprovementCount = 0
        else:
            noImprovementCount += 1
            if noImprovementCount >= patience:
                if verbose:
                    print(f"early stopping at iteration {i + 1}")
                break

    # restore best parameters
    if bestParams:
        alpha, betaI, globalAlpha = bestParams

    return alpha, betaI, bestValidMSE, globalAlpha
