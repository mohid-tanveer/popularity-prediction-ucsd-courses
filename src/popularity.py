from typing import Any

from collections import defaultdict

# TODO: this is boilerplate for my rating prediction from assignment 1, so we should try and repurpose it for popularity prediction


##################################################
# rating prediction                              #
##################################################


def alphaUpdate(ratingsTrain, alpha, betaU, betaI):
    # update equation for alpha
    newAlpha = 0
    for user, item, rating in ratingsTrain:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        newAlpha += rating - (bu + bi)
    return newAlpha / len(ratingsTrain)


def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lambU):
    # update equation for betaU
    newBetaU = defaultdict[Any, float](float)
    for user in ratingsPerUser:
        newBetaU[user] = 0
        for item, rating in ratingsPerUser[user]:
            bi = betaI.get(item, 0)
            newBetaU[user] += rating - (alpha + bi)
        newBetaU[user] /= lambU + len(ratingsPerUser[user])
    return newBetaU


def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lambI):
    # update equation for betaI
    newBetaI = defaultdict[Any, float](float)
    for item in ratingsPerItem:
        newBetaI[item] = 0
        for user, rating in ratingsPerItem[item]:
            bu = betaU.get(user, 0)
            newBetaI[item] += rating - (alpha + bu)
        newBetaI[item] /= lambI + len(ratingsPerItem[item])
    return newBetaI


def msePlusReg(ratingsTrain, alpha, betaU, betaI, lambU, lambI):
    # compute the mse and the mse+regularization term
    mse = 0
    for user, item, rating in ratingsTrain:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        pred = alpha + bu + bi
        residual = pred - rating

        mse += residual**2

    mse /= len(ratingsTrain)

    # regularization terms
    regularizerU = sum(betaU[user] ** 2 for user in betaU)
    regularizerI = sum(betaI[item] ** 2 for item in betaI)
    regularizer = lambU * regularizerU + lambI * regularizerI

    return mse, mse + regularizer


def validMSE(ratingsValid, alpha, betaU, betaI):
    # compute the MSE on the validation set
    mse = 0
    for user, item, rating in ratingsValid:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        pred = alpha + bu + bi
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
):
    # hyperparameters
    maxIter = 100
    patience = 5
    earlyStopTolerance = 5e-5

    # initialize parameters
    alpha = getDepartmentAverage([r for _, _, r in ratingsTrain])
    betaU = defaultdict[Any, float](float)
    betaI = defaultdict[Any, float](float)

    bestValidMSE = float("inf")
    bestParams = None
    noImprovementCount = 0

    for i in range(maxIter):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lambU)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lambI)

        trainMSE, trainMSEReg = msePlusReg(
            ratingsTrain,
            alpha,
            betaU,
            betaI,
            lambU,
            lambI,
        )
        vMSE = validMSE(ratingsValid, alpha, betaU, betaI)

        print(
            f"Iteration {i + 1}: Training MSE = {trainMSE:.4f}, MSE+Reg = {trainMSEReg:.4f}, Valid MSE = {vMSE:.4f}"
        )

        # early stopping check
        if vMSE < bestValidMSE - earlyStopTolerance:
            bestValidMSE = vMSE
            bestParams = (alpha, dict(betaU), dict(betaI))
            noImprovementCount = 0
        else:
            noImprovementCount += 1
            if noImprovementCount >= patience:
                print(f"early stopping at iteration {i + 1}")
                break

    # restore best parameters
    if bestParams:
        alpha, betaU, betaI = bestParams

    return alpha, betaU, betaI, bestValidMSEs
