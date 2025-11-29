from typing import Any
import gzip

from collections import defaultdict
import random

# TODO: this is boilerplate for my rating prediction from assignment 1, so we should try and repurpose it for popularity prediction


def readGz(path):
    for l in gzip.open(path, "rt"):
        yield eval(l)


def readCSV(path):
    f = gzip.open(path, "rt")
    f.readline()
    for l in f:
        u, b, r = l.strip().split(",")
        r = int(r)
        yield u, b, r


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer / denom
    return 0


def clip(x, min_val, max_val):
    return max(min_val, min(x, max_val))


##################################################
# rating prediction                              #
##################################################


def getGlobalAverage(trainRatings):
    # return the average rating in the training set
    return sum(trainRatings) / len(trainRatings)


def trivialValidMSE(ratingsValid, globalAverage):
    return sum((rating[2] - globalAverage) ** 2 for rating in ratingsValid) / len(
        ratingsValid
    )


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
    alpha = getGlobalAverage([r for _, _, r in ratingsTrain])
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

    return alpha, betaU, betaI, bestValidMSE


def writePredictionsRating(alpha, betaU, betaI):
    # write predictions to file for submission
    predictions = open("predictions_Rating.csv", "w")
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(",")
        bu = betaU.get(u, 0)
        bi = betaI.get(b, 0)
        pred = clip(alpha + bu + bi, 1, 5)
        _ = predictions.write(u + "," + b + "," + str(pred) + "\n")

    predictions.close()


##################################################
# runners #
##################################################


##################################################
# rating prediction runner                       #
##################################################


def runRatingBaselinePrediction(ratingsTrain, ratingsValid):
    """
    baseline rating prediction using user averages (from baselines.py)
    """
    allRatings = []
    userRatings = defaultdict(list)

    for user, book, rating in ratingsTrain:
        allRatings.append(rating)
        userRatings[user].append(rating)

    globalAverage = sum(allRatings) / len(allRatings)
    userAverage = {}
    for u in userRatings:
        userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

    # compute mse on validation set
    mse = 0
    for user, book, rating in ratingsValid:
        if user in userAverage:
            pred = userAverage[user]
        else:
            pred = globalAverage
        mse += (pred - rating) ** 2
    mse /= len(ratingsValid)

    return mse


def runRatingPrediction():
    """
    train and evaluate rating prediction model
    """
    # hyperparameters
    lambdaU = 3.5
    lambdaI = 16
    validationSplit = 0.1
    finalTrainingRounds = 10

    # file paths
    trainPath = "train_Interactions.csv.gz"

    # load all ratings
    allRatings = []
    for user, book, rating in readCSV(trainPath):
        allRatings.append((user, book, rating))

    # split into train/validation
    random.seed(42)
    random.shuffle(allRatings)
    split_idx = int(len(allRatings) * (1 - validationSplit))
    ratingsTrain = allRatings[:split_idx]
    ratingsValid = allRatings[split_idx:]

    # build per-user and per-item dictionaries
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)

    for user, item, rating in ratingsTrain:
        ratingsPerUser[user].append((item, rating))
        ratingsPerItem[item].append((user, rating))

    print(
        f"\ntraining on {len(ratingsTrain)} ratings, validating on {len(ratingsValid)} ratings"
    )
    print(f"unique users: {len(ratingsPerUser)}, unique items: {len(ratingsPerItem)}")

    # baseline model
    print("\n" + "=" * 60)
    print("BASELINE MODEL")
    print("=" * 60)
    baseline_MSE = runRatingBaselinePrediction(ratingsTrain, ratingsValid)
    print(f"baseline validation MSE: {baseline_MSE:.4f}")

    # train model with validation
    print("\n" + "=" * 60)
    print("MY MODEL (Bias-Only Model)")
    print(f"lambdaU: {lambdaU}, lambdaI: {lambdaI}")
    print("=" * 60)

    alpha, betaU, betaI, my_MSE = myRatingModel(
        ratingsTrain,
        ratingsValid,
        ratingsPerUser,
        ratingsPerItem,
        lambdaU,
        lambdaI,
    )

    print(f"\nfinal validation MSE: {my_MSE:.4f}")

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"baseline MSE: {baseline_MSE:.4f}")
    print(f"my MSE:       {my_MSE:.4f}")
    improvement = baseline_MSE - my_MSE
    improvement_pct = (improvement / baseline_MSE) * 100
    print(f"improvement:  {improvement:+.4f} ({improvement_pct:+.2f}%)")

    # retrain on all data for final rounds
    print("\n" + "=" * 60)
    print(f"FINAL TRAINING: {finalTrainingRounds} rounds on all data")
    print("=" * 60)

    # rebuild per-user and per-item dictionaries with all data
    ratingsPerUserAll = defaultdict(list)
    ratingsPerItemAll = defaultdict(list)

    for user, item, rating in allRatings:
        ratingsPerUserAll[user].append((item, rating))
        ratingsPerItemAll[item].append((user, rating))

    print(f"training on {len(allRatings)} ratings (train + validation combined)")
    print(
        f"unique users: {len(ratingsPerUserAll)}, unique items: {len(ratingsPerItemAll)}"
    )

    # continue training from the current parameters
    for i in range(finalTrainingRounds):
        alpha = alphaUpdate(allRatings, alpha, betaU, betaI)
        betaU = betaUUpdate(ratingsPerUserAll, alpha, betaU, betaI, lambdaU)
        betaI = betaIUpdate(ratingsPerItemAll, alpha, betaU, betaI, lambdaI)

        trainMSE, trainMSEReg = msePlusReg(
            allRatings,
            alpha,
            betaU,
            betaI,
            lambdaU,
            lambdaI,
        )

        print(
            f"Round {i + 1}: Training MSE = {trainMSE:.4f}, MSE+Reg = {trainMSEReg:.4f}"
        )

    print(f"\nfinal training MSE after combined data: {trainMSE:.4f}")

    # write predictions for test set
    print("\n" + "=" * 60)
    print("generating test predictions...")
    writePredictionsRating(alpha, betaU, betaI)
    print("predictions written to predictions_Rating.csv")

    return my_MSE, baseline_MSE


if __name__ == "__main__":

    ##################################################
    # rating prediction                              #
    ##################################################

    my_MSE, baseline_MSE = runRatingPrediction()
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"baseline model: {baseline_MSE:.4f}")
    print(f"my model:       {my_MSE:.4f}")
    print(f"improvement:    {my_MSE - baseline_MSE:+.4f}")
    print("=" * 60)
