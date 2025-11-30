import numpy as np
from sklearn.linear_model import Ridge

##################################################
# tfidf residual regression                      #
##################################################

def tfidfRidgeReg(ratingsTrainWithTFIDF, preds, a=1.0):
    yTrain = np.array([r[-1] for r in ratingsTrainWithTFIDF])
    residuals = yTrain - preds
    
    # get just the TFIDF vectors
    XTrainTFIDF = [r[2:-1] for r in ratingsTrainWithTFIDF]
    
    ridge = Ridge(alpha=a)
    ridge.fit(XTrainTFIDF, residuals)
    
    return ridge

