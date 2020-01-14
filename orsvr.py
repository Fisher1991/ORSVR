import sys
import numpy as np
from scipy.linalg import solve


def sign(x):
    """ Returns sign. Numpys sign function returns 0 instead of 1 for zero values. :( """
    if x >= 0:
        return 1
    else:
        return -1


class OnlineSVR:
    def __init__(self, numFeatures, C, nu, kernelParam, bias=0, debug=False):
        # Configurable Parameters
        self.numFeatures = numFeatures
        self.C = C
        self.nu = nu
        self.kernelParam = kernelParam
        self.bias = bias
        self.debug = debug
        print('SELF', self.C, self.nu, self.kernelParam)
        # Algorithm initialization
        self.numSamplesTrained = 0
        self.weights = np.array([])

        # Samples X (features) and Y (truths)
        self.X = list()
        self.Y = list()
        # Working sets, contains indices pertaining to X and Y
        self.supportSetIndices = list()
        self.errorSetIndices = list()
        self.remainderSetIndices = list()
        self.R = np.matrix([])

    def findMinVariation(self, H, beta, gamma, i):
        """ Finds the variations of each sample to the new set.
        Lc1: distance of the new sample to the SupportSet
        Lc2: distance of the new sample to the ErrorSet
        Ls(i): distance of the support samples to the ErrorSet/RemainingSet
        Le(i): distance of the error samples to the SupportSet
        Lr(i): distance of the remaining samples to the SupportSet
        """
        # Find direction q of the new sample
        q = -sign(H[i])
        print('q1',q)
        # Compute variations
        Lc1 = self.findVarLc1(H, gamma, q, i)
        q = sign(Lc1)
        print('q2',q)
        Lc2 = self.findVarLc2(H, q, i)
        Ls = self.findVarLs(H, beta, q)
        Le = self.findVarLe(H, gamma, q)
        Lr = self.findVarLr(H, gamma, q)
        # Check for duplicate minimum values, grab one with max gamma/beta, set others to inf
        # Support set
        if Ls.size > 1:
            minS = np.abs(Ls).min()
            results = np.array([k for k, val in enumerate(Ls) if np.abs(val) == minS])
            if len(results) > 1:
                betaIndex = beta[results + 1].argmax()
                Ls[results] = q * np.inf
                Ls[results[betaIndex]] = q * minS
        # Error set
        if Le.size > 1:
            minE = np.abs(Le).max()
            results = np.array([k for k, val in enumerate(Le) if np.abs(val) == minE])
            if len(results) > 1:
                errorGamma = gamma[self.errorSetIndices]
                gammaIndex = errorGamma[results].argmax()
                Le[results] = q * np.inf
                Le[results[gammaIndex]] = q * minE
        # Remainder Set
        if Lr.size > 1:
            minR = np.abs(Lr).max()
            results = np.array([k for k, val in enumerate(Lr) if np.abs(val) == minR])
            if len(results) > 1:
                remGamma = gamma[self.remainderSetIndices]
                gammaIndex = remGamma[results].argmax()
                Lr[results] = q * np.inf
                Lr[results[gammaIndex]] = q * minR

        # Find minimum absolute variation of all, retain signs. Flag determines set-switching cases.
        minLsIndex = np.abs(Ls).argmin()
        minLeIndex = np.abs(Le).argmin()
        minLrIndex = np.abs(Lr).argmin()
        minIndices = [None, None, minLsIndex, minLeIndex, minLrIndex]
        print("nmbnmnm", Lr[minLrIndex])
        minValues = np.array([Lc1, Lc2, Ls[minLsIndex], Le[minLeIndex], Lr[minLrIndex]])

        if np.abs(minValues).min() == np.inf:
            print('No weights to modify! Something is wrong.')
            sys.exit()
        # index = 0
        # maxvalue = np.abs(minValues[0])
        # for k in range(1,len(minValues)):
        #     if np.abs(minValues[k]) == np.inf:
        #         continue
        #     elif np.abs(minValues[k]) > maxvalue:
        #         maxvalue = np.abs(minValues[k])
        #         index = k
        flag = np.abs(minValues).argmin()
        # flag = index
        if self.debug:
            print('MinValues', minValues)
        return minValues[flag], flag, minIndices[flag]

    def findVarLc1(self, H, gamma, q, i):
        # weird hacks below
        Lc1 = np.nan
        if gamma.size < 2:
            g = gamma
        else:
            g = gamma.item(i)
        # weird hacks above

        if g <= 0:
            Lc1 = np.array(q * np.inf)
        elif H[i] > 0 and 0 <= self.weights[i] and self.weights[i] <= self.C:
            Lc1 = (-H[i]) / (1*50)
        elif H[i] < 0 and 0 <= self.weights[i] and self.weights[i] <= self.C:
            Lc1 = (-H[i]) / (1*50)
        else:
            print('Something is weird.')
            print('i', i)
            print('q', q)
            print('gamma', gamma)
        print('g', g)
        print('H[i]', H[i])
        print('weights[i]', self.weights[i])
            # Lc1 = (-H[i]+H[i]) / g

        if np.isnan(Lc1):
            Lc1 = np.array(q * np.inf)
        return Lc1.item()

    def findVarLc2(self, H, q, i):
        if len(self.supportSetIndices) > 0:
            if q > 0:
                # Lc2 = -self.weights[i] + self.C
                Lc2 = np.array(q * np.inf)
            else:
                # Lc2 = self.weights[i] - self.C
                Lc2 = np.array(q * np.inf)
        else:
            Lc2 = np.array(q * np.inf)
        if np.isnan(Lc2):
            Lc2 = np.array(q * np.inf)
        return Lc2

    def findVarLs(self, H, beta, q):
        if len(self.supportSetIndices) > 0 and len(beta) > 0:
            Ls = np.zeros([len(self.supportSetIndices), 1])
            supportWeights = self.weights[self.supportSetIndices]
            supportH = H[self.supportSetIndices]
            print('supportWeights',supportWeights)
            print('supportH',supportH)
            for k in range(len(self.supportSetIndices)):
                if q * beta[k + 1] == 0:
                    Ls[k] = q * np.inf
                elif q * beta[k + 1] > 0:
                    print('q*beta>0')
                    if supportH[k] > 0.00001:
                        print('vvvv')
                        if supportWeights[k] < 0:
                            Ls[k] = (-supportWeights[k]) / beta[k + 1]
                        elif supportWeights[k] > 0:
                            Ls[k] = supportWeights[k] / beta[k + 1]
                        else:
                            Ls[k] = q * np.inf
                    elif supportH[k] < -0.00001:
                        print('qqqq')
                        if supportWeights[k] > self.C:
                            Ls[k] = (supportWeights[k] - self.C) / beta[k + 1]
                        elif supportWeights[k] < self.C:
                            Ls[k] = (-supportWeights[k] + self.C) / beta[k + 1]
                        else:
                            Ls[k] = q * np.inf
                    else:
                        print('bbbb')
                        if supportWeights[k] < 0:
                            Ls[k] = (-supportWeights[k]) / beta[k + 1]*1000
                        elif supportWeights[k] > 0:
                            Ls[k] = (supportWeights[k] / beta[k + 1])*1000
                        else:
                            Ls[k] = q * np.inf
                        print("Ls[k]",Ls[k])
                else:
                    print('q*beta<0')
                    if supportH[k] > 0.00001:
                        print('vvvv')
                        if supportWeights[k] > 0:
                            Ls[k] = -supportWeights[k] / beta[k + 1]
                        elif supportWeights[k] < 0:
                            Ls[k] = (supportWeights[k]) / beta[k + 1]
                        else:
                            Ls[k] = q * np.inf
                    elif supportH[k] < -0.00001:
                        print('qqqq')
                        if supportWeights[k] > self.C:
                            Ls[k] = (-supportWeights[k] + self.C) / beta[k + 1]
                        elif supportWeights[k] < self.C:
                            Ls[k] = (supportWeights[k] - self.C) / beta[k + 1]
                        else:
                            Ls[k] = q * np.inf
                    else:
                        print('bbbb')
                        if supportWeights[k] > 0:
                            Ls[k] = -supportWeights[k] / beta[k + 1]*1000
                            print('CCCC',Ls[k])
                        # elif supportWeights[k] < 0:
                        #     Ls[k] = (supportWeights[k]) / beta[k + 1]
                        else:
                            Ls[k] = q * np.inf
        else:
            Ls = np.array([q * np.inf])

        # Correct for NaN
        Ls[np.isnan(Ls)] = q * np.inf
        if Ls.size > 1:
            Ls.shape = (len(Ls), 1)
            # Check for broken signs
            for val in Ls:
                if sign(val) == -sign(q) and val != 0:
                    print('Sign mismatch error in Ls! Exiting.')
                    sys.exit()
        # print('findVarLs',Ls)
        return Ls

    def findVarLe(self, H, gamma, q):
        if len(self.errorSetIndices) > 0:
            Le = np.zeros([len(self.errorSetIndices), 1])
            errorGamma = gamma[self.errorSetIndices]
            errorWeights = self.weights[self.errorSetIndices]
            errorH = H[self.errorSetIndices]
            for k in range(len(self.errorSetIndices)):
                if q * errorGamma[k] == 0:
                    Le[k] = q * np.inf
                elif q * errorGamma[k] > 0:
                    if errorWeights[k] > 0:
                        if errorH[k] > 0:
                            Le[k] = (errorH[k]) / errorGamma[k]
                        else:
                            Le[k] = q * np.inf
                    # else:
                    #     if errorH[k] < 0:
                    #         Le[k] = (-errorH[k]) / errorGamma[k]
                    #     else:
                    #         Le[k] = q * np.inf
                else:
                    if errorWeights[k] > 0:
                        if errorH[k] < 0:
                            Le[k] = (errorH[k]) / errorGamma[k]
                        else:
                            Le[k] = q * np.inf
                    # else:
                    #     if errorH[k] > 0:
                    #         Le[k] = (-errorH[k]) / errorGamma[k]
                    #     else:
                    #         Le[k] = q * np.inf
        else:
            Le = np.array([q * np.inf])

        # Correct for NaN
        Le[np.isnan(Le)] = q * np.inf
        if Le.size > 1:
            Le.shape = (len(Le), 1)
            # Check for broken signs
            for val in Le:
                if sign(val) == -sign(q) and val != 0:
                    print('Sign mismatch error in Le! Exiting.')
                    sys.exit()
        # print('findVarLe',Le)
        return Le

    def findVarLr(self, H, gamma, q):
        if len(self.remainderSetIndices) > 0:
            Lr = np.zeros([len(self.remainderSetIndices), 1])
            remGamma = gamma[self.remainderSetIndices]
            remH = H[self.remainderSetIndices]
            print('remH', remH)
            print("xxxxx", len(self.X))
            for k in range(len(self.remainderSetIndices)):
                if q * remGamma[k] == 0:
                    Lr[k] = q * np.inf
                elif q * remGamma[k] > 0:
                    print('q * remGamma[k] > 0')
                    if remH[k] < -0.000001:
                        # Lr[k] = q * np.inf
                        Lr[k] = (-remH[k]) / remGamma[k]
                    elif remH[k] > 0.000001:
                    # elif remH[k] > 0:
                          Lr[k] = (remH[k]) / remGamma[k]
                    else:
                        print("vvvvvbbb")
                        Lr[k] = q * np.inf
                else:
                    print('q * remGamma[k] < 0')
                    if remH[k] < -0.000001:
                        Lr[k] = (remH[k]) / remGamma[k]
                        # Lr[k] = q * np.inf
                    elif remH[k] > 0.000001:
                    # elif remH[k] > 0:
                          Lr[k] = (-remH[k]) / remGamma[k]
                    else:
                        Lr[k] = q * np.inf
        else:
            Lr = np.array([q * np.inf])

        # Correct for NaN
        Lr[np.isnan(Lr)] = q * np.inf
        if Lr.size > 1:
            Lr.shape = (len(Lr), 1)
            # Check for broken signs
            for val in Lr:
                if sign(val) == -sign(q) and val != 0:
                    print('Sign mismatch error in Lr! Exiting.')
                    sys.exit()
        print('findVarLr',Lr)
        return Lr

    def computeKernelOutput(self, set1, set2, num):
        """Compute kernel output. Uses a radial basis function kernel."""
        X1 = np.matrix(set1)
        X2 = np.matrix(set2).T
        # Euclidean distance calculation done properly
        [S, R] = X1.shape
        [R2, Q] = X2.shape
        X = np.zeros([S, Q])
        if Q < S:
            copies = np.zeros(S, dtype=int)
            for q in range(Q):
                # if self.debug:
                #     print('X1', X1)
                #     print('X2copies', X2.T[q + copies, :])
                #     print('power', np.power(X1 - X2.T[q + copies, :], 2))
                xsum = np.sum(np.power(X1 - X2.T[q + copies, :], 2), axis=1)
                xsum.shape = (xsum.size,)
                X[:, q] = xsum
        else:
            copies = np.zeros(Q, dtype=int)
            for i in range(S):
                X[i, :] = np.sum(np.power(X1.T[:, i + copies] - X2, 2), axis=0)
        X = np.sqrt(X)
        y = (1/num)*np.matrix(np.exp(-self.kernelParam * X ** 2))
        # if self.debug:
        #     print('distance', X)
        #     print('kernelOutput', y)
        return y

    def predict(self, newSampleX, num):
        X = np.array(self.X)
        # print("kkkk",(X.size/2))
        newX = np.array(newSampleX)
        weights = np.array(self.weights)
        weights.shape = (weights.size, 1)
        if self.numSamplesTrained > 0:
            y = self.computeKernelOutput(X, newX, num)
            return (weights.T @ y).T + self.bias
        else:
            return np.zeros_like(newX) + self.bias

    def computeMargin(self, newSampleX, newSampleY, num):
        fx = self.predict(newSampleX,num)
        newSampleY = np.array(newSampleY)
        newSampleY.shape = (newSampleY.size, 1)
        if self.debug:
            # print('fx', fx)
            # print('newSampleY', newSampleY)
            print('hx', fx - newSampleY)
        return fx - newSampleY

    def computeBetaGamma(self, i, num):
        """Returns beta and gamma arrays."""
        # Compute beta vector
        X = np.array(self.X)
        Qsi = self.computeQ(X[self.supportSetIndices, :], X[i, :], num)
        if len(self.supportSetIndices) == 0 or self.R.size == 0:
            beta = np.array([])
        else:
            beta = -self.R @ np.append(np.matrix([1]), Qsi, axis=0)
        # Compute gamma vector
        Qxi = self.computeQ(X, X[i, :], num)
        Qxs = self.computeQ(X, X[self.supportSetIndices, :], num)
        if len(self.supportSetIndices) == 0 or Qxi.size == 0 or Qxs.size == 0 or beta.size == 0:
            gamma = np.array(np.ones_like(Qxi))
        else:
            gamma = Qxi + np.append(np.ones([self.numSamplesTrained, 1]), Qxs, 1) @ beta

        # Correct for NaN
        beta[np.isnan(beta)] = 0
        gamma[np.isnan(gamma)] = 0
        # if self.debug:
        #     print('R', self.R)
        #     print('beta', beta)
        #     print('gamma', gamma)
        return beta, gamma

    def computeQ(self, set1, set2, num):
        set1 = np.matrix(set1)
        set2 = np.matrix(set2)
        Q = np.matrix(np.zeros([set1.shape[0], set2.shape[0]]))
        for i in range(set1.shape[0]):
            for j in range(set2.shape[0]):
                Q[i, j] = self.computeKernelOutput(set1[i, :], set2[j, :], num)
        return np.matrix(Q)

    def adjustSets(self, H, beta, gamma, i, flag, minIndex, num):
        print('Entered adjustSet logic with flag {0} and minIndex {1}.'.format(flag, minIndex))
        if flag not in range(5):
            print('Received unexpected flag {0}, exiting.'.format(flag))
            sys.exit()
        # add new sample to Support set
        if flag == 0:
            print('Adding new sample {0} to support set.'.format(i+1))
            H[i] = np.sign(H[i]) * 0
            self.supportSetIndices.append(i)
            self.R = self.addSampleToR(i, 'SupportSet', beta, gamma, num)
            return H, True
        # add new sample to Error set
        elif flag == 1:
            print('Adding new sample {0} to error set.'.format(i+1))
            self.weights[i] = np.sign(self.weights[i]) * self.C
            self.errorSetIndices.append(i)
            return H, True
        # move sample from Support set to Error or Remainder set
        elif flag == 2:
            index = self.supportSetIndices[minIndex]
            weightsValue = self.weights[index]
            if np.abs(weightsValue) < np.abs(self.C - abs(weightsValue)):
                self.weights[index] = 0
                weightsValue = 0
            else:
                self.weights[index] = np.sign(weightsValue) * self.C
                weightsValue = self.weights[index]
            # Move from support to remainder set
            if weightsValue == 0:
                print('Moving sample {0} from support to remainder set.'.format(index+1))
                self.remainderSetIndices.append(index)
                self.R = self.removeSampleFromR(minIndex)
                self.supportSetIndices.pop(minIndex)
            # move from support to error set
            elif np.abs(weightsValue) == self.C:
                print('Moving sample {0} from support to error set.'.format(index+1))
                self.errorSetIndices.append(index)
                self.R = self.removeSampleFromR(minIndex)
                self.supportSetIndices.pop(minIndex)
            else:
                print('Issue with set swapping, flag 2.', 'weightsValue:', weightsValue)
                sys.exit()
        # move sample from Error set to Support set
        elif flag == 3:
            index = self.errorSetIndices[minIndex]
            print('Moving sample {0} from error to support set.'.format(index+1))
            H[index] = np.sign(H[index]) * 0
            self.supportSetIndices.append(index)
            self.errorSetIndices.pop(minIndex)
            self.R = self.addSampleToR(index, 'ErrorSet', beta, gamma, num)
        # move sample from Remainder set to Support set
        elif flag == 4:
            index = self.remainderSetIndices[minIndex]
            print('Moving sample {0} from remainder to support set.'.format(index+1))
            H[index] = np.sign(H[index]) * 0
            self.supportSetIndices.append(index)
            self.remainderSetIndices.pop(minIndex)
            self.R = self.addSampleToR(index, 'RemainingSet', beta, gamma, num)
        return H, False

    def addSampleToR(self, sampleIndex, sampleOldSet, beta, gamma, num):
        print('Adding sample {0} to R matrix.'.format(sampleIndex+1))
        X = np.array(self.X)
        sampleX = X[sampleIndex, :]
        # sampleX.shape = (int(sampleX.size / self.numFeatures), self.numFeatures)
        # Add first element
        if self.R.shape[0] <= 1:
            Rnew = np.ones([2, 2])
            Rnew[0, 0] = -self.computeKernelOutput(sampleX, sampleX, num)
            Rnew[1, 1] = 0
        # Other elements
        else:
            # recompute beta/gamma if from error/remaining set
            if sampleOldSet == 'ErrorSet' or sampleOldSet == 'RemainingSet':
                # beta, gamma = self.computeBetaGamma(sampleIndex)
                Qii = self.computeKernelOutput(sampleX, sampleX, num)
                Qsi = self.computeKernelOutput(X[self.supportSetIndices[0:-1], :], sampleX, num)
                beta = -self.R @ np.append(np.matrix([1]), Qsi, axis=0)
                beta[np.isnan(beta)] = 0
                beta.shape = (len(beta), 1)
                gamma[sampleIndex] = Qii + np.append(1, Qsi.T) @ beta
                gamma[np.isnan(gamma)] = 0
                gamma.shape = (len(gamma), 1)
            # add a column and row of zeros onto right/bottom of R
            r, c = self.R.shape
            Rnew = np.append(self.R, np.zeros([r, 1]), axis=1)
            Rnew = np.append(Rnew, np.zeros([1, c + 1]), axis=0)
            # update R
            if gamma[sampleIndex] != 0:
                # Numpy so wonky! SO WONKY.
                beta1 = np.append(beta, [[1]], axis=0)
                Rnew = Rnew + 1 / gamma[sampleIndex].item() * beta1 @ beta1.T
            if np.any(np.isnan(Rnew)):
                print('R has become inconsistent. Training failed at sampleIndex {0}'.format(sampleIndex))
                sys.exit()
        return Rnew

    def removeSampleFromR(self, sampleIndex):
        print('Removing sample {0} from R matrix.'.format(sampleIndex+1))
        sampleIndex += 1
        I = list(range(sampleIndex))
        I.extend(range(sampleIndex + 1, self.R.shape[0]))
        I = np.array(I)
        I.shape = (1, I.size)
        # if self.debug:
        #     print('I', I)
        #     print('RII', self.R[I.T, I])
        # Adjust R
        if self.R[sampleIndex, sampleIndex] != 0:
            Rnew = self.R[I.T, I] - (self.R[I.T, sampleIndex] * self.R[sampleIndex, I]) / self.R[
                sampleIndex, sampleIndex].item()
        else:
            Rnew = np.copy(self.R[I.T, I])
        # Check for bad things
        if np.any(np.isnan(Rnew)):
            print('R has become inconsistent. Training failed removing sampleIndex {0}'.format(sampleIndex))
            sys.exit()
        if Rnew.size == 1:
            print('Time to annhilate R? R:', Rnew)
            Rnew = np.matrix([])
        return Rnew

    def learn(self, newSampleX, newSampleY, num):
        print('---------------------------------------------')
        print("[",newSampleX[0],newSampleY,"]")
        self.numSamplesTrained += 1
        # print(self.numSamplesTrained)
        self.X.append(newSampleX)
        self.Y.append(newSampleY)
        self.weights = np.append(self.weights, self.C * self.nu)
        i = self.numSamplesTrained - 1  # stupid off-by-one errors
        H = self.computeMargin(self.X, self.Y, num)
        if i !=0:
            # self.weights[self.supportSetIndices] = 0
            # H = self.computeMargin(self.X, self.Y, num)
            # self.remainderSetIndices.append(0)
            # self.R = self.removeSampleFromR(0)
            # self.supportSetIndices.pop(0)
            # self.weights = np.append(self.weights, self.C * self.nu)
            # print(self.weights[self.supportSetIndices])
            X = np.array(self.X)
            Qsi = self.computeQ(X[self.supportSetIndices, :], X[i, :],num)
            # print('Qsi',Qsi)
            Qs = self.computeQ(X[self.supportSetIndices, :], X[self.supportSetIndices, :],num)
            # print('Qs',Qs)
            Qsi = np.append(Qs, Qsi, axis=1)
            # print('Qsi', Qsi)
            Qis = self.computeQ(X[i, :],X[self.supportSetIndices, :],num)
            # print('Qis',Qis)
            Qii = self.computeQ(X[i, :], X[i, :],num)
            # print('Qii',Qii)
            Qis = np.append(Qis, Qii, axis=1)
            # print('Qis', Qis)
            Qtemp = np.append(Qsi, Qis, axis=0)
            temp = np.append(np.ones([1, len(self.supportSetIndices)]), np.matrix([1]), axis=1)
            Qtemp = np.append(Qtemp, temp, axis=0)
            # print('Qtemp',Qtemp)
            Qb = np.append(np.ones([len(self.supportSetIndices)+1,1]), np.matrix([0]), axis=0)
            # print('Qb',Qb)
            Qtemp = np.append(Qtemp, Qb, axis=1)
            # print('Qtemp',Qtemp)
            margin = np.append(H[self.supportSetIndices, :],
                               H[i], axis=0)
            margin = np.append(margin,np.matrix([0]), axis=0)
            # print('margin',margin)
            jie = solve(Qtemp, margin)
            # print('jie',jie)
            weightDelta = np.array(jie[: -2])
            weightDelta.shape = (len(weightDelta),)
            self.weights[self.supportSetIndices] -= weightDelta
            self.weights[i] -= jie[-2]
            self.bias -= jie[-1]
            # print('self.weights', self.weights)
            # print('sumself.weights', sum(self.weights))
            # H = self.computeMargin(self.X, self.Y, num)
            Flag = True
            for k in range(len(self.weights)):
                if self.weights[k] < 0:
                    Flag = True
                    # print('有权重小于0.')
                    break

            if Flag:
                self.weights[self.supportSetIndices] += weightDelta
                self.weights[i] += jie[-2]
                self.bias += jie[-1]
                self.X.pop(i)
                self.Y.pop(i)
                self.weights = self.weights[:-1]
                self.numSamplesTrained -= 1
                H = self.computeMargin(self.X, self.Y, num)
                supportSet = []
                for j in range(len(self.supportSetIndices)):
                    supportSet.append(self.X[self.supportSetIndices[j]])
                supportSet = np.array(supportSet)
                # print("supportSet",supportSet)
                Qs = self.computeQ(supportSet, supportSet, num)
                Qs = np.append(Qs, np.ones([len(self.supportSetIndices), 1]), axis=1)
                temp = np.append(np.ones([1, len(self.supportSetIndices)]), np.matrix([0]), axis=1)
                Qs = np.append(Qs, temp, 0)
                # print('Qs',Qs)
                print("QQQ",H[self.supportSetIndices])
                # b = np.append(np.zeros([len(self.supportSetIndices), 1]), np.matrix([self.C*self.nu]), axis=0)
                b = np.append(-H[self.supportSetIndices]/50, np.matrix([self.C*self.nu]), axis=0)
                print('b',b)
                x = solve(Qs, b)
                print('方程解',x)
                # print("没有预处理的权重",self.weights)
                weightDelta = np.array(x[:-1])
                weightDelta.shape = (len(weightDelta),)
                self.weights[self.supportSetIndices] += weightDelta
                print("预处理的权重", self.weights)
                self.bias = self.bias + x[-1]
                print("预处理的偏差", self.bias)
                self.X.append(newSampleX)
                self.Y.append(newSampleY)
                self.weights = np.append(self.weights, 0)
                self.numSamplesTrained += 1
                i = self.numSamplesTrained - 1  # stupid off-by-one errors
                H = self.computeMargin(self.X, self.Y, num)

        # for k in range(len(H)-1):
        #     if H[k] < -0.00001:
        #         print('werid')
        #         forget = True
        #         return
        if (H[i] >= 0.1 and abs(self.weights[i])<=0.00001):
            print('Adding new sample {0} to remainder set.'.format(i + 1))
            self.remainderSetIndices.append(i)
            if self.debug:
                print('修改后的weights', self.weights)
                print('权重总和', sum(self.weights))
                print('修改后bias:', self.bias)
                print('修改后的H', H)
            return

        if (abs(H[i]) <= 0.00001 and 0 <= self.weights[i] <= self.C):
            print('Adding new sample {0} to support set.'.format(i + 1))
            beta, gamma = self.computeBetaGamma(i, num)
            self.supportSetIndices.append(i)
            self.R = self.addSampleToR(i, 'supportSet', beta, gamma, num)
            if self.debug:
                print('修改后的weights', self.weights)
                print('权重总和', sum(self.weights))
                print('修改后bias:', self.bias)
                print('修改后的H', H)
            return

        newSampleAdded = False
        iterations = 0
        while not newSampleAdded:
            # Ensure we're not looping infinitely
            iterations += 1
            if iterations > self.numSamplesTrained * 100:
                print('Warning: we appear to be in an infinite loop.')
                sys.exit()
                iterations = 0
            # Compute beta/gamma for constraint optimization
            beta, gamma = self.computeBetaGamma(i, num)
            # Find minimum variation and determine how we should shift samples between sets
            deltaC, flag, minIndex = self.findMinVariation(H, beta, gamma, i)
            print("deltaC",deltaC)
            # Update weights and bias based on variation
            if len(self.supportSetIndices) > 0 and len(beta) > 0:
                self.weights[i] += deltaC
                delta = beta * deltaC
                self.bias += delta.item(0)
                # numpy is wonky...
                weightDelta = np.array(delta[1:])
                weightDelta.shape = (len(weightDelta),)
                self.weights[self.supportSetIndices] += weightDelta
                H += gamma * deltaC
            else:
                self.bias += deltaC
                H += deltaC
            # Adjust sets, moving samples between them according to flag
            H, newSampleAdded = self.adjustSets(H, beta, gamma, i, flag, minIndex, num)
            print('经过这次调整的权重',self.weights)
        if self.debug:
            print('weights', self.weights)
            print('最后的权重总和', sum(self.weights))
            print('最后的偏差:', self.bias)
            H = self.computeMargin(self.X,self.Y, num)
            print('最后的margin', H)