

from scipy.io import loadmat, savemat
from neurocombat_sklearn import CombatModel
import numpy as np
import tqdm



def getCombatEC():

    data = loadmat(r"../../data/removeCovCombatEC").get('removeCovCombatEC')
    cov = loadmat(r"../../data/Cov.mat").get('data')
    cov = cov.T
    siteCov = cov[:,0]
    siteCov = np.reshape(siteCov,(1611,1))
    diagCov = cov[:,1]
    diagCov = np.reshape(diagCov,(1611,1))
    ageCov = cov[:,2]
    ageCov = np.reshape(ageCov,(1611,1))
    sexCov = cov[:,3]
    sexCov = np.reshape(sexCov,(1611,1))
    eduCov = cov[:,4]
    eduCov = np.reshape(eduCov,(1611,1))

    # Creating model
    model = CombatModel()

    # Fitting model
    # make sure that your inputs are 2D, e.g. shape [n_samples, n_discrete_covariates]
    model.fit(data,sites=siteCov,discrete_covariates=np.concatenate((diagCov,sexCov),axis=1),
              continuous_covariates=np.concatenate((eduCov,ageCov),axis=1))

    # Harmonize data
    # could be performed together with fitt by using .fit_transform method
    data_combat = model.transform(data,sites=siteCov,discrete_covariates=np.concatenate((diagCov,sexCov),axis=1),
              continuous_covariates=np.concatenate((eduCov,ageCov),axis=1))


    savemat(r'...\combatEC.mat', {'combatEC': data_combat})


def getRemoveCovCombatEC():


    data = loadmat(r"..\..\data\combatEC.mat").get('combatEC')
    cov = loadmat(r"...\Cov.mat").get('data')
    cov = cov.T

    ageCov = cov[:,2]
    ageCov = np.reshape(ageCov,(1611,1))
    sexCov = cov[:,3]
    sexCov = np.reshape(sexCov,(1611,1))
    ageCov = np.array(ageCov)
    sexCov = np.array(sexCov)
    sexCov = sexCov.flatten()
    ageCov = ageCov.flatten()

    # data=data.T

    """
    Multiple linear regression removing covariates of sex and age
    """
    from sklearn.linear_model import LinearRegression
    X=[[],[]]

    for i in range(1611):
        X[0].append(sexCov[i])

    for i in range(1611):
        X[1].append(ageCov[i])

    X = np.array(X)
    X = X.T
    for i in tqdm.tqdm(range(13340)): #13340 ECs have 13340 independent multiple linear regression models
    # for i in range(13340):
        y=[]
        for sub in range(1611):
            y.append(data[sub][i])
        reg = LinearRegression().fit(X, y)

        w=[]
        w.append(reg.coef_)
        # print(w)
        for sub in range(1611):
            data[sub][i]=data[sub][i]-w[0][0]*sexCov[sub]-w[0][1]*ageCov[sub]


    # data=data.T

    savemat(r'...\removeCovCombatEC.mat',{'removeCovCombatEC':data})

# Building functional connectivity matrix data
def getFcMatrix():


    data = loadmat(r"..\..\data\removeCovCombatfisherFC.mat").get('removeCovCombatfisher')
    data = data.T
    matrix = np.zeros((1611,116,116))

    for i in range(data.shape[0]):
        feature = data[i]
        index = 0
        for row in range(116):
            for col in range(row+1,116):
                matrix[i][row][col] = feature[index]
                matrix[i][col][row] = feature[index]
                index = index + 1


        for diag in range(116):
            matrix[i][diag][diag] = 1

    return matrix


def getEcMatrix():

    data = loadmat(r"..\..\data\removeCovCombatEC.mat").get('removeCovCombatEC')

    matrix = np.zeros((1611,116,116))

    for i in range(data.shape[0]):
        feature = data[i]
        index = 0
        for row in range(116):
            for col in range(row+1,116):
                matrix[i][row][col] = feature[index]
                index = index + 1

        for col in range(116):
            for row in range(col+1,116):
                matrix[i][row][col] = feature[index]
                index = index + 1

        for diag in range(116):
            matrix[i][diag][diag] = 1

    return matrix



if __name__ == "__main__":
    getCombatEC()
    getRemoveCovCombatEC()

    fcMatrix = getFcMatrix()
    np.save(r'...\FCMatrix.npy', fcMatrix)

    ecMatrix = getEcMatrix()
    np.save(r'...\ECMatrix.npy', ecMatrix)



