import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = np.sum(Z,axis=0)/self.N
        self.M = self.M.reshape(1, Z.shape[1])
        self.V = np.sum(np.power(Z-np.ones((Z.shape[0],1))@self.M.reshape(1,Z.shape[1]),2),axis=0)/self.N
        self.V = self.V.reshape(1, Z.shape[1])

        if eval == False:
            # training mode
            self.NZ = (Z-self.M)/np.sqrt(self.V+self.eps)
            self.BZ = self.BW*self.NZ+self.Bb

            self.running_M = self.alpha*self.running_M+(1-self.alpha)*self.M
            self.running_V = self.alpha*self.running_V+(1-self.alpha)*self.V
        else:
            # inference mode
            self.NZ = (Z - self.running_M)/np.sqrt(self.running_V+self.eps)
            self.BZ = self.BW*self.NZ+self.Bb

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ*self.NZ,axis=0)
        self.dLdBb = np.sum(dLdBZ,axis=0)
        Ones = np.ones((self.N, 1))
        dLdNZ = dLdBZ*(Ones@self.BW)
        dLdV = -0.5*np.sum(dLdNZ*(self.Z-Ones@self.M)*np.power(Ones@(self.V+self.eps),-1.5),axis=0)
        S = ((-2/self.N)*np.sum(self.Z-Ones@self.M, axis=0))
        dNZdM = -1*np.power(Ones@(self.V+self.eps),-0.5)-0.5*(self.Z-Ones@self.M)*np.power(Ones@(self.V+self.eps),-1.5)*S
        print(dLdNZ.shape)
        print(dNZdM.shape)
        dLdM = np.sum(dLdNZ*dNZdM, axis=0)
        
        dLdZ = dLdNZ*(Ones@np.power(self.V+self.eps,-0.5))+(dLdV)*(2/self.N)*(self.Z-Ones@self.M)+(1/self.N)*dLdM

        return dLdZ
