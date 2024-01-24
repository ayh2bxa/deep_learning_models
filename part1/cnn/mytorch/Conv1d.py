# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        Z = np.zeros((A.shape[0], self.W.shape[0], A.shape[2]-self.W.shape[2]+1))
        for o in range(Z.shape[2]):
            Z[:,:,o] = np.tensordot(A[:,:,o:o+self.W.shape[2]],self.W,axes=([1,2],[1,2]))
        for b in range(Z.shape[0]):
            for co in range(Z.shape[1]):
                Z[b,co,:] += self.b[co]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        pdLdZ = np.zeros((dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2] + 2*(self.W.shape[2]-1)))
        pdLdZ[:,:,self.W.shape[2]-1:self.W.shape[2]-1+dLdZ.shape[2]] = dLdZ
        dLdA = np.zeros(self.A.shape)
        
        for i in range(dLdA.shape[2]):
            dLdA[:,:,i] = np.tensordot(pdLdZ[:,:,i:i+self.W.shape[2]],np.flip(self.W,2),axes=([1,2],[0,2]))
        
        for k in range(self.W.shape[2]):
            self.dLdW[:,:,k] = np.tensordot(dLdZ, self.A[:,:,k:k+dLdZ.shape[2]], axes=([0,2],[0,2]))
        
        self.dLdb = np.sum(np.sum(dLdZ,axis=2),axis=0)
        
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() instance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdA = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA)

        return dLdA
