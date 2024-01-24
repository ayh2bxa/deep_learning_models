import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        Z = np.zeros((A.shape[0],self.W.shape[0],A.shape[2]-self.W.shape[2]+1,A.shape[3]-self.W.shape[3]+1))
        
        for wout in range(Z.shape[2]):
            for hout in range(Z.shape[3]):
                Z[:,:,wout,hout] = np.tensordot(A[:,:,wout:wout+self.W.shape[2],hout:hout+self.W.shape[3]],self.W,axes=([1,2,3],[1,2,3]))
        for b in range(Z.shape[0]):
            for cout in range(Z.shape[1]):
                Z[b,cout,:,:] += self.b[cout]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        pdLdZ = np.zeros((dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2] + 2*(self.W.shape[2]-1), dLdZ.shape[3] + 2*(self.W.shape[3]-1)))
        pdLdZ[:,:,self.W.shape[2]-1:self.W.shape[2]-1+dLdZ.shape[2],self.W.shape[3]-1:self.W.shape[3]-1+dLdZ.shape[3]] = dLdZ
        dLdA = np.zeros(self.A.shape)
        
        for i in range(dLdA.shape[2]):
            for j in range(dLdA.shape[3]):
                dLdA[:,:,i,j] = np.tensordot(pdLdZ[:,:,i:i+self.W.shape[2],j:j+self.W.shape[3]],np.flip(np.flip(self.W,2),3),axes=([1,2,3],[0,2,3]))

        for k in range(self.W.shape[2]):
            for j in range(self.W.shape[3]):
                self.dLdW[:,:,k,j] = np.tensordot(dLdZ, self.A[:,:,k:k+dLdZ.shape[2],j:j+dLdZ.shape[3]], axes=([0,2,3],[0,2,3]))
                
        self.dLdb = np.sum(np.sum(np.sum(dLdZ,axis=2),axis=2),axis=0)

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels,out_channels,kernel_size,weight_init_fn,bias_init_fn)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        dLdA = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA)

        return dLdA
