import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        Z = np.zeros((A.shape[0],A.shape[1],A.shape[2]-self.kernel+1,A.shape[3]-self.kernel+1))
        self.maxind = np.empty(Z.shape,dtype='i,i')
        for b in range(Z.shape[0]):
            for c in range(Z.shape[1]):
                for i in range(Z.shape[2]):
                    for j in range(Z.shape[3]):
                        Z[b,c,i,j] = np.amax(A[b,c,i:i+self.kernel,j:j+self.kernel])
                        maxind_unconverted = np.argmax(A[b,c,i:i+self.kernel,j:j+self.kernel])
                        col = maxind_unconverted % self.kernel
                        row = maxind_unconverted // self.kernel
                        maxcol = j+col
                        maxrow = i+row
                        self.maxind[b,c,i,j] = (maxrow, maxcol)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        for b in range(dLdZ.shape[0]):
            for ch in range(dLdZ.shape[1]):
                for i in range(dLdZ.shape[2]):
                    for j in range(dLdZ.shape[3]):
                        r = self.maxind[b,ch,i,j][0]
                        c = self.maxind[b,ch,i,j][1]
                        dLdA[b,ch,r,c] += dLdZ[b,ch,i,j]
                        
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        Z = np.zeros((A.shape[0], A.shape[1], A.shape[2]-self.kernel+1, A.shape[3]-self.kernel+1))
        for b in range(Z.shape[0]):
            for ch in range(Z.shape[1]):
                for i in range(Z.shape[2]):
                    for j in range(Z.shape[3]):
                        Z[b,ch,i,j] = np.mean(A[b,ch,i:i+self.kernel,j:j+self.kernel])
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        for b in range(dLdZ.shape[0]):
            for ch in range(dLdZ.shape[1]):
                for i in range(self.kernel):
                    for j in range(self.kernel):
                        dLdA[b,ch,i:i+dLdZ.shape[2],j:j+dLdZ.shape[3]] += (dLdZ[b,ch,:,:]/self.kernel/self.kernel)
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        return self.downsample2d.forward(Z)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        return self.maxpool2d_stride1.backward(dLdA)


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        return self.downsample2d.forward(Z)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)
        return dLdA
