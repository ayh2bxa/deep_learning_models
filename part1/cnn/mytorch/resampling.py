import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        Z = np.zeros((A.shape[0], A.shape[1], self.upsampling_factor*(A.shape[2]-1)+1))
        Z[:,:,0:Z.shape[2]:self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = dLdZ[:,:,0:dLdZ.shape[2]:self.upsampling_factor]
        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        
        self.input_width = A.shape[2]
        Z = A[:,:,0:A.shape[2]:self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_width))
        dLdA[:,:,0:dLdA.shape[2]:self.downsampling_factor] = dLdZ
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        Z = np.zeros((A.shape[0], A.shape[1], self.upsampling_factor*(A.shape[2]-1)+1, self.upsampling_factor*(A.shape[3]-1)+1))
        Z[:,:,0:Z.shape[2]:self.upsampling_factor,0:Z.shape[3]:self.upsampling_factor] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = dLdZ[:,:,0:dLdZ.shape[2]:self.upsampling_factor,0:dLdZ.shape[3]:self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        self.input_width = A.shape[2]
        self.input_height = A.shape[3]
        Z = A[:,:,0:A.shape[2]:self.downsampling_factor,0:A.shape[3]:self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_width, self.input_height))
        dLdA[:,:,0:dLdA.shape[2]:self.downsampling_factor,0:dLdA.shape[3]:self.downsampling_factor] = dLdZ
        return dLdA
