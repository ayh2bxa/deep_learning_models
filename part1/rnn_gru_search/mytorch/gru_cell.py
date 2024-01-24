import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()
        self.n_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        r_t = self.r_act(self.Wrx@self.x+self.brx+self.Wrh@h_prev_t+self.brh)
        z_t = self.z_act(self.Wzx@self.x+self.bzx+self.Wzh@h_prev_t+self.bzh)
        n_t = self.n_act(self.Wnx@self.x+self.bnx+r_t*(self.Wnh@h_prev_t+self.bnh))
        h_t = (1-z_t)*n_t+z_t*h_prev_t
        self.r = r_t
        self.z = z_t
        self.n = n_t
        self.h_t = h_t
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        # reshape self.x and self.hidden
        self.x = self.x.reshape((len(self.x),1))
        self.hidden = self.hidden.reshape((len(self.hidden),1))
        dLdh = delta
        dhdz = np.diag((self.hidden.T-self.n)[0])
        dLdz = dLdh@dhdz
        dhdn = np.diag(1-self.z)
        dLdn = dLdh@dhdn
        dn_act = self.n_act.backward().reshape((1,len(self.n_act.backward())))
        dr_act = self.r_act.backward().reshape((1,len(self.r_act.backward())))
        dz_act = self.z_act.backward().reshape((1,len(self.z_act.backward())))
        
        # dx
        dzdx = (dz_act.T)*self.Wzx
        drdx = (dr_act.T)*self.Wrx
        dndx = (dn_act.T)*(self.Wnx)
        
        dndr = dn_act*(self.Wnh@self.hidden+self.bnh)
        dhdr = dhdn*dndr
        dLdr = dLdh@dhdr

        dx = dLdn@dndx+dLdz@dzdx+dLdr@drdx
        
        # dbs
        self.dbzh = dLdz*dz_act
        self.dbzx = dLdz*dz_act
        self.dbrx = dLdr*dr_act
        self.dbrh = dLdr*dr_act
        self.dbnx = dLdn*dn_act 
        self.dbnh = dLdn*dn_act*self.r
        
        # dWs
        self.dWrx = (self.x@dLdr*dr_act).T
        self.dWzx = (self.x@dLdz*dz_act).T
        self.dWnx = (self.x@dLdn*dn_act).T
        self.dWrh = (self.hidden@dLdr*dr_act).T
        self.dWzh = (self.hidden@dLdz*dz_act).T
        self.dWnh = (self.hidden@dLdn*dn_act*self.r).T
        
        # dh_prev_t
        dhdh_prev_t = self.z.reshape(1,self.h)
        dndh_prev_t = (dn_act*self.r.reshape(1,self.h)).T*self.Wnh
        dzdh_prev_t = dz_act.T*self.Wzh
        drdh_prev_t = dr_act.T*self.Wrh
        dh_prev_t = dLdh*dhdh_prev_t+dLdn@dndh_prev_t+dLdz@dzdh_prev_t+dLdr@drdh_prev_t

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
        