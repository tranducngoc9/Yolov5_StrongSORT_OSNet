
from __future__ import absolute_import, division

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z

#debug log
from inspect import currentframe, getframeinfo
import datetime

debug_log = True

def Debug_log(cf, filename, name = ''):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================  file {filename} , line : {cf.f_lineno} {name}')

# Debug_log(currentframe(), getframeinfo(currentframe()).filename)


class KalmanFilterNew(object):
    def __init__(self, dim_x, dim_z, dim_u=0):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))        # state
        self.P = eye(dim_x)               # uncertainty covariance
        self.Q = eye(dim_x)               # process uncertainty
        self.B = None                     # control transition matrix
        self.F = eye(dim_x)               # state transition matrix
        self.H = zeros((dim_z, dim_x))    # measurement function
        self.R = eye(dim_z)               # measurement uncertainty
        self._alpha_sq = 1.               # fading memory control
        self.M = np.zeros((dim_x, dim_z)) # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()             
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # keep all observations 
        self.history_obs = []

        self.inv = np.linalg.inv

        self.attr_saved = None
        self.observed = False 


    def predict(self, u=None, B=None, F=None, Q=None):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q


        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()



    def freeze(self):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        """
            Save the parameters before non-observation forward
        """
        self.attr_saved = deepcopy(self.__dict__)


    def unfreeze(self):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if self.attr_saved is not None:
            new_history = deepcopy(self.history_obs)
            self.__dict__ = self.attr_saved
            # self.history_obs = new_history 
            self.history_obs = self.history_obs[:-1]
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur)==0)[0]
            index1 = indices[-2]
            index2 = indices[-1]
            box1 = new_history[index1]
            x1, y1, s1, r1 = box1 
            w1 = np.sqrt(s1 * r1)
            h1 = np.sqrt(s1 / r1)
            box2 = new_history[index2]
            x2, y2, s2, r2 = box2 
            w2 = np.sqrt(s2 * r2)
            h2 = np.sqrt(s2 / r2)
            time_gap = index2 - index1
            dx = (x2-x1)/time_gap
            dy = (y2-y1)/time_gap 
            dw = (w2-w1)/time_gap 
            dh = (h2-h1)/time_gap
            for i in range(index2 - index1):
                """
                    The default virtual trajectory generation is by linear
                    motion (constant speed hypothesis), you could modify this 
                    part to implement your own. 
                """
                x = x1 + (i+1) * dx 
                y = y1 + (i+1) * dy 
                w = w1 + (i+1) * dw 
                h = h1 + (i+1) * dh
                s = w * h 
                r = w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                """
                    I still use predict-update loop here to refresh the parameters,
                    but this can be faster by directly modifying the internal parameters
                    as suggested in the paper. I keep this naive but slow way for 
                    easy read and understanding
                """
                self.update(new_box)
                if not i == (index2-index1-1):
                    self.predict()


    def update(self, z, R=None, H=None):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # append the observation
        self.history_obs.append(z)
        
        if z is None:
            if self.observed:
                """
                    Got no observation so freeze the current parameters for future
                    potential online smoothing.
                """
                self.freeze()
            self.observed = False 
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return
        
        # self.observed = True
        if not self.observed:
            """
                Get observation, use online smoothing to re-update parameters
            """
            self.unfreeze()
        self.observed = True

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()



