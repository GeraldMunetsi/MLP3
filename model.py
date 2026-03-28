import torch 
import torch.nn as nn
import numpy as np
import math
from scipy.interpolate import BSpline


#random fouriers

class RFF(nn.Module):
    def __init__(self, input_dim=3,D=64,sigma=1.0): # D is the number of featiures
        super().__init__()
        self.input_dim=input_dim
        self.D=D
        self.sigma=sigma
    
        W=torch.randn(input_dim,D)/sigma # remember .randn , generate sample from N(0,1)
        b=torch.rand(D) *2*np.pi

        self.W=nn.Parameter(W) # recall you are just wrapping the tensor and telling the frame work that these are tesnsore need to be optimized, W, b becomes part of the model.parametes  , gradients are computed for it , teh optimizer updated it         self.b=nn.Parameter(b) 
        self.b=nn.Parameter(b)

        self.scale=math.sqrt(2.0/D)

    def forward(self,x):
        projection= x@self.W + self.b
        return self.scale*torch.cos(projection)
    
#Parameter encoder

class FourierParameterEncoder(nn.Module):
    def __init__(self, input_dim=3,n_fourier=64, hidden_dim=32, output_dim=16):
        super().__init__()
        self.ff=RFF(
            input_dim=input_dim,
            D=n_fourier,
            sigma=1.0
        )

        self.mlp=nn.Sequential(
            nn.Linear(n_fourier,hidden_dim),
            nn.BatchNorm1d(hidden_dim), #nn.BatchNorm1d is Batch Normalization for 1-D feature vectors. It stabilizes and speeds up training by normalizing activations across the batch before passing them to the next layer.
            nn.ReLu(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self,params):
        fourier_features=self.ff(self.parameters)
        return self.mlp(fourier_features)
    
#B splines

class Spline(nn.Module):
    def __init__(self, n_knots=12, degree=3,points=50): #points represents the number of points along the time axis where the spline will be evaluated, 
        super().__init__()
        self.n_knots=n_knots
        self.degree=degree
        self.points=points

        knots= self._knot_vector(n_knots,degree) # knots are the points where the bssis functiond join, so lets create a B-spline knot sequence, # subic spline requires degree +1 repeated boundary nots to ensure curve starts and ends smoothly
        self.register_buffer("knots","knots")  # we store tensor inside the model

        eval=points=torch.linspace(1,1,points) #uniform knot vector
        self.register_buffer("eval",eval)
        basis_matrix=self.compute_basis_matrix_scipy()
        self.register_buffer("basis_matrix",basis_matrix)

    def _knot_vector(self,n_basis, n_knots,degree):
        n_knots=n_basis+ degree +1 # follows fom the intuition
        interior_count=n_knots-2*(degree+1) # this computes free interior knots 
        interior=torch.linspace(0,1,interior_count+2)[1:-1] #uniformly distributing interiour nots, removing boundary points, this also avoids degenerate spline collapse
        return torch.cat(
            torch.zeros(degree+1), #clamped spline boundary condition
            interior,
            torch.ones(degree+1)
        )



