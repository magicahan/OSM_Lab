#======================================================================
#
#     This routine interfaces with the TASMANIAN Sparse grid
#     The crucial part is 
#
#     aVals[iI]=solveriter.iterate(aPoints[iI], n_agents)[0]  
#     => at every gridpoint, we solve an optimization problem
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================
import config
import TasmanianSG
import numpy as np
from parameters import *
import nonlinear_solver_iterate as solveriter

#======================================================================
theta_vals = np.array([0.9, 0.95, 1.0, 1.05, 1.10])
probs_vals = np.ones(5)*1/5
def sparse_grid_iter(n_agents, iDepth, valold):
    
    grid  = TasmanianSG.TasmanianSparseGrid()

    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))


    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents
    iOut=1

    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid.setDomainTransform(ranges)

    aPoints=grid.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])
    
    file=open("comparison1.txt", 'w')
    for iI in range(iNumP1):
        #for i in range(theta_vals.shape[0]):
            #config.theta = theta_vals[i]
           # prob = probs_vals[i]
        aVals[iI] = solveriter.iterate(aPoints[iI], n_agents, valold)[0]
        v=aVals[iI]*np.ones((1,1))
        to_print=np.hstack((aPoints[iI].reshape(1,n_agents), v))
        np.savetxt(file, to_print, fmt='%2.16f')
        
    file.close()
    grid.loadNeededPoints(aVals)
    
    f=open("grid_iter.txt", 'w')
    np.savetxt(f, aPoints, fmt='% 2.16f')
    f.close()
    
    return grid

#======================================================================
