#======================================================================
#
#     This routine solves an infinite horizon growth model 
#     with dynamic programming and sparse grids
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     external libraries needed:
#     - IPOPT (https://projects.coin-or.org/Ipopt)
#     - PYIPOPT (https://github.com/xuy/pyipopt)
#     - TASMANIAN (http://tasmanian.ornl.gov/)
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================
import config
import nonlinear_solver_initial as solver     #solves opt. problems for terminal VF
import nonlinear_solver_iterate as solviter   #solves opt. problems during VFI
from parameters import *                      #parameters of model
import interpolation as interpol              #interface to sparse grid library/terminal VF
import interpolation_iter as interpol_iter    #interface to sparse grid library/iteration
import postprocessing as post                 #computes the L2 and Linfinity error of the model

import TasmanianSG                            #sparse grid library
import numpy as np
import interpolation_adaptive as interp_adap

refinement_level = 3
fTol = 1e-20

#======================================================================
# Start with Value Function Iteration

# terminal value function
valnew=TasmanianSG.TasmanianSparseGrid()
if (numstart==0):
    valnew=interpol.sparse_grid(n_agents, iDepth + refinement_level)
    valnew.write("valnew_1." + str(numstart) + ".txt") #write file to disk for restart

# value function during iteration
else:
    valnew.read("valnew_1." + str(numstart) + ".txt")  #write file to disk for restart
    
valold=TasmanianSG.TasmanianSparseGrid()
valold=valnew

theta_vals = np.array([0.9, 0.95, 1.0, 1.05, 1.10])
probs = np.ones(5)*1/5
for i in range(numstart, numits):
    val_list = []
    k_range=np.array([k_bar, k_up])
    ranges=np.empty((n_agents, 2))
    iDim=n_agents
    iOut=1
    val_temp = TasmanianSG.TasmanianSparseGrid()
    val_temp.makeLocalPolynomialGrid(iDim, iOut, 1, which_basis, "localp")
    val_temp.setDomainTransform(ranges)

    aPoints=val_temp.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])
    val_list = []    
    for j in range(theta_vals.shape[0]):
        config.theta = theta_vals[j]
        valnew=TasmanianSG.TasmanianSparseGrid()
        valnew=interpol_iter.sparse_grid_iter(n_agents, 1, valold)
        for k in range(refinement_level):
            valnew = interp_adap.sparse_grid_adap(n_agents, valnew, valold)
        valnew.write("valnew_1."+str(i + 1) + "shock" + str(j+1) +'.txt')
        val_list.append(valnew)
    
    for iI in range(iNumP1):
        for j in range(theta_vals.shape[0]):
            aVals[iI] += 1/5 * val_list[j].evaluate(aPoints[iI])
    val_temp.loadNeededPoints(aVals)
    for k in range(refinement_level):
        val_temp.setSurplusRefinement(fTol, 1 , "fds")
        aPoints = val_temp.getNeededPoints()
        aVals = np.empty([aPoints.shape[0], 1])
        iNumP1 = aVals.shape[0]
        for iI in range(iNumP1):
            for j in range(theta_vals.shape[0]):
                aVals[iI] += 1/5 * val_list[j].evaluate(aPoints[iI])
        val_temp.loadNeededPoints(aVals)
    #valold=TasmanianSG.TasmanianSparseGrid()
    valold = val_temp
    valold.write("valnew_1." + str(i+1) + ".txt")
    
#======================================================================
print "==============================================================="
print " "
print " Computation of a growth model of dimension ", n_agents ," finished after ", numits, " steps"
print " "
print "==============================================================="
#======================================================================

# compute errors   
avg_err=post.ls_error(n_agents, numstart, numits, No_samples)

#======================================================================
print "==============================================================="
print " "
print " Errors are computed -- see errors.txt"
print " "
print "==============================================================="
#======================================================================
