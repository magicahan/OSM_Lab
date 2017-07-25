import TasmanianSG
import numpy as np
import os

# imports specifically needed by the examples
import math
from random import uniform
from datetime import datetime
import matplotlib.pyplot as plt

fTol = 1e-5
which_basis = 1

def non_adap(f, c, w, iDim, iOut, iDepth):
    grid  = TasmanianSG.TasmanianSparseGrid()
    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    aPoints = grid.getPoints()
    iNumP1 = aPoints.shape[0]
    aVals = np.empty([aPoints.shape[0], 1])
    for iI in range(aPoints.shape[0]):
        aVals[iI] = f(aPoints[iI], c, w)
    grid.loadNeededPoints(aVals)
    return grid

def adap(f, c, w, iDim, iOut, iDepth, refinement_level):
    grid  = TasmanianSG.TasmanianSparseGrid()
    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    aPoints = grid.getPoints()
    aVals = np.empty([aPoints.shape[0], 1])
    for iI in range(aPoints.shape[0]):
        aVals[iI] = f(aPoints[iI], c, w)
    grid.loadNeededPoints(aVals)
    for iK in range(refinement_level):
        grid.setSurplusRefinement(fTol, 1, "fds")   #also use fds, or other rules
        aPoints = grid.getNeededPoints()
        aVals = np.empty([aPoints.shape[0], 1])
        for iI in range(aPoints.shape[0]):
            aVals[iI] = f(aPoints[iI], c, w)
        grid.loadNeededPoints(aVals)
    return grid
    

def compare_method(f, c, w, iDim, iOut, iDepth, refinement_level):
    aPoints = np.random.uniform(-10, 10, (1000, iDim))
    print(aPoints)
    non_adap_error = []
    adap_error = []
    error_list_nonadap = []
    error_list_adap = []
    
    #for non adaptive method
    for k in range(iDepth):
        non_adap_error = []
        grid_nonadap = non_adap(f, c, w, iDim, iOut, k + 1)
        for i in range(aPoints.shape[0]):
            error = abs(grid_nonadap.evaluate(aPoints[i, :]) - f(aPoints[i, :],c, w))
            non_adap_error.append(error)
        non_adap_errormax = max(non_adap_error)
        error_list_nonadap.append(non_adap_errormax)
    
    for k in range(iDepth):
        adap_error = []
        grid_adap = adap(f, c, w, iDim, iOut, 1, k)
        for i in range(aPoints.shape[0]):
            error = abs(grid_adap.evaluate(aPoints[i, :]) - f(aPoints[i, :], c, w))
            adap_error.append(error)
        adap_errormax = max(adap_error)
        error_list_adap.append(adap_errormax)
    
    #plt.plot(list(range(iDepth)), error_list_nonadap, label = 'Non Adaptive')
    plt.plot(list(range(iDepth)), error_list_adap, label = 'Adaptive')
    plt.legend()
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "compare_method")
    plt.savefig(output_path)
    plt.close()    

def f(x, c, w):
    rv = np.cos(2 * np.pi * w[0] + np.dot(c, x))
    return rv
    

if __name__=='__main__':
    c = np.array([1, 2, 3, 4])
    w = np.array([1, 2, 3, 4])
    iDim = 4
    iOut = 1
    iDepth = 6
    refinement_level = iDepth - 1
    compare_method(f, c, w, iDim, iOut, iDepth, refinement_level)
