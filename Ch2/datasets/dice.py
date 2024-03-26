import nibabel as nb
import numpy as np
from argparse import ArgumentParser
import matplotlib
from matplotlib import pyplot as plt
import os
from math import floor


def dice_roi(input_dir, output_dir, atlas1, atlas2, verbose=True):
    """Calculates the dice coefficient for every ROI combination from atlas1 and atlas2

    Parameters
    ----------
    atlas1 : str
        path to first atlas to compare
    atlas2 : str
        path to second atlas to compare
    """

    #Create output name for png file
    yname = atlas1.split('_space-')[0]
    res=atlas1.split('space-MNI152NLin6_res-')[1]
    res=res.split('.nii')[0]
    xname = atlas2.split('_space-')[0]
    
    png_name=f"DICE_{yname}_&_{xname}_res-{res}"

    at1 = nb.load(f'{input_dir}/{atlas1}')
    at2 = nb.load(f'{input_dir}/{atlas2}')

    atlas1 = at1.get_fdata()
    atlas2 = at2.get_fdata()
    
    labs1 = np.unique(atlas1)
    labs2 = np.unique(atlas2)

    Dice = np.zeros((labs1.size, labs2.size))

    max_y=len(labs1)-1
    max_x=len(labs2)-1

    for i in range(len(labs1)):
        val1=labs1[i]
        for j in range(len(labs2)):
            val2=labs2[j]
            dice = np.sum(atlas1[atlas2==val2]==val1)*2.0 / (np.sum(atlas1[atlas1==val1]==val1) + np.sum(atlas2[atlas2==val2]==val2))
            
            Dice[int(i)][int(j)]=float(dice)

            if verbose:
                print(f'Dice coefficient for {yname} {i} of {max_y}, {xname} {j} of {max_x} = {dice}')

            if dice > 1 or dice < 0:
                raise ValueError(f"Dice coefficient is greater than 1 or less than 0 ({dice}) at atlas1: {val1}, atlas2: {val2}")

    return Dice, labs1, labs2