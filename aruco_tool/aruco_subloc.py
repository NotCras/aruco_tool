import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb


class ArucoSubLoc(ArucoLoc):

    def __init__(self, id_num, poses, main_id, main_id_poses, data_attributes=None, file_folder=None):
        super.__init__(id_num, poses, data_attributes=data_attributes, file_folder=file_folder)

        self.main_id = main_id 
        self.main_id_poses = main_id_poses 


    def is_subloc():
        return True


    def yield_abs_poses():
        pass


    def gen_abs_poses_df():
        pass


    def gen_aruco_loc():
        pass


    def save_abs_poses():
        pass