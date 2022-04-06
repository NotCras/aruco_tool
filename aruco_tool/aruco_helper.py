import cv2
from cv2 import aruco
import pdb
import pandas as pd
import numpy as np
import glob
from .corner_finder import CornerFinder
from .aruco_corner import ArucoCorner
from .aruco_loc import ArucoLoc

class ArucoHelper:
    """
    Helper class with static functions designed to help a user review or debug images with aruco data 
    """

    @staticmethod
    def load_corners(file_loc, id):
        """
        Loads corner data that was saved as a csv previously, returns an ArucoCorner obj with imported data
        """
        # import csv
        df = pd.read_csv(file_loc)  # TODO: should I parse the file_loc for information like id and folder loc?

        # get numpy array
        data = df.to_numpy()

        # reformat to aruco-style corners array
        data_len = len(data)

        # can't include the frame number that you get from pandas
        corners = np.reshape(data[:, 1:9], (data_len, 4, 2)) # TODO: need to double check I have right order
        
        return ArucoCorner(id, corners)


    @staticmethod
    def load_poses(file_loc, id):
        """
        Loads pose data that was saved as a csv previously, returns an ArucoLoc obj with imported data
        """
        df = pd.read_csv(file_loc)  # TODO: should I parse the file_loc for information like id and folder loc?
        
        # convert dataframe to numpy array, format is correct
        data = df.to_numpy()

        # reformat to aruco-style corners array
        data_len = len(data)

	    # can't include the frame number that you get from pandas
        poses = data[:, 1:9] # TODO: need to double check I have right order

        return ArucoLoc(id, data)


    @staticmethod
    def show_image(file_loc, include_corners=False, marker_size=3):
        """
        Show an image, can include the detected corners as red squares
        """
        img = cv2.imread(file_loc, cv2.IMREAD_COLOR)

        if include_corners:
            ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
            ar_params = aruco.DetectorParameters_create()

            cf = CornerFinder("")
            c_data = cf._analyze_single_image(file_loc, ar_dict, ar_params)

            for k in c_data.keys():

                for cs in c_data[k]:
                    x1 = cs[0]-marker_size
                    y1 = cs[1]+marker_size

                    x2 = cs[0]+marker_size
                    y2 = cs[1]-marker_size

                    # TODO: enable user to set color?
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), -1)
                
        cv2.imshow(f"{file_loc}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    @staticmethod
    def view_data_video(acode, include_corners=False):
        """
        Shows image stream as a video. Useful for debugging. Trying to only use opencv for this.
        Will ignore function for now...
        """
        try:
            # get folder location of the aruco code
            folder_loc = acode.folder_loc
        except:
            raise Exception("ArucoCorner object does not have a folder location associated with it")

        pass
    

    @staticmethod
    def camera_calibrate(folder_loc):
        """Helper script that runs camera calibration for opencv.

        Args:
            folder_loc (string): folder location that contains all the images for camera calibration. 
        """
        
        # Defining the dimensions of checkerboard
        CHECKERBOARD = (6,9)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = [] 


        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        # Extracting path of individual image stored in a given directory
        images = glob.glob(f"{folder_loc}/*.jpg")
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            
            cv2.imshow('img',img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

        h,w = img.shape[:2]

        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return mtx, dist

