import cv2 as cv
import numpy as np

class DepthMap:

    imgLeft, imgRight, disparityMap, depthMap, variance = None

    def __init__(self, img_size, baseline, f, cx, cy, window_size=5, min_disp=14, num_disp=16*10, block_size=15):
        self.img_size = img_size
        self.baseline = baseline
        self.f = f

        self.K = np.array([[f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1]])
        R = np.eye(3)
        T = np.array([baseline, 0, 0])

        self.R1, self.R2, self.P1, self.P2 = cv.stereoRectify(
            self.K, None, self.K, None, img_size, R, T
        )

        self.stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    def rectify(self):
        map1_x, map1_y = cv.initUndistortRectifyMap(self.K, None, self.R1, self.P1, self.img_size, cv.CV_32FC1)
        map2_x, map2_y = cv.initUndistortRectifyMap(self.K, None, self.R2, self.P2, self.img_size, cv.CV_32FC1)

        rectified_left = cv.remap(self.imgLeft, map1_x, map1_y, cv.INTER_LINEAR)
        rectified_right = cv.remap(self.imgRight, map2_x, map2_y, cv.INTER_LINEAR)
        
        self.imgLeft = rectified_left
        self.imgReft = rectified_right

    def disparity(self):
        disparity = self.stereo.compute(self.imgLeft, self.imgRight)    
        
        disparity_normalized = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)

        self.disparityMap = disparity_normalized

    def depth(self):
        disparity_actual = self.disparityMap.astype(np.float32) * (self.stereo.getNumDisparities() / 255.0)

        disparity_actual[disparity_actual <= 0] = np.nan

        self.depthMap = (self.f * self.baseline) / disparity_actual
    
    def variance(self):
        disparity_actual = self.disparityMap.astype(np.float32) * (self.stereo.getNumDisparities() / 255.0)

        disparity_actual[disparity_actual <= 0] = np.nan

        sigma_d = 1.0

        sigma_Z = (self.f * self.baseline / disparity_actual**2) * sigma_d
        sigma_Z2 = sigma_Z**2

        self.variance = sigma_Z2

    def compute(self, imgLeft, imgRight):
        self.imgLeft = imgLeft
        self.imgRight = imgRight

        self.rectify()
        self.disparity()
        self.depth()
        self.variance()

        return self.depthMap, self.variance
    

    
