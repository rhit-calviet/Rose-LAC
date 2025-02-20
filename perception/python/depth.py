import numpy as np
import cv2 as cv

class DepthMap:

    def __init__(self, img_size, baseline, fx, fy, cx, cy, window_size=1, min_disp=0, max_disp=30):
        
        num_disp = max_disp*16
        block_size = window_size

        self.img_size = img_size
        self.baseline = baseline
        self.fx = fx

        self.K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
        R = np.eye(3)
        T = np.array([baseline, 0, 0])

        self.R1, self.R2, self.P1, self.P2, _, _, _ = cv.stereoRectify(
            self.K, None, self.K, None, img_size, R, T
        )

        self.stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=2,
            speckleRange=2,
            preFilterCap=5,
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
        disparity[disparity == 0] = 1

        disparity_normalized = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        self.disparityMapVis = disparity_normalized
        self.disparityMap = disparity

    def depth(self):
        disparity_actual = self.disparityMap.astype(np.float32) * (self.stereo.getNumDisparities() / 255.0)

        self.depthMap = (self.fx * self.baseline) / self.disparityMap
    
    def variance(self):
        # disparity_actual = self.disparityMap.astype(np.float32) * (self.stereo.getNumDisparities() / 255.0)

        # disparity_actual[disparity_actual <= 0] = np.nan

        sigma_d = 1.0

        sigma_Z = (self.fx * self.baseline / self.disparityMap**2) * sigma_d
        sigma_Z2 = sigma_Z**2

        self.varianceMap = sigma_Z2

    def vectorize(self):
        h, w = self.depthMap.shape

        y_coords, x_coords = np.indices((h, w))

        self.vectorizedMap = np.zeros((h, w, 4), dtype=np.float32)
        self.vectorizedMap[:, :, 0] = x_coords
        self.vectorizedMap[:, :, 1] = y_coords
        self.vectorizedMap[:, :, 2] = self.depthMap
        self.vectorizedMap[:, :, 3] = self.varianceMap

    def vectorize_list(self):
        h, w = self.depthMap.shape

        # Generate x, y coordinate indices
        y_coords, x_coords = np.indices((h, w))

        # Create two separate lists:
        vectorized_positions = np.column_stack((x_coords.ravel(), y_coords.ravel(), self.depthMap.ravel()))
        vectorized_variances = self.varianceMap.ravel()  # Flattened 1D array of variances

        self.vectorized_positions = vectorized_positions
        self.vectorized_variances = vectorized_variances

    def compute(self, imgLeft, imgRight):
        self.imgLeft = cv.imread(imgLeft, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(imgRight, cv.IMREAD_GRAYSCALE)

        # stereo_pair = np.hstack((self.imgLeft, self.imgRight))
        # small_stereo_pair = cv.resize(stereo_pair, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        
        # cv.imshow("Pre-rectified stereo images", small_stereo_pair)
        # cv.waitKey(0)

        # self.rectify()

        # stereo_pair = np.hstack((self.imgLeft, self.imgRight))
        # small_stereo_pair = cv.resize(stereo_pair, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)

        # cv.imshow("Post-rectified stereo images", small_stereo_pair)
        # cv.waitKey(0)

        self.disparity()

        # small_disparity_map = cv.resize(self.disparityMapVis, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        # cv.imshow("Disparity image", small_disparity_map)
        # cv.waitKey(0)

        self.depth()

        # small_depth_map = cv.resize(self.depthMap, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        # cv.imshow("Depth image", small_depth_map)
        # cv.waitKey(0)

        self.variance()

        # small_variance_map = cv.resize(self.varianceMap, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        # cv.imshow("Varance image", small_variance_map)
        # cv.waitKey(0)

        self.vectorize()

        self.vectorize_list()

        # print("Vector: ", self.vectorized[300, 512])


        # return self.depthMap, self.variance
    

    
