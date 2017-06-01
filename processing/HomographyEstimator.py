#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
(c) L3i - Univ. La Rochelle
    joseph.chazalon (at) univ-lr (dot) fr

SmartDoc 2017 Evaluation Tools

Image warper
"""

# ==============================================================================
# Imports
import cv2
import numpy as np

from utils.log import *

# ==============================================================================
class HomographyEstimator(object):
    def __init__(self, debug=False):
        self._debug = debug
        self._logger = createAndInitLogger(__name__, debug)

    def estim_h_localdescr(self, from_img, to_img, second_match_tresh = 0.7, num_of_matches = 15):
        detector = cv2.SIFT(nfeatures=0,
                            nOctaveLayers=10,
                            contrastThreshold=0.04,
                            edgeThreshold=10.0,
                            sigma=1.6)
            
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        kpts_t, desc_t = detector.detectAndCompute(to_img,None)
        if len(kpts_t) < num_of_matches:
            self._logger.debug("(fail) Not enough keypoints in target image: %d (min %d)", 
                len(kpts_t), num_of_matches)
            return None

        matcher.add([desc_t]) 
        kpts_f, desc_f = detector.detectAndCompute(from_img,None)
        if len(kpts_f) < num_of_matches:
            self._logger.debug("(fail) Not enough keypoints in source image: %d (min %d)",
                len(kpts_f), num_of_matches)
            return None

        matches = matcher.knnMatch(desc_f, k = 2)
        matches = [m[0] for m in matches 
            if len(m) >= 2 
            and m[0].distance < m[1].distance * second_match_tresh]
        if len(matches) < num_of_matches:
            self._logger.debug("(fail) Not enough matches between images: %d (min %d)",
                len(matches), num_of_matches)
            return None

        pts_t = np.float32([kpts_t[m.trainIdx].pt for m in matches])
        pts_f = np.float32([kpts_f[m.queryIdx].pt for m in matches])
        H, s = cv2.findHomography(pts_f, pts_t, cv2.RANSAC, 3.0)
        s = s.ravel() != 0
        if s.sum() < num_of_matches:
            self._logger.debug("(fail) Not inliers after RANSAC: %d (min %d)",
                s.sum(), num_of_matches)
            return None

        return np.float32(H)
