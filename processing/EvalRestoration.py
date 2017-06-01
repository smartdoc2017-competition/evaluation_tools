#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
(c) L3i - Univ. La Rochelle
    joseph.chazalon (at) univ-lr (dot) fr

SmartDoc 2017 Evaluation Tools

Evaluation core logic.
"""

# ==============================================================================
# Imports
import json
from collections import namedtuple
import os.path
import datetime
import json

import cv2
import numpy as np
from skimage.measure import compare_ssim, compare_mse, compare_nrmse

from utils.log import *
from processing.HomographyDomainSolver import HomographyDomainSolver
from processing.HomographyEstimator import HomographyEstimator

# ==============================================================================
# Constants
GLOBAL_TOLERENCE_FACTOR = 0.01
LOCAL_TOLERENCE_FACTOR = 0.05
LOCAL_BLOCK_SIZE = 256 # Try 512?

OUTPUT_FORMAT_VERSION = 1.0

# ==============================================================================
class EvalRestoration(object):
    '''
    Full-reference quality assessment of some restored image against a 
    ground truth (reference) image.
    '''
    def __init__(self, debug=False, activate_gui=False):
        self._debug = debug
        self._logger = createAndInitLogger(__name__, debug)
        self._gui = activate_gui


    def _show_image(self, window, image):
        if self._gui:
            cv2.imshow(window, image)
            key_code = cv2.waitKey(20)
            if key_code & 0xff == ord('q'):
                raise KeyboardInterrupt()


    # def _overlay_poly(self, image, poly):
    #     if self._gui:
    #         cv2.polylines(image, [np.int32(poly)], True, (0, 255, 0), 2)
    #         for name, pt in zip(("TL", "BL", "BR", "TR"), poly):
    #             cv2.putText(image, name, (int(pt[0]), int(pt[1])),
    #                 cv2.FONT_HERSHEY_PLAIN, 2, (64, 255, 64), 2)


    def _check_open_image(self, filename):
        if not os.path.isfile(filename):
            raise IOError("'%s' does not exist, or is not a file." % filename)

        img = cv2.imread(filename) # 8-bit, no alpha
        # convert to grayscale if needed
        gray = img
        if gray.ndim > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray


    def _clip_homography(self, H, domain):
        '''Clips H to the domain allowed.'''
        result = H.flatten()
        for ii, vv in enumerate(["a1", "a2", "tx", "a4", "a5", "ty", "b1", "b2"]):
            vmin, vmax = domain[vv]
            result[ii] = np.max([vmin, np.min([vmax, result[ii]])])
        return result.reshape(3,3)


    def _mask_d_t(self, h_pr_to_gt, img_pr): # NOTE: OCV coordinates
        '''Creates a bitmap mask for GT region covered after transforming PR with H.'''
        x_len, y_len = img_pr.shape[1], img_pr.shape[0]
        img_pr_poly = np.float32([[0, 0],
                       [0, y_len-1],
                       [x_len-1, y_len-1],
                       [x_len-1, 0]])   
        mask = np.zeros(img_pr.shape, dtype=np.float32)
        D_t = cv2.perspectiveTransform(
            img_pr_poly.reshape(1, -1, 2), 
            h_pr_to_gt)
        D_t_poly = np.int32(D_t).reshape(-1, 2)
        cv2.fillPoly(mask, [D_t_poly], 1.0)
        # contract by 1 pixel?
        return mask

 
    def run(self, ground_truth_path, participant_result_path, output_path=None, extra_output_path=None):
        '''
        Main function for the evaluation.
        '''

        # TODO check that global SSIM is always lower than the pooling over local SSIM
        #      so we don't make evaluation worse with local refinements


        logger = self._logger

        # Init GUI if needed
        win_gt_orig = "GT Original"
        win_pr_orig = "PR Original"
        win_pr_global_warp = "PR Global Adjust."
        win_pr_local_warp = "PR local Adjust."
        win_mask_global_warp = "Mask for SSIM Global Warped"
        win_pr_local_warp = "PR Local Adjust."
        win_ssim_global_orig = "SSIM Global Original"
        win_ssim_global_warp = "SSIM Global Warped"
        win_ssim_global_warp_masked = "SSIM Global Warped with domain filter"
        win_ssim_local_warp = "SSIM Local Warped"
        win_ssim_local_warp_masked = "SSIM local Warped with domain filter"
        if self._gui:
            for win in (win_gt_orig, win_pr_orig, win_pr_global_warp, win_pr_local_warp, 
                        win_mask_global_warp, win_pr_local_warp, win_ssim_global_orig, 
                        win_ssim_global_warp, win_ssim_global_warp_masked, win_ssim_local_warp,
                        win_ssim_local_warp_masked):
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        # Check both files exist and open them
        img_gt = self._check_open_image(ground_truth_path)
        img_pr = self._check_open_image(participant_result_path)

        # Check both images have the exact same shape, fail otherwise
        y_len, x_len = img_gt.shape
        y_len_pr, x_len_pr = img_pr.shape
        if x_len != x_len_pr or y_len != y_len_pr:
            msg = "Bad shape for image under test: %dx%d (expected %d:%d)" % (y_len_pr, x_len_pr, y_len, x_len)
            logger.error(msg)
            raise ValueError(msg)

        logger.debug("Image target shape: x:%d; y:%d", x_len, y_len)
 


        # 1/ Global evaluation without any alignment
        logger.debug("** Computing global SSIM. **")
        ssim_data_range = img_gt.max() - img_gt.min()
        mssim_global_orig, ssim_map_global_orig = \
            compare_ssim(img_gt, img_pr, data_range=ssim_data_range, full=True)
        logger.info("Mean SSIM, global without registration adjustment: %f", mssim_global_orig)

        self._show_image(win_gt_orig, img_gt)
        self._show_image(win_pr_orig, img_pr)
        self._show_image(win_ssim_global_orig, ssim_map_global_orig)



        # 2/ Global registration adjustment and evaluation
        # Compute global boundaries of the domain for the global adjustment homography.
        logger.debug("** Computing H domain. **")
        h_dom_solv = HomographyDomainSolver(self._debug, extra_output_path)
        h_global_domain = h_dom_solv.compute_homography_domain(
            x_len, y_len, GLOBAL_TOLERENCE_FACTOR * x_len, GLOBAL_TOLERENCE_FACTOR * y_len, coordinates="OpenCV")
        # Dom(H) will be constrained to identity if there is any issue here.

        logger.debug("** Global registration adjustment. **")
        h_estimator = HomographyEstimator(debug=self._debug)
        h_global = h_estimator.estim_h_localdescr(img_pr, img_gt) # PR to GT
        if h_global is None:
            logger.warning("Could not find an homography between images using local descriptors.")
            logger.warning("\tUsing default homography.")
            h_global = np.eye(3, dtype=np.float32)

        logger.debug("Homography for global alignment:")
        for i in range(h_global.shape[0]):
            logger.debug("\t%s", h_global[i,:])
        h_global_clip = self._clip_homography(h_global, h_global_domain)
        if not np.allclose(h_global, h_global_clip):
            logger.warning("The refinement of the global alignment is out of domain.")
            logger.warning("\tAn extra cost will be counted for missed pixels.")
            logger.debug("Clipped homography for global alignment:")
            for i in range(h_global_clip.shape[0]):
                logger.debug("\t%s", h_global_clip[i,:])
        img_pr_warped_glob = cv2.warpPerspective(img_pr, h_global_clip, (x_len, y_len))

        mssim_global_warped, ssim_map_global_warped = \
            compare_ssim(img_gt, img_pr_warped_glob, data_range=ssim_data_range, full=True)
        logger.debug("Mean SSIM (no domain correction), global WITH registration adjustment: %f", mssim_global_warped)

        # We want to tolerate small adjustments because what matters is the actual
        # correlation between the images.
        # We thus create a simple mask to filter out missing pixels
        mask_global_warp = self._mask_d_t(h_global_clip, img_pr)
        ssim_map_global_warped_masked = ssim_map_global_warped * mask_global_warp
        mssim_global_warped_dom = np.sum(ssim_map_global_warped_masked) / np.sum(mask_global_warp)
        logger.info("Mean SSIM (WITH domain correction), global WITH registration adjustment: %f", mssim_global_warped_dom)

        self._show_image(win_mask_global_warp, mask_global_warp)
        self._show_image(win_pr_global_warp, img_pr_warped_glob)
        self._show_image(win_ssim_global_warp, ssim_map_global_warped)
        self._show_image(win_ssim_global_warp_masked, ssim_map_global_warped_masked)


        # 3/ Local registration adjustment and evaluation
        # select best baseline (based on mean ssim)
        baseline_image = img_pr_warped_glob
        baseline_ssim_map = ssim_map_global_warped_masked
        baseline_ssim_map_mask = mask_global_warp
        baseline_choice = ""
        if mssim_global_orig < mssim_global_warped_dom:
            baseline_choice = "warped SSIM map"
            logger.info("Selecting warped SSIM map as baseline for local optimization.")
        else:
            logger.info("Selecting original SSIM map as baseline for local optimization.")
            baseline_choice = "original SSIM map"
            baseline_image = img_pr
            baseline_ssim_map = ssim_map_global_orig
            baseline_ssim_map_mask = np.ones_like(mask_global_warp)


        # compute domain for H at local level
        h_local_domain = h_dom_solv.compute_homography_domain(
            LOCAL_BLOCK_SIZE, LOCAL_BLOCK_SIZE, 
            LOCAL_TOLERENCE_FACTOR * LOCAL_BLOCK_SIZE, LOCAL_TOLERENCE_FACTOR * LOCAL_BLOCK_SIZE, 
            coordinates="OpenCV")
        # Dom(H) will be constrained to identity if there is any issue here.

        logger.debug("** Local registration adjustment. **")
        # This map will be constructed progressively by stiching local results
        ssim_map_local_merged = np.zeros_like(baseline_ssim_map_mask)
        img_pr_warped_local = np.zeros_like(img_pr) # just for fun if extra output requested
        # Sliding window for local optimization
        for ys in range(0, y_len - LOCAL_BLOCK_SIZE/2, LOCAL_BLOCK_SIZE/2):
            for xs in range(0, x_len - LOCAL_BLOCK_SIZE/2, LOCAL_BLOCK_SIZE/2):
                # logger.error("(BLOCK [%d:%d, %d:%d]) ",
                #     ys, ys+LOCAL_BLOCK_SIZE, xs, xs+LOCAL_BLOCK_SIZE)

                window_gt = img_gt[ys:ys+LOCAL_BLOCK_SIZE, xs:xs+LOCAL_BLOCK_SIZE]
                window_bl = baseline_image[ys:ys+LOCAL_BLOCK_SIZE, xs:xs+LOCAL_BLOCK_SIZE]
                # baseline_ssim_map is already masked
                # window below contains ssim(window_gt, window_bl)
                ssim_map_local_orig = baseline_ssim_map[ys:ys+LOCAL_BLOCK_SIZE, xs:xs+LOCAL_BLOCK_SIZE]
                mssim_local_orig = np.mean(ssim_map_local_orig)
                #<<<<< 1

                # Local optimisation
                # (we need real shapes here)
                x_len_local, y_len_local = window_bl.shape[1], window_bl.shape[0]
                h_local = h_estimator.estim_h_localdescr(window_bl, window_gt, num_of_matches=10) # PR to GT
                if h_local is None:
                    logger.debug("(local [%d:%d, %d:%d]) " 
                        "Could not find an homography between images using local descriptors.",
                        xs, xs+LOCAL_BLOCK_SIZE, ys, ys+LOCAL_BLOCK_SIZE)
                    logger.debug("\tUsing default homography.")
                    h_local = np.eye(3, dtype=np.float32)

                # logger.debug("Homography for local alignment:")
                # for i in range(h_local.shape[0]):
                #     logger.debug("\t%s", h_local[i,:])
                h_local_clip = self._clip_homography(h_local, h_local_domain)
                if not np.allclose(h_local, h_local_clip):
                    logger.debug("(local [%d:%d, %d:%d]) "
                        "The refinement of the local alignment is out of domain.",
                        xs, xs+LOCAL_BLOCK_SIZE, ys, ys+LOCAL_BLOCK_SIZE)
                    logger.debug("\tAn extra cost will be counted for missed pixels.")
                    logger.debug("Clipped homography for local alignment:")
                    for i in range(h_local_clip.shape[0]):
                        logger.debug("\t%s", h_local_clip[i,:])
                window_bl_warped_loc = cv2.warpPerspective(window_bl, h_local_clip, (x_len_local, y_len_local))

                mssim_local_warped, ssim_map_local_warped = \
                    compare_ssim(window_gt, window_bl_warped_loc, data_range=ssim_data_range, full=True)
                logger.debug("Mean SSIM (no domain correction), local WITH registration adjustment: %f", mssim_local_warped)

                mask_local_warp = self._mask_d_t(h_local_clip, window_bl) 
                ssim_map_local_warped_masked = ssim_map_local_warped * mask_local_warp
                mssim_local_warped_dom = np.sum(ssim_map_local_warped_masked) / np.sum(mask_local_warp)
                logger.debug("Mean SSIM (WITH domain correction), local WITH registration adjustment: %f", mssim_local_warped_dom)
                #<<<<< 2

                # Select best option
                ssim_map_best_local = ssim_map_local_warped_masked
                img_pr_best_local = window_bl_warped_loc
                if mssim_local_warped_dom < mssim_local_orig:
                    ssim_map_best_local = ssim_map_local_orig
                    img_pr_best_local = window_bl

                # Then max-blend to the big ssim map of all local best
                previous = ssim_map_local_merged[ys:ys+LOCAL_BLOCK_SIZE, xs:xs+LOCAL_BLOCK_SIZE]
                ssim_map_local_merged[ys:ys+LOCAL_BLOCK_SIZE, xs:xs+LOCAL_BLOCK_SIZE] = \
                    np.amax(np.dstack((previous, ssim_map_best_local)), axis=-1)

                # Show updated image
                self._show_image(win_ssim_local_warp, ssim_map_local_merged)

                if extra_output_path is not None:
                    # and for debug also produce global warped image
                    img_pr_warped_local[ys:ys+LOCAL_BLOCK_SIZE, xs:xs+LOCAL_BLOCK_SIZE] = \
                        img_pr_best_local
                    self._show_image(win_pr_local_warp, img_pr_warped_local)

        # global mssim is ssim_map_local_merged * baseline_ssim_map_mask
        ssim_map_local_merged_mask = ssim_map_local_merged * baseline_ssim_map_mask
        mssim_local_merged_dom = np.sum(ssim_map_local_merged_mask) / np.sum(baseline_ssim_map_mask)
        logger.info("Mean SSIM (WITH domain correction), local WITH registration adjustment: %f", mssim_local_merged_dom)
        self._show_image(win_ssim_local_warp_masked, ssim_map_local_merged_mask)

        # NOTE: we could perform a ssim comparison on img_pr_warped_local vs img_gt
        #       but the blending is destructive for this image 
        #       so it would not make sense


        # 4/ Select best response (based on mssim) and output results

        final_result = {
            # info strings
            "format_version": OUTPUT_FORMAT_VERSION,
            "software": "SD17_EvalRestoration",
            "software_version": "1.0",
            "creation_date": datetime.datetime.now().isoformat(), # warning: no TZ info here
            "ground_truth_file": ground_truth_path,
            "participant_result_file": participant_result_path,
            "image_variant_used_for_local_optim": baseline_choice,
            # result values
            "mssim_global_orig": float(mssim_global_orig),
            "mssim_global_registered": float(mssim_global_warped_dom),
            "mssim_local_registered": float(mssim_local_merged_dom),
            # final value
            "mssim_best": float(max(mssim_global_orig, mssim_global_warped_dom, mssim_local_merged_dom))
        }

        with open(output_path, "wb") as out_file:
            json.dump(final_result, out_file, indent=2)
            logger.info("Results wrote to '%s", output_path)
        

        # 5/ Optional output of intermediate files

        # str:name, np.array2D:img, bool:should_scale
        img_output_list = [
            ("img_gt.png", img_gt, False),
            ("img_pr.png", img_pr, False),
            ("img_pr.png", img_pr, False),
            ("img_pr_warped_local.png", img_pr_warped_local, False),
            ("ssim_map_global_orig.png", ssim_map_global_orig, True),
            ("ssim_map_global_warped_masked.png", ssim_map_global_warped_masked, True),
            ("ssim_map_local_merged_mask.png", ssim_map_local_merged_mask, True)]

        if extra_output_path is not None:
            for name, image, should_scale in img_output_list:
                out_path = os.path.join(extra_output_path, name)
                image_pngready = image
                if should_scale:
                    image_pngready *= 255
                image_pngready = np.uint8(image_pngready)
                cv2.imwrite(out_path, image_pngready)
                logger.info("Wrote work image to '%s'", output_path)


        ################################
        logger.info("Process complete.")
        # wait until user quits if GUI is active
        if self._gui:
            # Wait for key press at the end of the process.
            logger.info("Please press any of the following keys to exit:")
            logger.info("\t SPACE, ESC, Q, ENTER")
            should_quit = False
            exit_keys = [32, 27, ord('q'), 13]
            while not should_quit:
                key_code = cv2.waitKey(100) & 0xff
                should_quit = key_code in exit_keys
        logger.debug("EvalRestoration end.")
    # / EvalRestoration.run()
