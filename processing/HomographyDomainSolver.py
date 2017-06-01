#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
(c) L3i - Univ. La Rochelle
    joseph.chazalon (at) univ-lr (dot) fr

SmartDoc 2017 Evaluation Tools

Homography domain boundaries.

This modules computes how the homography domain can be determined given
the spatial domain on which the corners of a document can be projected to.

Adapted from 
https://www.coin-or.org/PuLP/CaseStudies/a_blending_problem.html
"""

# ==============================================================================
# Imports
from itertools import product
import os.path

from pulp import *
import numpy as np
import cv2

from utils.log import *


# ==============================================================================

class HomographyDomainSolver(object):
    def __init__(self, debug=False, create_files_in=None):
        self._debug = debug
        self._logger = createAndInitLogger(__name__, debug)
        self._create_files_in = create_files_in


    def _warp(self, coord, a1, a2, tx, a4, a5, ty, b1, b2):
        x, y = coord
        xp = (a1 * x + a2 * y + tx) / (b1 * x + b2 * y + 1.)
        yp = (a4 * x + a5 * y + ty) / (b1 * x + b2 * y + 1.)
        return xp, yp


    def compute_homography_domain(self, image_x_len, image_y_len, delta_x_max, delta_y_max, coordinates="OpenCV"):
        '''
        delta_x_max: 1/2 lenght of the variation domain in x direction

        Warning:
        Depending on the coordinate system we choose, the homography domain will be different, 
        and will be incompatible the other representation.
        - "OpenCV" means we use a coordinate system where the origin in on the
           corner of an image
        - "centered" means we use a coordinate system where the origin is at the center of
           the image
        '''
        logger = self._logger

        x_len = float(image_x_len)
        y_len = float(image_y_len)

        # the only constraint is that the resulting polygon is:
        # - made of 4 distinct points
        # - convex
        # - a non-degenerated quadrilateral (no subset of 3 points is aligned)
        # - non self-intersecting
        document_corners = None
        if "opencv" in coordinates.lower():
            document_corners = [
                      (0., 0.),
                      (0., y_len - 1.),
                      (x_len - 1.0, y_len - 1.),
                      (x_len - 1.0, 0.)]
        elif "center" in coordinates.lower():
            document_corners = [
                      (-x_len/2., -y_len/2.),
                      (-x_len/2.,  y_len/2.),
                      ( x_len/2.,  y_len/2.),
                      ( x_len/2., -y_len/2.)]
        else:
            msg = ("Invalid coordinate system selected for compute_homography_domain(): '%s'"
                        %coordinates)
            logger.error(msg)
            raise ValueError(msg)

        dx = abs(delta_x_max)
        dy = abs(delta_y_max)
        # boundaries = xmin, xmax, ymin, ymax for each corner
        boundaries = [(x - dx, x + dx, y - dy, y + dy) for (x, y) in document_corners]

        logger.debug("Document corners:")
        for c in document_corners:
            logger.debug("\t(%.0f, %.0f)" % c)
        logger.debug("Delta max: %.2fx%.2f", dx, dy)
        logger.debug("Boundaries:")
        for b in boundaries:
            logger.debug("\tx:[%.0f; %.0f]\ty:[%.0f; %.0f]" % b)


        theta_names = ["a1", "a2", "tx", "a4", "a5", "ty", "b1", "b2"]
        # 16 problems to solve: min and max for each var

        # Define an internal utility function
        def solve_pb(varname, minimize):
            if varname not in theta_names:
                raise ValueError("Bad component: %s" % varname)

            minmaxstr = "min" if minimize else "max"

            # Create the 'prob' variable to contain the problem data
            prob = LpProblem("Minimize %s" % varname, LpMinimize if minimize else LpMaximize)

            # A dictionary called 'ingredient_vars' is created to contain the referenced Variables
            theta_vars = LpVariable.dicts("theta", theta_names)

            # The objective function is added to 'prob' first
            prob += lpSum([theta_vars[varname]]), "%s %s value" % (varname, minmaxstr)

            # Adding constraints
            for ii, ((x, y), (xmin, xmax, ymin, ymax)) in enumerate(zip(document_corners, boundaries)):
                axmin = [x , y , 1., 0., 0., 0., -x*xmin, -y*xmin]
                axmax = [x , y , 1., 0., 0., 0., -x*xmax, -y*xmax]
                aymin = [0., 0., 0., x , y , 1., -x*ymin, -y*ymin]
                aymax = [0., 0., 0., x , y , 1., -x*ymax, -y*ymax]

                prob += lpSum([ai * theta_vars[ti] for (ai, ti) in zip(axmin, theta_names)]) >= xmin, "point %d, xmin" % ii
                prob += lpSum([ai * theta_vars[ti] for (ai, ti) in zip(axmax, theta_names)]) <= xmax, "point %d, xmax" % ii
                prob += lpSum([ai * theta_vars[ti] for (ai, ti) in zip(aymin, theta_names)]) >= ymin, "point %d, ymin" % ii
                prob += lpSum([ai * theta_vars[ti] for (ai, ti) in zip(aymax, theta_names)]) <= ymax, "point %d, ymax" % ii
                
                # # Test with a bug in the contraints
                # prob += x * theta_vars["b1"] + y * theta_vars["b2"] + 1.0 == 0.0, "point %d, bugbug" % ii

            # The problem data is written to an .lp file if needed
            if self._create_files_in is not None:
                prob.writeLP(os.path.join(
                    self._create_files_in, 
                    "h%s_%s.lp" % (minmaxstr, varname)))

            # The problem is solved using PuLP's choice of Solver
            prob.solve()

            # The status of the solution is printed to the screen
            # logger.debug("Status: %s", LpStatus[prob.status])
            if prob.status != 1: # 1 means "Optimal"
                logger.error("Problem is '%s' for variable '%s'.", LpStatus[prob.status], varname)
                return None # <<<<<<<<<<<<<<<

            # Each of the variables is printed with it's resolved optimum value
            a1 = a2 = tx = a4 = a5 = ty = b1 = b2 = 0.0
            for v in prob.variables():
                # logger.debug("%s = %s", v.name, v.varValue)
                if "_a1" in v.name: a1 = v.varValue
                if "_a2" in v.name: a2 = v.varValue
                if "_tx" in v.name: tx = v.varValue
                if "_a4" in v.name: a4 = v.varValue
                if "_a5" in v.name: a5 = v.varValue
                if "_ty" in v.name: ty = v.varValue
                if "_b1" in v.name: b1 = v.varValue
                if "_b2" in v.name: b2 = v.varValue

            # The optimised objective function value is printed to the screen    
            # logger.debug("%s %s value = %s", varname, minmaxstr, value(prob.objective))

            # print "Orig. polygon:", "\n\t".join([str(c) for c in document_corners])
            # Informative output
            # + Checks that (b1 * xi + b2 * yi + 1) != 0
            # logger.debug("New polygon:")
            # logger.debug("\t" + 
                # "\n\t".join([("(%.0f, %.0f)" 
                #     % self._warp(c, a1, a2, tx, a4, a5, ty, b1, b2)) for c in document_corners]))

            return value(prob.objective)
        ### / solve_pb ########################################

        defaults = {
                "a1": (1.0, 1.0),
                "a2": (0.0, 0.0),
                "tx": (0.0, 0.0),
                "a4": (0.0, 0.0),
                "a5": (1.0, 1.0),
                "ty": (0.0, 0.0),
                "b1": (0.0, 0.0),
                "b2": (0.0, 0.0)}

        results = {}
        for vv in theta_names:
            logger.debug("Solving %s domain", vv)
            vmin = solve_pb(vv, minimize=True)
            vmax = solve_pb(vv, minimize=False)
            if vmin is None or vmax is None:
                logger.error("Optimization problem ill-defined.")
                logger.warning("Will return default domain (identity matrix).")
                return defaults # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            results[vv] = (vmin, vmax)

        logger.debug("Found domain:")
        for vv in theta_names:
            logger.debug("\t%s: %s", vv, results[vv])
        return results

    # TODO cross-check results with OpenCV
    # IF COORDINATES ARE IMAGE-CENTERED -- but we may want something different or avoid conversions
    # a1 and a5 domains should be centered on 1.0 and within a few units
    # a2 and a4 domains should be centered on 0.0 and within a few units
    # tx and ty domains should be centered on 0.0 and large (pixel values)
    # b1 and b2 domains should be centered on 0.0 and very small


