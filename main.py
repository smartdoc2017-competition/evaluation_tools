#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
(c) L3i - Univ. La Rochelle
    joseph.chazalon (at) univ-lr (dot) fr

SmartDoc 2017 Evaluation Tool

This is the main interface to the evaluation tool for the SmartDoc 2017 
competition.
"""

# ==============================================================================
# Imports
import logging
import argparse
import os
import os.path
import sys

from utils.log import *
from utils.args import *
from processing.EvalRestoration import EvalRestoration

# ==============================================================================
# Constants
PROG_VERSION = "1.0"
PROG_NAME = "SD17-eval"
PROG_DESCRIPTION = "SmartDoc 2017 Evaluation Tool"
EXITCODE_OK = 0
EXITCODE_KBDBREAK = 10
EXITCODE_IOERROR = 20
EXITCODE_UNKERR = 254

DBGLINELEN = 80
DBGSEP = "-"*DBGLINELEN

# ==============================================================================
# ==============================================================================
class Application(object):
    '''Main application class.'''
    def __init__(self):
        self._logger = createAndInitLogger(__name__)

    def main(self):
        '''Public main function.'''
        # Parse args
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=PROG_DESCRIPTION, 
            version=PROG_VERSION)
        parser.add_argument('-d', '--debug', 
            action="store_true", 
            help="Activate debug output.")
        parser.add_argument('-g', '--gui', 
            action="store_true", 
            help="Activate visualization.")
        parser.add_argument('ground_truth', 
            action=StoreValidFilePath,
            help='Path to `ground_truth.png` file.')
        parser.add_argument('participant_result', 
            action=StoreValidFilePath,
            help='Path to `sampleNN.png` file to evaluate.')
        parser.add_argument('-o', '--output', 
            help='Path to output evaluation file.')
        parser.add_argument('-x', '--extra-output', 
            action=StoreExistingOrCreatableDir,
            help='Path to directory where extra output files (debug) will be stored.')

        args = parser.parse_args()
        # activate debug?
        if args.debug:
            self._logger.setLevel(logging.DEBUG)
        # debug header
        self._logger.debug(DBGSEP)
        dbg_head = "%s - v. %s" % (PROG_NAME, PROG_VERSION)
        dbg_head_pre = " " * (max(0, (DBGLINELEN - len(dbg_head)))/2)
        self._logger.debug(dbg_head_pre + dbg_head)
        self._logger.debug(DBGSEP)
        self._logger.debug("Arguments:")
        for (k, v) in args.__dict__.items():
            self._logger.debug("    %-20s = %s" % (k, v))
        self._logger.debug(DBGSEP)
        # safely start processing
        try:
            self._logger.debug("Launching EvalRestoration")
            eval_resto = EvalRestoration(args.debug, args.gui)
            eval_resto.run(args.ground_truth, args.participant_result, args.output, args.extra_output)
            self._logger.debug("Processing complete.")
            return EXITCODE_OK
        except KeyboardInterrupt:
            self._logger.info("Process interrupted by user.")
            return EXITCODE_KBDBREAK
        except IOError:
            self._logger.exception("Problem in reading or writing file.")
            return EXITCODE_IOERROR
        except:
            self._logger.exception("Unknown error.")
            return EXITCODE_UNKERR

# ==============================================================================
if __name__ == "__main__":
    res = Application().main()
    if res is not None:
        sys.exit(res)
