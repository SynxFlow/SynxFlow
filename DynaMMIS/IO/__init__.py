#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init
To do:
    initialize a package
Created on Wed Apr  1 14:56:15 2020

@author: Xiaodong Ming
"""
import numpy as np
from .demo_functions import demo_input
from .demo_functions import demo_output
from .demo_functions import demo_raster
from .demo_functions import get_sample_data
from .InputHipims import InputHipims, load_input_object
from .OutputHipims import OutputHipims, load_output_object
from .indep_functions import clean_output, _dict2grid
from .indep_functions import write_times_setup, write_device_setup, write_rain_source
from .Raster import Raster