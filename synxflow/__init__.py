#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init
To do:
    initialize a package
Created on Wed Apr  1 14:56:15 2020

@author: Xilin Xia
"""

from .version import __version__
from . import flood, IO

__all__ = [
  '__version__',
  'flood',
  'IO',
]