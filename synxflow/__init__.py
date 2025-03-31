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

# Defer imports to avoid circular dependencies
def _load_modules():
    from . import flood, IO
    return flood, IO

__all__ = [
    '__version__',
    'flood',  # Will be populated by __getattr__
    'IO',     # Will be populated by __getattr__
]

# Lazy loading of modules
def __getattr__(name):
    if name in ['flood', 'IO']:
        flood, IO = _load_modules()
        globals()['flood'] = flood
        globals()['IO'] = IO
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")