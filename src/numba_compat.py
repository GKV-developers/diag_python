#!/usr/bin/env python
# coding: utf-8
try:
    from numba import njit, jit, vectorize, guvectorize, prange
except ImportError:
    print("Numba not installed; using dummy decorators.")

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        else:
            return lambda f: f

    def jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        else:
            return lambda f: f

    def vectorize(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        else:
            return lambda f: f

    def guvectorize(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        else:
            return lambda f: f

    def prange(n):
        return range(n)
