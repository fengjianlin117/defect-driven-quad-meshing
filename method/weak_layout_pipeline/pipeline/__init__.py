"""Pipeline orchestration and shared mesh utilities."""
"""Weak-layout pipeline package.

Public workflows live in :mod:`run` and :mod:`generated`.  They are not eagerly
imported here because the direction-field adapter depends on ``pipeline.mesh``.
"""
