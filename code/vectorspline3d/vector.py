"""
Green's functions and gridding for 3D vector elastic deformation
"""
import itertools
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree  # pylint: disable=no-name-in-module

import numba
from numba import jit

from verde import get_region, cross_val_score
from verde.base import check_fit_input, BaseGridder, least_squares
from verde.utils import n_1d_arrays
from verde.model_selection import DummyClient


# Default arguments for numba.jit
JIT_ARGS = dict(nopython=True, target="cpu", fastmath=True, parallel=True)


class VectorSpline3D(BaseGridder):
    r"""

    Parameters
    ----------
    poisson : float
        The Poisson's ratio for the elastic deformation Green's functions.
        Default is 0.5.
    depth : float
        The depth of the forces (should be a positive scalar). Data points are
        considered to be at 0 depth. Acts as the *mindist* parameter for
        :class:`verde.Spline` (a smoothing agent). A good rule of thumb is to use the
        average spacing between data points.
    damping : None or float
        The positive damping regularization parameter. Controls how much smoothness is
        imposed on the estimated forces. If None, no regularization is used.
    force_coords : None or tuple of arrays
        The easting and northing coordinates of the point forces. If None (default),
        then will be set to the data coordinates the first time
        :meth:`~verde.VectorSpline3D.fit` is called.

    Attributes
    ----------
    force_ : array
        The estimated forces that fit the observed data.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.VectorSpline3D.grid` and :meth:`~verde.VectorSpline3D.scatter`
        methods.

    """

    def __init__(
        self,
        poisson=0.5,
        depth=10e3,
        damping=None,
        coupling=1,
        force_coords=None,
        depth_scale="nearest",
    ):
        self.poisson = poisson
        self.depth = depth
        self.damping = damping
        self.coupling = coupling
        self.force_coords = force_coords
        self.depth_scale = depth_scale

    def fit(self, coordinates, data, weights=None):
        """
        Fit the gridder to the given 3-component vector data.

        The data region is captured and used as default for the
        :meth:`~verde.VectorSpline3D.grid` and :meth:`~verde.VectorSpline3D.scatter`
        methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : tuple of array
            A tuple ``(east_component, north_component, up_component)`` of
            arrays with the vector data values at each point.
        weights : None or tuple array
            If not None, then the weights assigned to each data point. Must be
            one array per data component. Typically, this should be 1 over the
            data uncertainty squared.

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.

        """
        coordinates, data, weights = check_fit_input(
            coordinates, data, weights, unpack=False
        )
        if len(data) != 3:
            raise ValueError(
                "Need three data components. Only {} given.".format(len(data))
            )
        # Capture the data region to use as a default when gridding.
        self.region_ = get_region(coordinates[:2])
        if any(w is not None for w in weights):
            weights = np.concatenate([i.ravel() for i in weights])
        else:
            weights = None
        data = np.concatenate([i.ravel() for i in data])
        if self.force_coords is None:
            self.force_coords = tuple(i.copy() for i in n_1d_arrays(coordinates, n=2))
        else:
            self.force_coords = n_1d_arrays(self.force_coords, n=2)
        if self.depth_scale is None:
            self._depth_scale = np.zeros_like(self.force_coords[0])
        elif self.depth_scale == "nearest":
            points = np.transpose(self.force_coords)
            nndist = np.median(KDTree(points).query(points, k=20)[0], axis=1)
            self._depth_scale = nndist - nndist.min()
        else:
            self._depth_scale = self.depth_scale
        jacobian = self.jacobian(coordinates[:2], self.force_coords)
        self.force_ = least_squares(jacobian, data, weights, self.damping)
        return self

    def predict(self, coordinates):
        """
        Evaluate the fitted gridder on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.VectorSpline3D.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.

        Returns
        -------
        data : tuple of arrays
            A tuple ``(east_component, north_component, up_component)`` of
            arrays with the predicted vector data values at each point.

        """
        check_is_fitted(self, ["force_"])
        force_east, force_north = n_1d_arrays(self.force_coords, n=2)
        east, north = n_1d_arrays(coordinates, n=2)
        cast = np.broadcast(*coordinates[:2])
        components = predict_3d_numba(
            east,
            north,
            force_east,
            force_north,
            self.depth + self._depth_scale,
            self.poisson,
            self.coupling,
            self.force_,
            np.empty(cast.size, dtype=east.dtype),
            np.empty(cast.size, dtype=east.dtype),
            np.empty(cast.size, dtype=east.dtype),
        )
        return tuple(comp.reshape(cast.shape) for comp in components)

    def jacobian(self, coordinates, force_coords, dtype="float64"):
        """
        Make the Jacobian matrix for the 3D coupled elastic deformation.

        The forces and data are assumed to be stacked into 1D arrays with the east
        component on top of the north component on top of the up component.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting and
            northing will be used, all subsequent coordinates will be ignored.
        force_coords : tuple of arrays
            Arrays with the coordinates for the forces. Should be in the same order as
            the coordinate arrays.
        dtype : str or numpy dtype
            The type of the Jacobian array.

        Returns
        -------
        jacobian : 2D array
            The (n_data*3, n_forces*3) Jacobian matrix.

        """
        force_east, force_north = n_1d_arrays(force_coords, n=2)
        east, north = n_1d_arrays(coordinates, n=2)
        shape = (east.size * 3, force_east.size * 3)
        jac = jacobian_3d_numba(
            east,
            north,
            force_east,
            force_north,
            self.depth + self._depth_scale,
            self.poisson,
            self.coupling,
            np.empty(shape, dtype=dtype),
        )
        return jac


def greens_func_3d(east, north, depth, poisson):
    "Calculate the Green's functions for the 3D elastic case."
    distance = np.sqrt(east ** 2 + north ** 2 + depth ** 2)
    # Pre-compute common terms for the Green's functions of each component
    over_r = 1 / distance
    over_rz = 1 / (distance + depth)
    aux = 1 - 2 * poisson
    g_ee = over_r * (
        1
        + (east * over_r) ** 2
        + aux * distance * over_rz
        - aux * (east * over_rz) ** 2
    )
    g_nn = over_r * (
        1
        + (north * over_r) ** 2
        + aux * distance * over_rz
        - aux * (north * over_rz) ** 2
    )
    g_uu = over_r * (1 + aux + (depth * over_r) ** 2)
    g_en = east * north * over_r * (over_r ** 2 - aux * over_rz ** 2)
    g_ne = g_en
    g_eu = east * over_r * (depth * over_r ** 2 - aux * over_rz)
    g_ue = east * over_r * (depth * over_r ** 2 + aux * over_rz)
    g_nu = north * over_r * (depth * over_r ** 2 - aux * over_rz)
    g_un = north * over_r * (depth * over_r ** 2 + aux * over_rz)
    return g_ee, g_en, g_eu, g_ne, g_nn, g_nu, g_ue, g_un, g_uu


# JIT compile the Greens functions for use in numba functions
GREENS_FUNC_3D_JIT = jit(**JIT_ARGS)(greens_func_3d)


@jit(**JIT_ARGS)
def predict_3d_numba(
    east,
    north,
    force_east,
    force_north,
    depth,
    poisson,
    coupling,
    forces,
    vec_east,
    vec_north,
    vec_up,
):
    """
    Calculate predicted data from the estimated forces.
    """
    nforces = forces.size // 3
    for i in numba.prange(east.size):  # pylint: disable=not-an-iterable
        vec_east[i] = 0
        vec_north[i] = 0
        vec_up[i] = 0
        for j in range(nforces):
            g_ee, g_en, g_eu, g_ne, g_nn, g_nu, g_ue, g_un, g_uu = GREENS_FUNC_3D_JIT(
                east[i] - force_east[j], north[i] - force_north[j], depth[j], poisson
            )
            vec_east[i] += (
                g_ee * forces[j]
                + g_en * forces[j + nforces]
                + coupling * g_eu * forces[j + 2 * nforces]
            )
            vec_north[i] += (
                g_ne * forces[j]
                + g_nn * forces[j + nforces]
                + coupling * g_nu * forces[j + 2 * nforces]
            )
            vec_up[i] += (
                coupling * g_ue * forces[j]
                + coupling * g_un * forces[j + nforces]
                + g_uu * forces[j + 2 * nforces]
            )
    return vec_east, vec_north, vec_up


@jit(**JIT_ARGS)
def jacobian_3d_numba(
    east, north, force_east, force_north, depth, poisson, coupling, jac
):
    """
        |J_ee J_en J_ev| |f_e| |d_e|
        |J_ne J_nn J_nv|*|f_n|=|d_n|
        |J_ve J_vn J_vv| |f_v| |d_v|
    """
    nforces = force_east.size
    npoints = east.size
    for i in numba.prange(npoints):  # pylint: disable=not-an-iterable
        for j in range(nforces):
            g_ee, g_en, g_eu, g_ne, g_nn, g_nu, g_ue, g_un, g_uu = GREENS_FUNC_3D_JIT(
                east[i] - force_east[j], north[i] - force_north[j], depth[j], poisson
            )
            jac[i, j] = g_ee
            jac[i, j + nforces] = g_en
            jac[i, j + 2 * nforces] = coupling * g_eu
            jac[i + npoints, j] = g_ne
            jac[i + npoints, j + nforces] = g_nn
            jac[i + npoints, j + 2 * nforces] = coupling * g_nu
            jac[i + 2 * npoints, j] = coupling * g_ue
            jac[i + 2 * npoints, j + nforces] = coupling * g_un
            jac[i + 2 * npoints, j + 2 * nforces] = g_uu
    return jac


class VectorSpline3DCV(BaseGridder):
    r"""
    Cross-validated version of the 3-component vector spline.

    Parameters
    ----------
    poisson : iterable (list, tuple, array)
        The Poisson's ratio for the elastic deformation Green's functions.
    depth : float
        The depth of the forces (should be a positive scalar). Data points are
        considered to be at 0 depth. Acts as the *mindist* parameter for
        :class:`verde.Spline` (a smoothing agent). A good rule of thumb is to use the
        average spacing between data points.
    damping : None or float
        The positive damping regularization parameter. Controls how much smoothness is
        imposed on the estimated forces. If None, no regularization is used.
    force_coords : None or tuple of arrays
        The easting and northing coordinates of the point forces. If None (default),
        then will be set to the data coordinates the first time
        :meth:`~verde.VectorSpline3D.fit` is called.

    Attributes
    ----------
    force_ : array
        The estimated forces that fit the observed data.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.VectorSpline3D.grid` and :meth:`~verde.VectorSpline3D.scatter`
        methods.

    """

    def __init__(
        self,
        poissons=(0, 0.25, 0.5),
        depths=(1e3, 10e3, 100e3),
        dampings=(None, 1e-3, 1e-1),
        couplings=(0, 1),
        depth_scale="nearest",
        force_coords=None,
        cv=None,
        client=None,
    ):
        self.poissons = poissons
        self.depths = depths
        self.dampings = dampings
        self.couplings = couplings
        self.force_coords = force_coords
        self.depth_scale = depth_scale
        self.cv = cv
        self.client = client

    def fit(self, coordinates, data, weights=None):
        """
        Fit the gridder to the given 3-component vector data.

        The data region is captured and used as default for the
        :meth:`~verde.VectorSpline3D.grid` and :meth:`~verde.VectorSpline3D.scatter`
        methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : tuple of array
            A tuple ``(east_component, north_component, up_component)`` of
            arrays with the vector data values at each point.
        weights : None or tuple array
            If not None, then the weights assigned to each data point. Must be
            one array per data component. Typically, this should be 1 over the
            data uncertainty squared.

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.

        """
        if self.client is None:
            client = DummyClient()
            futures = (coordinates, data, weights)
        else:
            client = self.client
            futures = tuple(client.scatter(i) for i in (coordinates, data, weights))
        combinations = list(
            itertools.product(self.dampings, self.depths, self.poissons, self.couplings)
        )
        scores = []
        for damping, depth, poisson, coupling in combinations:
            spline = VectorSpline3D(
                damping=damping,
                poisson=poisson,
                depth=depth,
                coupling=coupling,
                force_coords=self.force_coords,
                depth_scale=self.depth_scale,
            )
            scores.append(
                client.submit(
                    cross_val_score,
                    spline,
                    coordinates=futures[0],
                    data=futures[1],
                    weights=futures[2],
                    cv=self.cv,
                )
            )
        if self.client is not None:
            scores = [score.result() for score in scores]
        self.scores_ = np.mean(scores, axis=1)
        best = np.argmax(self.scores_)
        self.damping_ = combinations[best][0]
        self.depth_ = combinations[best][1]
        self.poisson_ = combinations[best][2]
        self.coupling_ = combinations[best][3]
        self.gridder_ = VectorSpline3D(
            damping=self.damping_,
            poisson=self.poisson_,
            depth=self.depth_,
            coupling=self.coupling_,
            force_coords=self.force_coords,
            depth_scale=self.depth_scale,
        )
        self.gridder_.fit(coordinates, data, weights=weights)
        return self

    def predict(self, coordinates):
        """
        Evaluate the fitted gridder on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.VectorSpline3D.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.

        Returns
        -------
        data : tuple of arrays
            A tuple ``(east_component, north_component, up_component)`` of
            arrays with the predicted vector data values at each point.

        """
        check_is_fitted(self, ["gridder_"])
        return self.gridder_.predict(coordinates)
