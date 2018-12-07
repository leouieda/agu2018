"""
CV version of Spline
"""
import itertools
import numpy as np
from sklearn.utils.validation import check_is_fitted

from verde import cross_val_score, Spline
from verde.base import BaseGridder
from verde.model_selection import DummyClient


class SplineCV(BaseGridder):
    r"""
    Cross-validated version of the spline.
    """

    def __init__(
        self,
        dampings=(None, 1e-8, 1e-4, 1e-1),
        mindists=(1e3, 10e3, 100e3),
        force_coords=None,
        engine="auto",
        cv=None,
        client=None,
    ):
        self.dampings = dampings
        self.mindists = mindists
        self.force_coords = force_coords
        self.engine = engine
        self.cv = cv
        self.client = client

    @property
    def parameter_combinations(self):
        "All combinations of the parameters on the grid search"
        grid = itertools.product(self.dampings, self.mindists)
        combinations = [dict(zip(["damping", "mindist"], params)) for params in grid]
        return combinations

    def make_gridder(self, **parameters):
        """
        Instantiate a gridder given the parameters in the grid search
        """
        spline = Spline(
            force_coords=self.force_coords, engine=self.engine, **parameters,
        )
        return spline

    def fit(self, coordinates, data, weights=None):
        """
        Fit the gridder to the given data.

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
        scores = []
        for parameters in self.parameter_combinations:
            gridder = self.make_gridder(**parameters)
            scores.append(
                client.submit(
                    cross_val_score,
                    gridder,
                    coordinates=futures[0],
                    data=futures[1],
                    weights=futures[2],
                    cv=self.cv,
                )
            )
        if self.client is not None:
            scores = [score.result() for score in scores]
        self.scores_ = np.mean(scores, axis=1)
        best_parameters = self.parameter_combinations[np.argmax(self.scores_)]
        for parameter in best_parameters:
            setattr(self, parameter + "_", best_parameters[parameter])
        self.gridder_ = self.make_gridder(**best_parameters)
        self.gridder_.fit(coordinates, data, weights=weights)
        return self

    def predict(self, coordinates):
        """
        Evaluate the fitted gridder on the given set of points.
        """
        check_is_fitted(self, ["gridder_"])
        return self.gridder_.predict(coordinates)
