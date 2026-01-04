from functools import cached_property
from matplotlib.figure import Figure
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from dataclasses import dataclass
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


@dataclass
class Voroni(ABC):

    n_cells: int = 150
    x_dim_mm: float = 200
    y_dim_mm: float = 300
    edge_mm: float = 5
    
    # Random sampling
    n_random_samples: int = 100000
    random_state: int | None = None
    
    
    @cached_property
    @abstractmethod
    def random_samples(self) -> NDArray:
        ...
    
    @cached_property
    def centres(
        self,
    ) -> NDArray:
        """Returns an array of x, y coordinates of where to place the centres
        for the lamp.

        Returns:
            NDArray: Array of shape (n, 2)
        """
        kmeans = KMeans(
            n_clusters=self.n_cells,
            random_state=self.random_state,
            init="k-means++",
            n_init=1,
            max_iter=100,
            algorithm="elkan",
        )
        kmeans.fit_predict(self.random_samples)
        return kmeans.cluster_centers_

    @cached_property
    def figure(self) -> Figure:
        
        ratio = self.y_dim_mm / self.x_dim_mm
        fig, ax = plt.subplots(
            figsize=(8, 8 * ratio),
            dpi=200,
        )
        
        # Scatter centres
        ax.scatter(
            self.centres[:, 0],
            self.centres[:, 1],
        )
        
        
        ax.set_aspect('equal')
        ax.set_xlim(0, self.x_dim_mm)
        ax.set_ylim(0, self.y_dim_mm)
        
        return fig
        
        

        
@dataclass
class SampleVoroni(Voroni):
    
    @cached_property
    def random_samples(self) -> NDArray:
        
        return np.stack([
            np.random.normal(
                loc=self.x_dim_mm / 2,
                scale=self.x_dim_mm / 8,
                size=self.n_random_samples,
            ),
            np.random.normal(
                loc=self.y_dim_mm / 2,
                scale=self.y_dim_mm / 8,
                size=self.n_random_samples,
            ),
        ]).T
        
        
    
if __name__ == "__main__":
    
    sv = SampleVoroni()
    sv.figure.savefig(Path(__file__).parent / 'demo.png')