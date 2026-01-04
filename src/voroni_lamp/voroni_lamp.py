from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry.polygon import Polygon
from scipy.spatial import Voronoi as Vor
from shapely.geometry import JOIN_STYLE
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

    n_cells: int = 40
    x_dim_mm: float = 200
    y_dim_mm: float = 400
    edge_mm: float = 5
    cell_pad_mm: float = 4
    bit_rad_mm: float = (25.4 / 8)  # 1/8"
    
    # Random sampling
    n_random_samples: int = 100000
    random_state: int | None = None
    
    """Generate Voroni visualization
    
    Cell: A hole in the design
    Edge: The sides of the rectangle
    
    
    n_cells (int): The number of holes to generate.
    x_dim_mm (float): The width of the rectangle.
    y_dim_mm (float): The height or the rectangle.
    edge_mm (float): The spacing between the edge and the polygons.
    cell_pad_mm (float): The spacing between cells.
    bit_rad_mm (float): The inner radius of the rounded polygons. Or, the router
    bit radius.
    """
    
    
    @cached_property
    @abstractmethod
    def random_samples(self) -> NDArray:
        """Generate the random samples used for kmeans clustering."""
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
    def flush_polygons(self) -> list[Polygon]:
        
        large_scale = 100
        large_x = self.x_dim_mm * large_scale
        large_y = self.y_dim_mm * large_scale
        
        large_points = np.array([
            (large_x, large_y),
            (large_x, -large_y),
            (-large_x, -large_y),
            (-large_x, large_y),
        ])
        
        vor = Vor(points=np.concatenate([self.centres, large_points]))

        buffer_clip = Polygon([
            (self.edge_mm, self.edge_mm),
            (self.edge_mm, self.y_dim_mm - self.edge_mm),
            (self.x_dim_mm - self.edge_mm, self.y_dim_mm - self.edge_mm),
            (self.x_dim_mm - self.edge_mm, self.edge_mm),
            (self.edge_mm, self.edge_mm),
            
        ])
        polygons = []
        for region in vor.regions:
            if len(region) == 0 or -1 in region:
                continue
            
            poly = Polygon(vor.vertices[region])
            
            # Clip to buffer
            poly = poly.intersection(buffer_clip)
            
                        
            polygons.append(poly)
            
            
        return polygons
            
    @cached_property
    def polygons(self) -> list[Polygon]:
        
        polygons = []
        for poly in self.flush_polygons:
            
            # Remove any complications of the polygon.
            poly = poly.buffer(distance=0)
            
            # Inset the polygons by the true cell pad amount and the radius of
            # the polygons. The polygon radius will be accounted for in the
            # offset. Since the cell pad is applied twice, apply half.
            inset_mm = (self.cell_pad_mm / 2) + self.bit_rad_mm
            poly = poly.buffer(distance= -inset_mm)
            
            # Give the polygon rounded corners.
            poly = poly.buffer(
                distance=self.bit_rad_mm,
                join_style=JOIN_STYLE.round,
                resolution=64,
            )
            
            # Store
            polygons.append(poly)

        return polygons
            
    @cached_property
    def n_polygons(self) -> int:
        return len(self.polygons)
        

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
            zorder=15,
            color='black',
            s=5,
            alpha=0.25,
        )
        
        # Polygons
        mpl_polys = [
            MplPolygon(list(poly.exterior.coords))
            for poly in self.polygons
        ]
        
        # Colors
        cmap = plt.get_cmap("viridis")
        cmap_ints = np.random.uniform(low=0.2, high=0.8, size=self.n_polygons)
        colors = [cmap(i) for i in cmap_ints]
        
        ax.add_collection(PatchCollection(
            mpl_polys,
            facecolor=colors,
            edgecolor='k',
            linewidth=0,
            alpha=0.7,
            zorder=10,
        ))
            
        # Remove all buffer
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_aspect('equal')
        ax.set_xlim(0, self.x_dim_mm)
        ax.set_ylim(0, self.y_dim_mm)
        
        return fig
        
        

        
@dataclass
class SampleVoroni(Voroni):
    
    rel_norm_scale: float = 0.2
    
    @cached_property
    def random_samples(self) -> NDArray:
        
        return np.stack([
            np.random.normal(
                loc=self.x_dim_mm / 2,
                scale=self.x_dim_mm * self.rel_norm_scale,
                size=self.n_random_samples,
            ),
            np.random.normal(
                loc=self.y_dim_mm / 2,
                scale=self.y_dim_mm * self.rel_norm_scale,
                size=self.n_random_samples,
            ),
        ]).T
        
        
    
if __name__ == "__main__":
    
    sv = SampleVoroni(
        rel_norm_scale=0.18,
    )
    sv.figure.savefig(Path(__file__).parent / 'demo.png')