
import numpy as np
import igl
import meshplot as mp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay #used Scipy triangulation because igl triangulation is not working

class Remesher: 
    def __init__(self, v,f, meshname = "", uniform_coeff = .8, grid_size=100):
        """ When instantiated, the class will execute the harmonic parameterization, 
            compute the color maps, set up the grid, and compute the pixel colors. 
            The user will then be able to call the remesher object to plot the mesh,
            plot the pixel colors, and remesh the mesh using a dithering algorithm.
        """
        self.v = v
        self.f = f
        self.meshname = meshname
        # List of maps we will fill 
        self.embedding_map = None # maps 2d point to 3d point 
        self.h = None             # harmonic map from parameterization 
        self.coords2d = None      # 2d coordinates of the vertices in the mesh
        self.color_map = None     # current color map for the mesh (i.e each face has a color)
        self.uniform_coeff = uniform_coeff # coefficient for the uniform map
        print(f"computing harmonic parameterization for {self.v.shape[0]} vertices and {self.f.shape[0]} faces")
        self.harmonic(plot=True)
        print(f"computing area maps")
        self.compute_area_maps() # computes the area maps, and sets the default map to log map
        print(f"computing pixel colors: \n   Building grid \n   Getting face indices \n   Getting pixel colors")
        self.build_grid(grid_size) # build the grid for the pixels 
        self.get_face_indices() # get the face indices for the pixels
        self.get_pixel_colors()
        print(f"Finished instantiating. Your default color map is set to: ", self.current_map)
        
        
    def harmonic(self, plot=True, wf = False):
        """ Computing Harmonic Parametrization """
        # Get boundaries 
        b = igl.boundary_loop(self.f) # #vv by 3 array of boundary vertices 
        # Map mesh boundary to circle 
        uv = igl.map_vertices_to_circle(self.v,b) # uv = #w by 2 list of 2d positions of the vertices on the boundary
        # Caculate Harmonic map
        h = igl.harmonic(self.v, self.f, b, uv, 1) # harmonic map of the mesh to the circle -- #V by #W list of weights
        # Plot the mesh and the harmonic map
        if plot:
            p2 = mp.subplot(self.v, self.f, uv=h, s=[1, 2, 0], shading={"wireframe": wf, "flat": False})
            mp.subplot(h, self.f, s=[1, 2, 1], data=p2, shading={"wireframe": True})
        self.h = h
        self.coords2d = h[self.f]
        # return h, h[self.f]

    def compute_area_maps(self):
        v = self.v
        f = self.f
        h = self.h
        epsilon = 1e-6  # Small constant to avoid log(0)
        area_3d = igl.doublearea(v, f)
        area_2d = igl.doublearea(h, f)
        distort = area_3d/area_2d # distortion per face = ratio of the areas 
        norm_distort = (distort - np.min(distort)) / (np.max(distort) - np.min(distort)) # Normalize distortion to [0, 1]
        log_distort = np.log(distort + epsilon)
        normalized_log_distort = (log_distort - np.min(log_distort)) / (np.max(log_distort) - np.min(log_distort))
        self.log_map = 1-normalized_log_distort
        self.area_map = 1-distort
        self.area_map_norm = 1-norm_distort
        self.uniform_map = np.ones_like(distort) * self.uniform_coeff # uniform map -- all values = .5
        self.set_map(self.log_map, "LOG") # default set the log map

    def set_map(self, map, map_type):
        self.color_map = map
        self.current_map = map_type


    def compute_3d_from_2d(self, u):
        if (self.coords2d is None):
            raise ValueError("You need to compute the 2d coordinates first")
        u = np.array(u).flatten()
        u_3d = np.array([u[0], u[1], 0.0])
        threshold = 1e-9
        for i in range(self.f.shape[0]):
            u1 = self.coords2d[i, 0]
            u2 = self.coords2d[i, 1]
            u3 = self.coords2d[i, 2]
            a = np.array([u1[0], u1[1], 0.0])
            b = np.array([u2[0], u2[1], 0.0])
            c = np.array([u3[0], u3[1], 0.0])
            bc = igl.barycentric_coordinates_tri(np.array([u_3d]), np.array([a]), np.array([b]), np.array([c]))

            if np.all(bc >= threshold) and np.isclose(np.sum(bc), 1.0, atol=threshold):
                    v_indices = self.f[i]
                    x1 = self.v[v_indices[0]]
                    x2 = self.v[v_indices[1]]
                    x3 = self.v[v_indices[2]]
                    x = bc[0] * x1 + bc[1] * x2 + bc[2] * x3
                    return x

        return None # Return None if the 2D point is not found

    def color_to_3d(self, map):
        colors = np.stack([map,
                   map,
                   map], axis=-1)
        colors[self.get_invalid_color_idx(map)] = [1, 0, 0]
        return colors
    
    def plot(self, map=None, wf=False):
        if map is None:
            map = self.color_map
        c = self.color_to_3d(map)
        # print(f"Plotting with {self.current_map} map")
        p1 = mp.subplot(self.v, self.f, c=c, s=[1, 2, 0], shading={"wireframe": wf, "flat": False})
        mp.subplot(self.h, self.f, s=[1, 2, 1], data=p1,  c=c, shading={"wireframe": wf})
        return p1
    
    def build_grid(self, grid_size=100):
        min_u, min_v = np.min(self.h, axis=0)
        max_u, max_v = np.max(self.h, axis=0)
        eps = 0  # 1e-9
        u_coords = np.linspace(min_u + eps, max_u - eps, grid_size)
        v_coords = np.linspace(min_v + eps, max_v - eps, grid_size)
        uu, vv = np.meshgrid(u_coords, v_coords)
        zeros = np.zeros_like(uu)

        self.pixels = np.stack((uu.flatten(), vv.flatten(), zeros.flatten()), axis=-1)
        print(f"Built a {grid_size} by {grid_size} grid. Flattened shape: {self.pixels.shape}")
        self.uu = uu
        self.vv = vv
        self.grid_size = grid_size

    def get_face_indices(self):
        """ Gets the index of face corresponding to 2d grid points """
        face_index = []
        a_triangles = self.coords2d[:, 0, :]
        b_triangles = self.coords2d[:, 1, :]
        c_triangles = self.coords2d[:, 2, :]
        print(f"a_triangles.shape: {a_triangles.shape}")
        num_rows = a_triangles.shape[0]
        zeros_column = np.zeros((num_rows, 1))

        a_triangles = np.hstack((a_triangles, zeros_column))
        b_triangles = np.hstack((b_triangles, zeros_column))
        c_triangles = np.hstack((c_triangles, zeros_column))
        for p in self.pixels:
            repeated_array = np.tile(p, (a_triangles.shape[0],1))
            bc = igl.barycentric_coordinates_tri(repeated_array, a_triangles,b_triangles,c_triangles) # barycentric coordinates of the mesh in the uv space
            all_positive_rows = np.all(bc > 0, axis=1)
            positive_row_index = np.where(all_positive_rows)[0]
            if len(positive_row_index) > 0:
                face_index.append(positive_row_index[0])
            else:
                face_index.append(-1)
        self.face_index = face_index


    def get_pixel_colors(self, map=None): 
        """ Gets the color of each pixel in the grid (according to the face it belongs to) """
        color_ = np.zeros(self.pixels.shape[0])
        if map is None:
            map = self.color_map
        for i, index in enumerate(self.face_index):
            point = self.pixels[i]
            if index != -1:
                # print(f"Point {point} is inside triangles with index: {index}")
                color_[i] = map[index]
            else:
                color_[i] = -1
        self.pixel_colors = color_
        return color_

    def get_valid_color_idx(self, color):
        return np.where(color != -1)
    
    def get_invalid_color_idx(self, color):
        return np.where(color == -1)
    
    def get_valid_color(self, color):
        return color[self.get_valid_color_idx(color)]
    
    def get_invalid_color(self, color):
        return color[self.get_invalid_color_idx(color)]

    def print_stats(self):
        print(f"Number of vertices: {self.v.shape[0]}")
        print(f"Number of faces: {self.f.shape[0]}")        
        print(f"Map type: {self.current_map}")
        print(f"Number of pixels: {len(self.pixel_colors)}")
        print(f"Number of valid pixel colors (inside circle): {len(self.get_valid_color(self.pixel_colors))}")
        print(f"Number of invalid pixel colors (outside circle): {len(self.get_invalid_color(self.pixel_colors))}")

    def plot_grid(self, cmap=None):
        """ Plots the grid of pixels with colors -- mostly used for debugging -- will not work with color maps that have not been normalized.
        To get a sense of the color map it is better to use the plot function."""
        # Extract x and y coordinates
        x = self.pixels[:, 0]
        y = self.pixels[:, 1]
        pc = self.get_pixel_colors(cmap)
        if cmap is None:
            pc = self.pixel_colors
        # color values to plot -- make rgb
        color_plot = self.color_to_3d(pc) # Set color to black for points outside the triangle

        # Create the scatter plot
        plt.figure(figsize=(8, 8))  # Adjust figure size as needed
        plt.scatter(x, y, c=color_plot)
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.title("Plot of pixels with Colors")
        plt.grid(True)
        plt.show()

    """ 
    Dithering scheme Taken from this: 
        https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf
    """
    def dithering(self, bayer_kernel=1):
        if bayer_kernel == 1:
            I = np.array([[1, 2], [3, 0]])
        elif bayer_kernel == 2:
            I = np.array([[5,9, 6, 10], [13,1,14,2], [7,11,4,8], [15,3,12,0]])
        N = I.shape[0]  # N will be 2 in this case
        T = (I + 0.5) / (N * N)
        pixel_color_matrix = self.pixel_colors.reshape(self.uu.shape)
        points = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (np.isclose(pixel_color_matrix[i, j], -1)):
                    continue
                if pixel_color_matrix[i, j] < T[i%N, j%N]:
                    points.append([self.uu[i, j], self.vv[i, j]])
        return points

    def dithering_basic(self, desired_points, threshold=0.5):
        """ Basic dithering scheme that just checks if the pixel color is less than the threshold -- was my initial version of the dithering scheme """
        pixel_color_matrix = self.pixel_colors.reshape(self.uu.shape)
        points = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (pixel_color_matrix[i, j] == -1):
                    continue
                if pixel_color_matrix[i, j] < threshold:
                    points.append([self.uu[i, j], self.vv[i, j]])
        self.new_points = np.array(points)
        return self.new_points
        # return points

    def remesh(self, kernel=1):
        """ Remesh the mesh using the dithering scheme with the Bayer kernel """
        print(f"Remeshing with Bayer kernel: {kernel}")
        if kernel == 1:
            points = self.dithering(1)
        else:
            points = self.dithering(2)
        points = np.array(points)
        print(f"Triangulating {len(points)} points")
        tri = Delaunay(points)
        new_f = tri.simplices  # Extract the simplices (triangles)
        print(f"Finding 3d Coordinates:")
        points_3d = np.array([self.compute_3d_from_2d(uv) for uv in points])
        p = mp.subplot(points, s=[1,2,0], shading={"point_size": .1})
        mp.subplot(points_3d, new_f, s=[1, 2, 1], data=p, shading={"wireframe": True})
        self.remeshed_points = points_3d
        self.remeshed_faces = new_f
        np.savetxt(f"{self.meshname}_{self.grid_size}_points.txt", self.remeshed_points, fmt='%d', delimiter=',', newline='\n')
        np.savetxt(f"{self.meshname}_{self.grid_size}_faces.txt", self.remeshed_faces, fmt='%d', delimiter=',', newline='\n')
        return points_3d, new_f
    def dithering_target_points(self, target_points):
        """
        Dithering scheme that aims for a specific number of output points
        by dynamically adjusting the threshold.
        """
        pixel_color_matrix = self.pixel_colors.reshape(self.uu.shape)
        all_valid_colors = self.get_valid_color(self.pixel_colors)

        if not all_valid_colors.size:
            return np.array([])

        # Sort the valid pixel colors
        sorted_colors = np.sort(all_valid_colors)
        num_valid_pixels = len(sorted_colors)

        if target_points >= num_valid_pixels:
            # If the target is more than all valid pixels, return all valid pixel centers
            valid_indices = self.get_valid_color_idx(self.pixel_colors)[0]
            points = self.pixels[valid_indices][:, :2]
            return points
        elif target_points <= 0:
            return np.array([])
        else:
            # Determine the threshold based on the sorted colors
            threshold_index = target_points - 1
            threshold = sorted_colors[threshold_index]
            print(f"Dithering with dynamic threshold: {threshold} to get {target_points} points.")

            points = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (pixel_color_matrix[i, j] != -1) and (pixel_color_matrix[i, j] < threshold):
                        points.append([self.uu[i, j], self.vv[i, j]])
            return np.array(points)

    def remesh_with_target(self, target_points):
        """ Remesh the mesh using dithering to achieve a target number of points. """
        print(f"Remeshing to get approximately {target_points} points.")
        points = self.dithering_target_points(target_points)
        print(f"Triangulating {len(points)} points")
        if len(points) < 3:
            print("Warning: Not enough points for triangulation.")
            return None, None

        tri = Delaunay(points)
        new_f = tri.simplices  # Extract the simplices (triangles)
        print(f"Finding 3d Coordinates:")
        points_3d = np.array([self.compute_3d_from_2d(uv) for uv in points])
        p = mp.subplot(points, s=[1,2,0], shading={"point_size": .1})
        mp.subplot(points_3d, new_f, s=[1, 2, 1], data=p, shading={"wireframe": True})
        self.remeshed_points = points_3d
        self.remeshed_faces = new_f
        np.savetxt(f"{self.meshname}_{self.grid_size}_{target_points}_points.txt", self.remeshed_points, fmt='%g', delimiter=',', newline='\n') # Use %g for floating point
        np.savetxt(f"{self.meshname}_{self.grid_size}_{target_points}_faces.txt", self.remeshed_faces, fmt='%d', delimiter=',', newline='\n')
        return points_3d, new_f

    def compare_maps(self, target_point_count=2000): # Set a default target if needed
        """ Compares the remeshing result for log_map and uniform_map with a target point count. """
        print(f"\n--- Remeshing with LOG map to get ~{target_point_count} points ---")
        self.set_map(self.log_map, "LOG")
        self.get_pixel_colors() # Update pixel colors based on the current map
        log_points_3d, log_faces = self.remesh_with_target(target_point_count)

        print(f"\n--- Remeshing with UNIFORM map to get ~{target_point_count} points ---")
        self.set_map(self.uniform_map, "UNIFORM")
        self.get_pixel_colors() # Update pixel colors based on the current map
        uniform_points_3d, uniform_faces = self.remesh_with_target(target_point_count)

        return log_points_3d, log_faces, uniform_points_3d, uniform_faces

def return_neighbors(i,j):
    """
    Get the neighbors of a pixel in a 2D grid.
    """
    return [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]