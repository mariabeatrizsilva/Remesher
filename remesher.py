
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
        self.coords2d = None      # 2d coordinates of the vertices in the mesh (by face)
        self.color_map = None     # current color map for the mesh (i.e each face has a color)
        self.uniform_coeff = uniform_coeff # coefficient for the uniform map
        print(f"computing harmonic parameterization for {self.v.shape[0]} vertices and {self.f.shape[0]} faces")
        self.harmonic(plot=False)
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
        """ Computes the area maps for the mesh """
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
        self.uniform_map = np.ones_like(distort) * self.uniform_coeff # uniform map -- all values = uniform_coeff
        self.set_map(self.log_map, "LOG") # default set the log map

    def set_map(self, map, map_type):
        self.color_map = map
        self.current_map = map_type


    def compute_3d_from_2d(self, u):
        """ Computes the 3d coordinates of a 2d point u """
        if (self.coords2d is None):
            raise ValueError("You need to compute the 2d coordinates first")
        u = np.array(u).flatten() 
        u_3d = np.array([u[0], u[1], 0.0]) # Pad with a zero z-coordinate
        threshold = 1e-9
        for i in range(self.f.shape[0]): # Loop through every face and get the barycentric coordinates of u in that face -- if theyre all positive, then u is inside the triangle
            corner1 = self.coords2d[i, 0] 
            corner2 = self.coords2d[i, 1]
            corner3 = self.coords2d[i, 2]
            a = np.array([corner1[0], corner1[1], 0.0]) # Pad with a zero z-coordinate
            b = np.array([corner2[0], corner2[1], 0.0])
            c = np.array([corner3[0], corner3[1], 0.0])
            bc = igl.barycentric_coordinates_tri(np.array([u_3d]), np.array([a]), np.array([b]), np.array([c]))

            if np.all(bc >= threshold) and np.isclose(np.sum(bc), 1.0, atol=threshold):
                    v_indices = self.f[i] # the indices of the vertices of the face
                    # convert the u into 3d coordinates using the barycentric coordinates
                    x1 = self.v[v_indices[0]] # the coordinates of the first vertex
                    x2 = self.v[v_indices[1]]
                    x3 = self.v[v_indices[2]]
                    x = bc[0] * x1 + bc[1] * x2 + bc[2] * x3
                    return x

        return None # Return None if the 2D point is not found

    def compute_3d_from_2d_vectorized(self, points_2d):
            """ Computes the 3d coordinates of a 2d point u """
            if self.coords2d is None:
                raise ValueError("You need to compute the 2d coordinates first")
            
            num_points = points_2d.shape[0]
            num_faces = self.f.shape[0]

            points_2d_padded = np.hstack((points_2d, np.zeros((num_points, 1)))) # Pad each point with a zero z-coordinate -- size (N,3)

            zero_stack = np.zeros((num_faces, 1)) 
            # get coordiantes of the triangle vertices in 3d space -- size F,3
            triangle_vertices_a = np.hstack((self.coords2d[:, 0, :], zero_stack)) 
            triangle_vertices_b = np.hstack((self.coords2d[:, 1, :], zero_stack))
            triangle_vertices_c = np.hstack((self.coords2d[:, 2, :], zero_stack))

            # we need to check each point against all faces, so we need to repeat the points and the triangle vertices (N * F, 3)
            repeated_points = np.tile(points_2d_padded, (1, num_faces)).reshape(num_points * num_faces, 3)

            # repeat the triangle vertices for each point
            repeated_a = np.tile(triangle_vertices_a, (num_points, 1))
            repeated_b = np.tile(triangle_vertices_b, (num_points, 1))
            repeated_c = np.tile(triangle_vertices_c, (num_points, 1))

            bc = igl.barycentric_coordinates_tri(repeated_points, repeated_a, repeated_b, repeated_c)

            threshold = 1e-9
            valid_barycentric = np.all(bc >= -threshold, axis=1) & (np.abs(np.sum(bc, axis=1) - 1) < threshold)
            valid_barycentric = valid_barycentric.reshape((num_points, num_faces))

            points_3d = np.ones((num_points, 3)) * -1 # Initialize with -1 to indicate invalid points
            for point_index in range(num_points):
                for face_index in range(num_faces):
                    if valid_barycentric[point_index, face_index]:
                        vertex_indices = self.f[face_index]
                        v1_3d = self.v[vertex_indices[0]]
                        v2_3d = self.v[vertex_indices[1]]
                        v3_3d = self.v[vertex_indices[2]]
                        bc_idx = point_index * num_faces + face_index
                        points_3d[point_index] = (
                            bc[bc_idx, 0] * v1_3d + 
                            bc[bc_idx, 1] * v2_3d + 
                            bc[bc_idx, 2] * v3_3d
                        )
                        break  # Exit inner loop after finding a valid face
            return points_3d

    def color_to_3d(self, map):
        """ Converts the color map to a 3d color map for plotting -- each face has a color """
        colors = np.stack([map,
                   map,
                   map], axis=-1)
        colors[self.get_invalid_color_idx(map)] = [1, 0, 0]
        return colors
    


    def plot(self, map=None, wf=False):
        """ Plots the mesh with the color map """
        if map is None:
            map = self.color_map
        c = self.color_to_3d(map)
        # print(f"Plotting with {self.current_map} map")
        p1 = mp.subplot(self.v, self.f, c=c, s=[1, 2, 0], shading={"wireframe": wf, "flat": False})
        mp.subplot(self.h, self.f, s=[1, 2, 1], data=p1,  c=c, shading={"wireframe": wf})
        return p1
    
    def build_grid(self, grid_size=100):
        """ Builds a grid of pixels in the uv space """
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
            # point = self.pixels[i]
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
        """ Dithering scheme using Bayer kernel """
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

    def dithering_basic(self, threshold=0.5):
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
        new_f = tri.simplices  
        print(f"Finding 3d Coordinates:")
        points_3d = self.compute_3d_from_2d_vectorized(points) # np.array([self.compute_3d_from_2d(uv) for uv in points])
        p = mp.subplot(points, s=[1,2,0], shading={"point_size": .1})
        mp.subplot(points_3d, new_f, s=[1, 2, 1], data=p, shading={"wireframe": True})
        self.remeshed_points = points_3d
        self.remeshed_faces = new_f
        np.savetxt(f"{self.meshname}_{self.grid_size}_points.txt", self.remeshed_points, fmt='%d', delimiter=',', newline='\n')
        np.savetxt(f"{self.meshname}_{self.grid_size}_faces.txt", self.remeshed_faces, fmt='%d', delimiter=',', newline='\n')
        return points_3d, new_f
