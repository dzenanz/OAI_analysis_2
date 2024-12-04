"""
Comprises various functions needed for mesh processing.
For ex. mesh smoothing, face normal calculation, face centroid calculation, 
mesh splitting, mesh thickness computation, mesh attributes mapping to atlas mesh,
connected component analysis of mesh.

Sample usage for thickness computation
    import mesh_processing as mp
    distance_inner, distance_outer = mp.get_thickness_mesh(itk_image, mesh_type=='FC')

Sample usage for mapping attributes/data to atlas mesh (target mesh)
    mapped_mesh = mp.map_attributes(source_mesh, target_mesh)
"""
import os
import tempfile

import numpy as np
from sklearn.cluster import KMeans
import itk
import trimesh
import vtk
from vtk.util import numpy_support as ns
from vtkmodules.vtkCommonCore import reference
from vtk import vtkPointLocator, vtkKdTreePointLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from oai_analysis.pipeline import write_vtk_mesh


# Helper Functions for Mesh Processing

def read_vtk_mesh(filename):
    if str(filename)[-4:] == ".ply":
        reader = vtk.vtkPLYReader()
    else:
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


# Get Centroid of all the cells
def get_cell_centroid(mesh: itk.Mesh):
    mesh = get_trimesh(mesh)
    centroid_array = np.zeros([len(mesh.faces), 3])
    num_of_cells = len(mesh.faces)

    points_array = np.zeros([num_of_cells, 3, 3])

    for i in range(num_of_cells):
        c = mesh.faces[i]
        p = [mesh.vertices[c[0]], mesh.vertices[c[1]], mesh.vertices[c[2]]]
        points_array[i] = p

    sum_array = np.sum(points_array, axis=1)
    centroid_array = sum_array / 3.0
    return centroid_array


# Get Normal of all the cells
def get_cell_normals(mesh: itk.Mesh):
    mesh = get_trimesh(mesh)
    return mesh.face_normals

# Convert from ITK Mesh to Trimesh
def get_trimesh(itk_mesh):
    d = itk.dict_from_mesh(itk_mesh)
    faces = d['cells'].reshape((len(d['cells'])//5,5))[:,2:]
    vertices = d['points'].reshape((len(d['points'])//3, 3))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


# Convert from VTK Mesh to ITK Mesh to make it serializable
def get_itk_mesh(vtk_mesh):
    Dimension = 3
    PixelType = itk.D

    MeshType = itk.Mesh[PixelType, Dimension]
    itk_mesh = MeshType.New()

    # Get points array from VTK mesh
    points = vtk_mesh.GetPoints().GetData()
    points_numpy = np.array(points).flatten().astype("float32")

    polys = vtk_mesh.GetPolys().GetData()
    polys_numpy = np.array(polys).flatten()

    # Triangle Mesh
    vtk_cells_count = vtk_mesh.GetNumberOfPolys()
    polys_numpy = np.reshape(polys_numpy, [vtk_cells_count, Dimension + 1])

    # Extracting only the points by removing first column that denotes the VTK cell type
    polys_numpy = polys_numpy[:, 1:]
    polys_numpy = polys_numpy.flatten().astype(np.uint64)

    # Get point data from VTK mesh to insert in ITK Mesh
    point_data_numpy = np.array(
        vtk_mesh.GetPointData().GetScalars()
    )  # .astype('float64')

    # Get cell data from VTK mesh to insert in ITK Mesh
    cell_data_numpy = np.array(
        vtk_mesh.GetCellData().GetScalars()
    )  # .astype('float64')

    itk_mesh.SetPoints(itk.vector_container_from_array(points_numpy))
    itk_mesh.SetCellsArray(
        itk.vector_container_from_array(polys_numpy),
        itk.CommonEnums.CellGeometry_TRIANGLE_CELL,
    )
    if point_data_numpy.dtype == np.float64:
        itk_mesh.SetPointData(itk.vector_container_from_array(point_data_numpy))
        itk_mesh.SetCellData(itk.vector_container_from_array(cell_data_numpy))
    return itk_mesh


def itk_mesh_to_vtk_mesh(itk_mesh, intermediate_filename=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'temp.vtk')
        if intermediate_filename is None:
            intermediate_filename = path
        itk.meshwrite(itk_mesh, intermediate_filename, compression=True)
        vtk_mesh = read_vtk_mesh(intermediate_filename)
    return vtk_mesh


# Get VTK mesh using vertices and faces array
def get_vtk_mesh(verts, faces):
    # Add Cells
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, faces.shape[0]*3 + 1, 3).astype('int')
    connectivity = faces.flatten().astype('int')
    cells.SetData(ns.numpy_to_vtk(offsets), ns.numpy_to_vtk(connectivity))

    # Create a poly data object
    vtk_mesh = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(verts, deep=True))

    vtk_mesh.SetPoints(vtk_points)
    vtk_mesh.SetPolys(cells)
    vtk_mesh.Modified()

    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(vtk_mesh)
    connectivityFilter.SetExtractionModeToAllRegions()
    connectivityFilter.Update()

    append_polydata = vtk.vtkAppendPolyData()

    # Filter out small regions
    filter_thresh = 3000

    for i in range(connectivityFilter.GetNumberOfExtractedRegions()):
        connectivityFilter.AddSpecifiedRegion(i)
        connectivityFilter.SetExtractionModeToSpecifiedRegions()
        connectivityFilter.Update()

        out1 = connectivityFilter.GetOutput()

        if out1.GetNumberOfCells() > filter_thresh:
            polydata = vtk.vtkPolyData()
            polydata.ShallowCopy(out1)
            append_polydata.AddInputData(polydata)

        connectivityFilter.DeleteSpecifiedRegion(i)

    append_polydata.Update()
    combined_mesh = append_polydata.GetOutput()

    return combined_mesh


# Create a Mesh by selecting relevant faces only
def get_vtk_sub_mesh(input_mesh, inner_face_list):
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Add Cells
    counter = 0
    point_ids_hash = {}
    for i in inner_face_list:
        tri = vtk.vtkTriangle()
        current_cell_points = input_mesh.GetCell(i).GetPointIds()

        if current_cell_points.GetNumberOfIds() != 3:
            print("Ids are not three Some Issue ", current_cell_points.GetNumberOfIds())

        for i in range(3):
            point_id = current_cell_points.GetId(i)

            if point_id in point_ids_hash:
                new_point_id = point_ids_hash[point_id]
            else:
                new_point_id = counter
                point_ids_hash[point_id] = new_point_id
                counter = counter + 1

            tri.GetPointIds().SetId(i, new_point_id)

        cells.InsertNextCell(tri)

    points.SetNumberOfPoints(len(list(point_ids_hash.keys())))

    # Add points
    for i in point_ids_hash:
        p = input_mesh.GetPoint(i)
        new_id = point_ids_hash[i]
        points.SetPoint(new_id, p)

    # Create a poly data object
    polydata = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    polydata.Modified()

    return polydata


# For splitting the Tibial cartilage
def split_tibial_cartilage_surface(mesh, mesh_normals, mesh_centroids):
    mesh_centroids_normalized = (mesh_centroids - np.mean(mesh_centroids, axis=0)) / (
        np.max(mesh_centroids, axis=0) - np.min(mesh_centroids, axis=0)
    )

    # clustering normals
    features = np.concatenate(
        (mesh_centroids_normalized * 1, mesh_normals * 10), axis=1
    )

    est = KMeans(n_clusters=2, algorithm="lloyd", random_state=5)
    labels = est.fit(features).labels_

    # transfer 0/1 labels to -1/1 labels
    inner_outer_label_list = labels * 2 - 1

    # set inner surface which contains mostly negative normals
    if mesh_normals[inner_outer_label_list == -1, 2].mean() > 0:
        inner_outer_label_list = -inner_outer_label_list

    inner_face_list = np.where(inner_outer_label_list == -1)[0]
    outer_face_list = np.where(inner_outer_label_list == 1)[0]

    inner_mesh = get_vtk_sub_mesh(mesh, inner_face_list)
    outer_mesh = get_vtk_sub_mesh(mesh, outer_face_list)

    return inner_mesh, outer_mesh, inner_face_list, outer_face_list


# Clusters the feature vectors and segments the mesh
def cluster_and_segment(mesh_centroids_normalized, face_normal_value, dot_output):
    features = np.concatenate(
        (mesh_centroids_normalized * 1, face_normal_value, dot_output), axis=1
    )
    est1 = KMeans(n_clusters=2, algorithm="lloyd", n_init=5, random_state=5)
    labels_upper = est1.fit(features).labels_
    labels_upper = labels_upper * 2 - 1

    # set inner surface which contains mostly positive normals
    if face_normal_value[labels_upper == -1, 2].mean() < 0:
        labels_upper = -labels_upper

    return labels_upper


# For splitting the Femoral cartilage
def split_femoral_cartilage_surface(mesh, face_normal, face_centroid, num_divisions=3):
    mesh_centroids_normalized = (face_centroid - np.mean(face_centroid, axis=0)) / (
        np.max(face_centroid, axis=0) - np.min(face_centroid, axis=0)
    )

    (xmin, xmax, ymin, ymax, zmin, zmax) = mesh.GetBounds()
    bbox_min = np.array([xmin, ymin, zmin])
    bbox_max = np.array([xmax, ymax, zmax])

    center = (bbox_min + bbox_max) / 2

    inner_outer_label_list = np.zeros(mesh.GetNumberOfCells())  # up:1, down:-1
    connect_direction = normalize(center - face_centroid, axis=1, norm='l2')

    dot_output = np.multiply(connect_direction, face_normal)
    y_coord = mesh_centroids_normalized[:, 1]

    # For puting the labels at correct indices for all the segments
    inner_outer_label_list = np.zeros(mesh_centroids_normalized.shape[0])

    # For dividing the mesh into smaller segments for better clustering
    min_y = np.min(mesh_centroids_normalized[:, 1])
    max_y = np.max(mesh_centroids_normalized[:, 1])
    step_value = (max_y - min_y) / num_divisions

    # Perform clustering for each segment individually
    for i in range(num_divisions):
        lower_y = min_y + step_value * i
        upper_y = lower_y + step_value
        current_indices = np.where((y_coord >= lower_y) & (y_coord < upper_y))[0]

        mesh_centroids_normalized_extracted = mesh_centroids_normalized[current_indices]
        face_normal_value_extracted = face_normal[current_indices]
        dot_output_extracted = dot_output[current_indices]

        # Clustering the current segment
        current_labels = cluster_and_segment(
            mesh_centroids_normalized_extracted,
            face_normal_value_extracted,
            dot_output_extracted,
        )

        # Put the labels into the correct positions in the full mesh
        np.put(inner_outer_label_list, current_indices, current_labels)

    inner_face_list = np.where(inner_outer_label_list == -1)[0]
    outer_face_list = np.where(inner_outer_label_list == 1)[0]

    inner_mesh = get_vtk_sub_mesh(mesh, inner_face_list)
    outer_mesh = get_vtk_sub_mesh(mesh, outer_face_list)

    return inner_mesh, outer_mesh, inner_face_list, outer_face_list


# For smoothing the input mesh. Change number of iterations as per usecase.
def smooth_mesh(input_mesh, num_iterations=150):
    smoothing_filter = vtk.vtkSmoothPolyDataFilter()
    smoothing_filter.SetNumberOfIterations(num_iterations)

    smoothing_filter.SetInputData(input_mesh)
    smoothing_filter.Update()

    output_mesh = smoothing_filter.GetOutput()
    return output_mesh


# For getting nearest neighbour distance between inner and outer mesh.
def get_distance(inner_mesh, outer_mesh):
    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.SetInputData(0, inner_mesh)
    distance_filter.SetInputData(1, outer_mesh)
    distance_filter.SignedDistanceOff()
    distance_filter.SetComputeSecondDistance(True)
    distance_filter.Update()

    distance_inner = distance_filter.GetOutput()
    distance_outer = distance_filter.GetSecondDistanceOutput()

    return distance_inner, distance_outer


# To obtain mesh and attributes from itk_image
def get_mesh(itk_image, num_iterations=150):
    import skimage.measure

    spacing = itk_image.GetSpacing()
    img_array = np.swapaxes(np.asarray(itk_image), 0, 2).astype(float)

    # Obtain the mesh from Probability maps using Marching Cubes
    verts, faces, normals, values = skimage.measure.marching_cubes(
        img_array, level=None, spacing=spacing, step_size=1, gradient_direction="ascent"
    )

    mesh = get_vtk_mesh(verts, faces)

    # For smoothing the mesh surface to obtain gradually varying face normals
    mesh = smooth_mesh(mesh, num_iterations=num_iterations)
    return mesh

# To obtain mesh from itk image probability map
def get_mesh_from_probability_map(image: itk.Image) -> itk.Mesh:
    mesh = itk.cuberille_image_to_mesh_filter(image,
                                              generate_triangle_faces=True,
                                              iso_surface_value=0.5,
                                              project_vertices_to_iso_surface=True,
                                              project_vertex_surface_distance_threshold=0.05)
    return mesh


# To obtain inner and outer mesh splits given the mesh type
def split_mesh(mesh, mesh_type="FC"):
    # Obtain the cell normals and centroids to be used for splittig the cartilage
    itk_mesh = get_itk_mesh(mesh)
    mesh_cell_normals = get_cell_normals(itk_mesh)
    mesh_cell_centroids = get_cell_centroid(itk_mesh)

    # Split the mesh
    if mesh_type == "FC":
        (
            inner_mesh,
            outer_mesh,
            inner_face_list,
            outer_face_list,
        ) = split_femoral_cartilage_surface(
            mesh, mesh_cell_normals, mesh_cell_centroids
        )
    else:
        (
            inner_mesh,
            outer_mesh,
            inner_face_list,
            outer_face_list,
        ) = split_tibial_cartilage_surface(mesh, mesh_cell_normals, mesh_cell_centroids)

    return inner_mesh, outer_mesh


# Obtain the thickness of the input itk_image by creating a mesh and splitting it.
def get_split_mesh(vtk_mesh, atlas_mesh, mesh_type="FC", num_iterations=150, distance_threshold_mm=10.0):
    """
    Takes the probability map obtained from the segmentation algorithm as an itk image.
    Constructs a VTK mesh from it and returns the thickness between the inner and outer splitted mesh.
    Takes as argument the type of mesh 'FC' or 'TC'.
    atlas_mesh is used for island filtering - we keep islands close to it
    """
    # Keep the largest 1 (FC) or 2 (TC) regions
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(vtk_mesh)
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()
    region_sizes = ns.vtk_to_numpy(connect.GetRegionSizes())
    sorted_indices = np.argsort(region_sizes)[::-1]
    # Usual sizes: FC 150k, TC 2x 25k

    # Let's keep all the regions bigger than 10k, and at least 1 for FC and 2 for TC
    # We also use size as the centroid of a curved cartilage surfaces could be far from atlas mesh
    indices_to_keep = [sorted_indices[0]]  # The largest region

    region_iterator = 1
    while region_iterator < len(sorted_indices) and region_sizes[sorted_indices[region_iterator]] > 10000:
        indices_to_keep.append(sorted_indices[region_iterator])
        region_iterator += 1

    if mesh_type == "TC" and region_iterator < 2:
        indices_to_keep.append(sorted_indices[1])  # Second largest region
        region_iterator += 1

    # now check island centroid's distance to the atlas mesh and keep the close ones
    cc = connect.GetOutput()
    region_ids = ns.vtk_to_numpy(cc.GetPointData().GetScalars())
    vertices = ns.vtk_to_numpy(cc.GetPoints().GetData())
    locator = vtkPointLocator()
    locator.SetDataSet(atlas_mesh)
    locator.BuildLocator()
    np.set_printoptions(precision=2)
    while region_iterator < len(sorted_indices):
        region = sorted_indices[region_iterator]
        region_vertices = vertices[region_ids == region]
        region_centroid = np.mean(region_vertices, axis=0)
        dist2 = reference(-1.0)
        locator.FindClosestPointWithinRadius(distance_threshold_mm, region_centroid, dist2)
        if 0 <= dist2 < distance_threshold_mm ** 2:
            indices_to_keep.append(region)
        region_iterator += 1

    connect.SetExtractionModeToSpecifiedRegions()
    for keep in indices_to_keep:
        connect.AddSpecifiedRegion(keep)
    connect.Update()
    cc = connect.GetOutput()

    # Smooth extracted mesh
    mesh = smooth_mesh(cc, num_iterations=num_iterations)

    # Split the mesh into inner and outer
    inner_mesh, outer_mesh = split_mesh(mesh, mesh_type)

    return inner_mesh, outer_mesh


# Map the attributes from the source mesh to target mesh (atlas mesh)
# Gets the closest point to map the attribute
def map_attributes(source_mesh, target_mesh):
    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetNullPointsStrategyToClosestPoint()
    interpolator.SetInputData(target_mesh)
    interpolator.SetSourceData(source_mesh)
    interpolator.Update()
    mapped_mesh = interpolator.GetOutput()
    return mapped_mesh


# For fitting a circle using given set of points
def compute_least_square_circle(x, y):
    method_2b = "leastsq with jacobian"
    from scipy import optimize

    def calc_R(xc, yc):
        """calculate the distance of each data points from the center (xc, yc)"""
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2b(c):
        """calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc)"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x) / Ri  # dR/dxc
        df2b_dc[1] = (yc - y) / Ri  # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

    Ri_2b = calc_R(*center_2b)
    R_2b = Ri_2b.mean()
    return center_2b, R_2b


# Project the vertices to the cylinder.
def get_projection_from_circle_and_vertices(vertices, circle):
    center, r = circle
    y, z = vertices[:, 1], vertices[:, 2]
    radian = np.arctan2(z - center[1], y - center[0])

    embedded = np.zeros([len(vertices), 2])
    embedded[:, 0] = radian
    embedded[:, 1] = vertices[:, 0]

    plot_yz = np.zeros_like(embedded)
    angle = (np.pi / 2 - radian) % (2*np.pi)  # shift and wrap around to avoid plotting discontinuity
    plot_yz[:, 0] = angle * r  # convert from radians to millimeters
    plot_yz[:, 1] = -vertices[:, 0]
    return embedded, plot_yz


# Projects the thickness in mapped mesh to 2D
# Takes as arugment the projected points (embedded), if not given then re-uses the
# already transformed points in the atlas mesh for the given mesh type.
def project_thickness(mapped_mesh, mesh_type="FC", embedded=None):
    def do_linear_pca(vertex, dim=3):
        from sklearn.decomposition import KernelPCA

        kpca = KernelPCA(n_components=2, degree=dim, n_jobs=None)
        embedded = kpca.fit_transform(vertex)
        return embedded

    thickness = np.array(mapped_mesh.GetPointData().GetScalars())

    if mesh_type == "FC":
        vertices = np.array(mapped_mesh.GetPoints().GetData())
        circle = compute_least_square_circle(vertices[:, 1], vertices[:, 2])
        embedded, plot_yz = get_projection_from_circle_and_vertices(vertices, circle)

        return plot_yz[:, 0], plot_yz[:, 1], thickness
    else:
        vertices = np.array(mapped_mesh.GetPoints().GetData())
        # thickness = np.array(mapped_mesh.GetPointData().GetScalars())

        vertice_left = vertices[vertices[:, 0] < -50]
        index_left = np.where(vertices[:, 0] < -50)[0]

        vertice_right = vertices[vertices[:, 0] >= -50]
        index_right = np.where(vertices[:, 0] >= -50)[0]

        embedded_left = do_linear_pca(vertice_left)
        embedded_right = do_linear_pca(vertice_right)

        embedded_right[:, 0] = -embedded_right[:, 0]  # flip x

        combined_embedded_x = np.concatenate(
            [embedded_right[:, 0], embedded_left[:, 0]]
        )
        combined_embedded_y = np.concatenate(
            [embedded_right[:, 1] + 50, embedded_left[:, 1]]
        )
        combined_thickness = np.concatenate(
            [thickness[index_right], thickness[index_left]]
        )

        return combined_embedded_x, combined_embedded_y, combined_thickness
