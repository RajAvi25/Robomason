import argparse
import ifcopenshell
import ifcopenshell.geom
import numpy as np
import json
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties

def loadIFC(ifc_path, debug=False):
    """
    Parse an IFC file to extract geometric data for each constructible element.

    Opens the given IFC BIM file and iterates through all `IfcProduct` entities (except openings and sites).
    For each element, it computes the coordinates of all vertices (using IfcOpenShell and OpenCASCADE) 
    and aggregates them. The result is a dictionary mapping each element's name to its type and coordinate lists.

    **Parameters:**
    - *ifc_path (str)* – Path to the IFC file to load.
    - *debug (bool)* – If True, prints detailed debug info about each face and vertex processed.

    **Returns:**
    - *dict* – A dictionary with keys:
        - `"name"`: list of element names (strings from the IFC).
        - `"type"`: list of IFC entity types (e.g., IfcWall, IfcSlab).
        - `"xcoord", "ycoord", "zcoord"`: lists of lists of coordinates for each element's vertices.

    **Process:**
    - Uses `ifcopenshell.open` to load the IFC file.
    - Filters out irrelevant entities (openings, site).
    - For each remaining product with geometry, uses `ifcopenshell.geom.create_shape` to get a 3D shape.
    - Explores all faces and vertices of the shape with OpenCASCADE (OCC) to collect coordinates.
    - Aggregates the coordinates per element. Each element’s entry corresponds to one constructible component.

    **Assumptions:**
    - IFC file is well-formed and contains geometric representations for products.
    - OpenCASCADE is used to compute face areas and vertex coordinates for potential advanced metrics (face area, etc.), though here mainly vertex coordinates are collected.

    **Role in System:**
    This function is part of the *IFC-driven parsing*, extracting raw geometry from BIM data. 
    The output is later processed to derive placement poses for each element in the construction sequence.
    """
    xdata = []
    ydata = []
    zdata = []

    # Initialize IFC entities dictionary
    entities = dict()

    # IFC Load cell
    ifc_file = ifcopenshell.open(ifc_path)  # this is where we load the file
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)

    # Iterate over all products in Ifc file
    for ifc_entity in ifc_file.by_type('IfcProduct'):  # IfcProduct

        # Exclude the entities that are of type IfcOpeningElement or IfcSite etc.
        if ifc_entity.is_a("IfcOpeningElement") or ifc_entity.is_a("IfcSite"):
            continue

        if ifc_entity.is_a("IfElementAssembly"):
            print(ifc_entity)

        if ifc_entity.Representation is not None:
            shape = ifcopenshell.geom.create_shape(settings, ifc_entity)

            # Create an explorer on entity shape and find all the face on this shape
            explore_face = TopExp_Explorer(shape.geometry, TopAbs_FACE)

            while explore_face.More():
                face = topods.Face(explore_face.Current())

                props = GProp_GProps()
                brepgprop_SurfaceProperties(face, props)

                # props.Mass() provides area of face
                face_area = props.Mass()
                if debug:
                    print(f"face:{face}, face area:{face_area}")

                # Create an explorer on face and find all the vertex
                explore_vertex = TopExp_Explorer(face, TopAbs_VERTEX)

                while explore_vertex.More():
                    vertex = topods.Vertex(explore_vertex.Current())

                    # Coordinate of vertex on face
                    vertex_point = BRep_Tool().Pnt(vertex).Coord()
                    if debug:
                        print(f"vertex:{vertex}, vertex point:{vertex_point}")

                    xdata.append(vertex_point[0])
                    ydata.append(vertex_point[1])
                    zdata.append(vertex_point[2])

                    explore_vertex.Next()
                explore_face.Next()

            if "name" in entities:
                entities["name"].append(ifc_entity.Name)
                entities["type"].append(ifc_entity.is_a())
                entities["xcoord"].append(xdata)
                entities["ycoord"].append(ydata)
                entities["zcoord"].append(zdata)
            else:
                entities["name"] = [ifc_entity.Name]
                entities["type"] = [ifc_entity.is_a()]
                entities["xcoord"] = [xdata]
                entities["ycoord"] = [ydata]
                entities["zcoord"] = [zdata]

            xdata, ydata, zdata = [], [], []  # For some reason this line is important???

    return entities

def IFC_sort(IFC_data):
    """
    Sort IFC elements by height (average Z-coordinate).

    Given the raw IFC data dictionary from `loadIFC()`, this function computes each element's average 
    coordinates and sorts the elements by their mean Z value. This determines the assembly/disassembly order 
    (e.g., foundation first, then walls, then floors, etc.), aligning with a bottom-up construction sequence.

    **Parameters:**
    - *IFC_data (dict)* – Dictionary as returned by `loadIFC`, containing lists of element names and coordinates.

    **Returns:**
    - *np.ndarray* – NumPy array of shape (N, 4) where each row is `[Name, X_mean, Y_mean, Z_mean]` for an element. 
      Rows are sorted in ascending order of Z_mean (lowest element first).

    **Details:**
    - Computes the mean of each element's `xcoord`, `ycoord`, and `zcoord` lists.
    - Uses the Z mean as a key to sort all elements (via `np.argsort` of all Z means).
    - Returns a structured array (as list of lists converted to np.array) suitable for output or further processing.

    **Role in System:**
    This provides a deterministic *global block sequence* from IFC data:contentReference[oaicite:3]{index=3}. The sorted output is used by 
    the planning routines to decide the pick-and-place order for assembly or the reverse for disassembly.
    """

    list = []
    list_sorted = []
    zmean = []  # to decide on the order of construction
    for i in range(0, np.size(IFC_data["name"])):  # to decide on the order of construction
        list.append([
            IFC_data["name"][i],
            np.mean(IFC_data["xcoord"][i]),
            np.mean(IFC_data["ycoord"][i]),
            np.mean(IFC_data["zcoord"][i])
        ])
        zmean.append(np.mean(IFC_data["zcoord"][i]))  # to decide on the order of construction
        order = np.argsort(zmean)  # to decide on the order of construction

    for i in order:
        list_sorted.append(list[i])
    return np.array(list_sorted)

def main(ifc_path, debug=False):
    # Initialize and sort IFC data
    IFC = loadIFC(ifc_path, debug)  # Load the IFC file
    IFC_sorted = IFC_sort(IFC)  # Sort the IFC data
    
    if debug:
        # Add any additional debug prints here if needed
        print("Debug information...")
    
    # Convert the NumPy array to a list and then to a JSON string
    array_json = json.dumps(IFC_sorted.tolist())
    
    print(array_json)  # Ensure only this is printed if debug is off
    return array_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an IFC file and sort its contents.')
    parser.add_argument('ifc_path', type=str, help='Path to the IFC file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    main(args.ifc_path, args.debug)
