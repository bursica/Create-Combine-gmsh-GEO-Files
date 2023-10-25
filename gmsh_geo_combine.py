# common
import re
import threading
import os
import sys
import warnings

# start_combine
import tkinter as tk
import tkinter.ttk as ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
from collections import defaultdict

# start_execute
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import combinations

# Declare global variables
file_paths = []
file_names = []
DND_FILES = 'DND_Files'
execution_thread = None
selected_code = None
canvas = None
bg_image = Image.open('background.png')
bg_photo = None
root = None
label = None
label_text = None
abort_button = None
width = 300
height = 300

# Preset values for geometry
pre_numClusters = 24
pre_numClusters_rep = 1
pre_numEmptySpaces = 0
pre_numClusters_off = 0
pre_Coherence_sw = 1
pre_chosen_index = 1
pre_Physical_sw = 1

warnings.filterwarnings("ignore")

class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self.should_stop = False

def clear_terminal():
    # Clear the terminal screen based on the operating system
    if sys.platform.startswith('win'):
        os.system('cls')  # For Windows
    else:
        os.system('clear')  # For Linux and macOS

def renumber_geo_file(file_path, starting_indexes, combined_lines):
    while True:
        if execution_thread.should_stop:
            break

        with open(file_path, 'r') as file:
            lines = file.readlines()

            point_mapping = {}
            line_mapping = {}
            curve_loop_mapping = {}
            surface_mapping = {}
            surface_loop_mapping = {}
            volume_mapping = {}

            new_lines = []
            new_point_index = starting_indexes['point']
            new_line_index = starting_indexes['line']
            new_curve_loop_index = starting_indexes['curve_loop']
            new_surface_index = starting_indexes['surface']
            new_surface_loop_index = starting_indexes['surface_loop']
            new_volume_index = starting_indexes['volume']

            physical_surface_dict = defaultdict(list)
            physical_volume_dict = defaultdict(list)

            # Parsing lines
            for line in lines:
                if line.startswith('Point('):
                    old_point_index = int(line.split('(')[1].split(')')[0])
                    point_mapping[old_point_index] = new_point_index
                    new_point_index += 1
                elif line.startswith('Line('):
                    old_line_index = int(line.split('(')[1].split(')')[0])
                    line_mapping[old_line_index] = new_line_index
                    new_line_index += 1
                elif line.startswith('Curve Loop('):
                    old_curve_loop_index = int(line.split('(')[1].split(')')[0])
                    curve_loop_mapping[old_curve_loop_index] = new_curve_loop_index
                    new_curve_loop_index += 1
                elif line.startswith(('Plane Surface(', 'Surface(')):
                    old_surface_index = int(line.split('(')[1].split(')')[0])
                    surface_mapping[old_surface_index] = new_surface_index
                    new_surface_index += 1
                elif line.startswith('Surface Loop('):
                    old_surface_loop_index = int(line.split('(')[1].split(')')[0])
                    surface_loop_mapping[old_surface_loop_index] = new_surface_loop_index
                    new_surface_loop_index += 1
                elif line.startswith('Volume('):
                    old_volume_index = int(line.split('(')[1].split(')')[0])
                    volume_mapping[old_volume_index] = new_volume_index
                    new_volume_index += 1 

                if not line.startswith('Transfinite'):
                    new_lines.append(line)

            # Creating new lines
            for i, line in enumerate(new_lines):
                if line.startswith('Point('):
                    old_point_index = int(line.split('(')[1].split(')')[0])
                    new_lines[i] = line.replace(f'Point({old_point_index})', f'Point({point_mapping[old_point_index]})')
                elif line.startswith('Line('):
                    old_line_index = int(line.split('(')[1].split(')')[0])
                    points = line.split('{')[1].split('}')[0].split(',')
                    new_points = [str(point_mapping[int(point)]) for point in points]
                    new_lines[i] = re.sub(r'Line\(\d+\) = \{.*\}', f'Line({line_mapping[old_line_index]}) = {{{", ".join(new_points)}}}', line)
                    new_lines.insert(i+1, f'Transfinite Line {{{line_mapping[old_line_index]}}} = 2 Using Progression 1;\n')
                elif line.startswith('Curve Loop('):
                    old_curve_loop_index = int(line.split('(')[1].split(')')[0])
                    lines_in_loop = line.split('{')[1].split('}')[0].split(',')
                    new_lines_in_loop = [str(-line_mapping[abs(int(line.strip()))]) if int(line.strip()) < 0 else str(line_mapping[abs(int(line.strip()))]) for line in lines_in_loop]
                    new_lines[i] = re.sub(r'Curve Loop\(\d+\) = \{.*\}', f'Curve Loop({curve_loop_mapping[old_curve_loop_index]}) = {{{", ".join(new_lines_in_loop)}}}', line)
                elif line.startswith(('Plane Surface(', 'Surface(')):
                    old_surface_index = int(line.split('(')[1].split(')')[0])
                    new_lines[i] = line.replace(f'Surface({old_surface_index}) = {{{old_surface_index}}}', f'Surface({surface_mapping[old_surface_index]}) = {{{surface_mapping[old_surface_index]}}}')
                    new_lines.insert(i+1, f'Transfinite Surface {{{surface_mapping[old_surface_index]}}};\n')    
                elif line.startswith('Surface Loop('):
                    old_surface_loop_index = int(line.split('(')[1].split(')')[0])
                    surfaces_in_loop = line.split('{')[1].split('}')[0].split(',')
                    new_surfaces_in_loop = [str(surface_mapping[int(surface.strip())]) for surface in surfaces_in_loop]
                    new_lines[i] = re.sub(r'Surface Loop\(\d+\) = \{.*\}', f'Surface Loop({surface_loop_mapping[old_surface_loop_index]}) = {{{", ".join(new_surfaces_in_loop)}}}', line)
                elif line.startswith('Volume('):
                    old_volume_index = int(line.split('(')[1].split(')')[0])
                    volume_number = line.split('{')[1].split('}')[0]
                    new_lines[i] = line.replace(f'Volume({old_volume_index}) = {{{volume_number}}}', f'Volume({volume_mapping[old_volume_index]}) = {{{volume_mapping[old_volume_index]}}}')
                    new_lines.insert(i+1, f'Transfinite Volume {{{volume_mapping[old_volume_index]}}};\n')  

                # Physical entities
                elif line.startswith('Physical Surface'):
                    surface_name = line.split('"')[1]
                    if Physical_sw == 1:
                        surface_indexes = [str(surface_mapping[old_surface_index]) for old_surface_index in surface_mapping]
                        new_lines[i] = f'Physical Surface("{surface_name}") = {{{", ".join(surface_indexes)}}};\n'
                    else:
                        surface_indexes = line.split('{')[1].split('}')[0].split(", ")  # Extracts the surface indexes 
                        old_surface_indexes = [int(index) for index in surface_indexes]  # Converts the indexes to integers
                        new_surface_indexes = [str(surface_mapping[idx]) for idx in old_surface_indexes]
                        new_lines[i] = f'Physical Surface("{surface_name}") = {{{", ".join(new_surface_indexes)}}};\n'

                elif line.startswith('Physical Volume'):
                    volume_name = line.split('"')[1]  # Extracts the volume name
                    if Physical_sw == 1:
                        volume_indexes = [str(volume_mapping[old_volume_index]) for old_volume_index in volume_mapping]
                        new_lines[i] = f'Physical Volume("{volume_name}") = {{{", ".join(volume_indexes)}}};\n'
                    else:    
                        volume_indexes = line.split('{')[1].split('}')[0].split(", ")  # Extracts the volume indexes 
                        old_volume_indexes = [int(index) for index in volume_indexes]  # Converts the indexes to integers
                        new_volume_indexes = [str(volume_mapping[idx]) for idx in old_volume_indexes]
                        new_lines[i] = f'Physical Volume("{volume_name}") = {{{", ".join(new_volume_indexes)}}};\n'

            return new_lines, {
                'point': max(point_mapping.values()) if point_mapping else 0,
                'line': max(line_mapping.values()) if line_mapping else 0,
                'curve_loop': max(curve_loop_mapping.values()) if curve_loop_mapping else 0,
                'surface': max(surface_mapping.values()) if surface_mapping else 0,
                'surface_loop': max(surface_loop_mapping.values()) if surface_loop_mapping else 0,
                'volume': max(volume_mapping.values()) if volume_mapping else 0
            }

def combine_geo_files(file_paths, output_file_path):
    global pre_Coherence_sw, pre_Physical_sw, Physical_sw   
    # Reset the input values
    combined_lines = []
    starting_indexes = {
        'point': 1,
        'line': 1,
        'curve_loop': 1,
        'surface': 1,
        'surface_loop': 1,
        'volume': 1,

    }
    max_mappings = {
        'point': 0,
        'line': 0,
        'curve_loop': 0,
        'surface': 0,
        'surface_loop': 0,
        'volume': 0,

    }

    
    try:
        # Ask the user to join clusters
        user_input1 = input("Please change current selection to 0 if you want to keep overlapping entities in model separated [1]: ")
        Coherence_sw = int(user_input1) if user_input1 != '' else pre_Coherence_sw

        # Ask the user to join clusters
        user_input2 = input("Please change current selection to 0 if you want to keep Physical Entities unchanged and renumerated only.\nCurrent selection populates all Physical Sufaces and Volumes to last named Physical entity within a geo file [1]: ")
        Physical_sw = int(user_input2) if user_input2 != '' else pre_Physical_sw

    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

    for file_path in file_paths:
        renumbered_lines, max_mappings = renumber_geo_file(file_path, starting_indexes, combined_lines)
        combined_lines.extend(renumbered_lines)

        for key in starting_indexes:
            starting_indexes[key] = max_mappings[key] + 1000

    with open(output_file_path, 'w') as file:
        file.writelines(combined_lines)

        # Join clusters
        if Coherence_sw == 1:
            file.write(f"Coherence;\n")

def generate_geo_file(file_paths, output_file_path):
    global pre_numClusters, pre_numClusters_rep, pre_numEmptySpaces, pre_numClusters_off, pre_Coherence_sw, pre_chosen_index
    while True:
        if execution_thread.should_stop:
            break
        # Assuming only one file in the file_paths list
        file_path = file_paths[0]

        # Create dictionaries to store the points, lines, and loops
        points_dict = {}
        lines_dict = {}
        new_key = None
        first_square_indices = None

        # Assign preset values before try-except block
        numClusters = pre_numClusters
        numClusters_rep = pre_numClusters_rep
        numEmptySpaces = pre_numEmptySpaces
        numClusters_off = pre_numClusters_off
        chosen_index = pre_chosen_index

        def find_quad_cycles(graph):
            cycles = list(nx.simple_cycles(graph))
            quad_cycles = [cycle for cycle in cycles if len(cycle) == 4]
            return quad_cycles

        # Function to calculate the total distance of a pair of points
        def total_distance(pair):
            return np.linalg.norm(np.array(pair[0]) - np.array(pair[1]))

        try:
            # Ask the user for inputs and reassign the variables if no exception occurs
            user_input1 = input("Please enter the number of clusters as it would be side by side in full circle [24]: ")
            numClusters = int(user_input1) if user_input1 != '' else pre_numClusters

            user_input2 = input("Please enter the number of cluster repetitions one after another [1]: ")
            numClusters_rep = int(user_input2) if user_input2 != '' else pre_numClusters_rep

            user_input3 = input("Please enter the number of empty spaces repetitions one after another [0]: ")
            numEmptySpaces = int(user_input3) if user_input3 != '' else pre_numEmptySpaces

            user_input4 = input(f"Please enter the number for first Cluster offset within {numClusters} [0]: ")
            numClusters_off = int(user_input4) if user_input4 != '' else pre_numClusters_off

            # Validate the inputs
            if numClusters < 1 or numClusters_rep <= 0 or numEmptySpaces < 0 or not 0 <= numClusters_off < numClusters:
                print("Invalid inputs. Please try again.")
                return
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
            return

        if execution_thread.should_stop:
            break

        # Read points from the file
        with open(file_path, 'r') as f:
            points = {}
            for line in f:
                if 'CARTESIAN_POINT' in line:
                    # Extract the point name and coordinates
                    name = line.split("'")[1]
                    coordinates = line.split("(")[2].replace(")", "").split(",")
                    coordinates = [float(re.sub(r"[^0-9.-]", "", c)) for c in coordinates]

                    # Add the point to the dictionary
                    if name.strip() == '':
                        name = 'Empty'  # Replace empty name with 'Empty'
                    if name in points:
                        points[name].append(coordinates)
                    else:
                        points[name] = [coordinates]

        # Calculate the rotation angle for each instance
        rotation_angle = 2 * math.pi / numClusters        

        # Calculate the total length of the ClusterPattern
        pattern_length = numClusters_rep + numEmptySpaces

        # Calculate the number of repetitions of the pattern
        repetitions = numClusters // pattern_length

        # Calculate the remaining elements after complete repetitions
        remaining_elements = numClusters % pattern_length

        # Create the pattern for one complete repetition
        pattern = [1] * numClusters_rep + [0] * numEmptySpaces

        # Create the ClusterPattern by repeating the pattern
        ClusterPattern = pattern * repetitions + pattern[:remaining_elements]
        print(f"Pattern for cluster repetitions: {ClusterPattern}")

        # Print all point designations
        print("Available point groups:")
        for i, key in enumerate(points.keys()):
            print(f"{i+1}. {key}")

        # Ask the user to choose a point designation
        chosen_index = int(input("Enter the number of the point group you want to choose [1]: ")) - 1

        # Get the chosen key
        chosen_keys = list(points.keys())
        chosen_key = chosen_keys[chosen_index]

        # Ask the user to change the name of the chosen key
        new_key = input(f"Enter a new name for the chosen key '{chosen_key}' (or press Enter to keep it as is): ")
        if new_key.strip() != '':
            points[new_key] = points.pop(chosen_key)  # Update the key in the points dictionary
            chosen_key = new_key

        # Assuming points is a list of your points' coordinates
        points_non_sorted = np.array([(p[0], p[1], p[2]) for p in points[chosen_key]])

        # Sort the points by their x and z coordinates
        points = sorted(points_non_sorted, key=lambda p: (p[0], p[2]))

        # Convert the sorted list to a numpy array and add a dummy point at the beginning
        points_array = np.vstack([[[0, 0, 0]], points])
        print(points_array)

        # Check if the points form a quadrilateral grid and have at least 4 points
        if len(points_array) % 2 != 0 and len(points_array) < 4:
            print("Points do not form a valid grid. The number of points must be divisible by 2 and at least 4 points are required. Aborting...")
            output_file_path = f"None, since points in '{chosen_key}' do not form a valid grid."
            return output_file_path


        # Populate points_dict
        points_dict = {i: point for i, point in enumerate(points_array)}

        # Turn on interactive mode
        plt.ion()

        # Plot the points (starting from index 1 to skip the dummy point)
        plt.plot(points_array[1:, 0], points_array[1:, 2], 'o')
        for i, point in enumerate(points_array[1:], start=1):
            plt.text(point[0], point[2], str(i))

        # Ask the user to input the indices of the points for the first square
        first_square_indices = input("Enter the indices of the points for the first square, separated by spaces.\nStart from first left point. Try highest/lowest point, Type in CW or CCW direction, until you are content with result.\nLast two points are initial points for next adjacent square: ")
        first_square_indices = list(map(int, first_square_indices.split()))

        if execution_thread.should_stop:
            break

        # Create a graph
        G = nx.Graph()

        # Add the first square to the graph
        for i in range(4):
            G.add_edge(tuple(points_array[first_square_indices[i]]), tuple(points_array[first_square_indices[(i + 1) % 4]]))

        # Draw the first square
        for edge in G.edges:
            plt.plot([edge[0][0], edge[1][0]], [edge[0][2], edge[1][2]], 'r-')

        # Keep track of the last two points that were added
        last_points = [tuple(points_array[i]) for i in first_square_indices[2:]]


        # Add the rest of the squares to the graph
        remaining_points = [tuple(p) for i, p in enumerate(points_array[1:], start=1) if i not in first_square_indices]
        connected_points = []  # List to keep track of points connected by lines
        connected_indices = []  # List to keep track of indices of points connected by lines
        for _ in range(len(remaining_points) // 2):

            # Find the next two points that should be connected
            possible_points = sorted(remaining_points, key=lambda p: min(np.linalg.norm(np.array(p) - np.array(lp)) for lp in last_points))[:4]
            possible_pairs = list(combinations(possible_points, 2))

            #Calculate the total distance for each pair of points and the distance from each point in the previous pair to each point in the new pair
            print("Possible pairs:", possible_pairs)
            print("Last points:", last_points)

            distances = []
            for pair in possible_pairs:
                # Calculate the total distance for the pair
                total_dist = total_distance(pair)
                
                # Calculate the distance from each point in the previous pair to each point in the new pair
                dist_0_0 = np.linalg.norm(np.array(pair[0]) - np.array(last_points[0]))
                dist_0_1 = np.linalg.norm(np.array(pair[0]) - np.array(last_points[1]))
                dist_1_0 = np.linalg.norm(np.array(pair[1]) - np.array(last_points[0]))
                dist_1_1 = np.linalg.norm(np.array(pair[1]) - np.array(last_points[1]))
                
                # Store the pair, the indices of the points in the last_points and next_points lists, and the total distance
                distances.append((pair, (0, 0), (1, 1), total_dist + dist_0_0 + dist_1_1))
                distances.append((pair, (1, 0), (1, 0), total_dist + dist_0_1 + dist_1_0))
                #print('Distances:', distances)

            # Find the pair of points that form the shortest path to create a quadrilateral
            next_points, connections_1, connections_2, _ = min(distances, key=lambda x: x[3])
            print('next_points:', next_points)
            print('connections_1:', connections_1)
            print('connections_2:', connections_2)

            # Add a quadrilaterale to the graph
            if connections_1[0] == connections_1[0]:
                G.add_edge(last_points[0], next_points[0])
                G.add_edge(last_points[1], next_points[1])
            else:
                G.add_edge(last_points[0], next_point[1])
                G.add_edge(last_points[1], next_points[0])

            # Add an edge between the points in the new pair
            G.add_edge(next_points[0], next_points[1])

            # Update the last two points
            last_points = next_points

            # Remove the next points from the remaining points
            remaining_points = [p for p in remaining_points if p not in next_points]

            # Draw the new quadrilateral
            for edge in G.edges:
                plt.plot([edge[0][0], edge[1][0]], [edge[0][2], edge[1][2]], 'r-')

        # Print the list of connected points and their indices
        for i, (line, indices) in enumerate(zip(connected_points, connected_indices), start=1):
            print(f"Line {i}: {line}, Indices: {indices}")

        # Update the plot
        plt.draw()
        plt.pause(5)

        if execution_thread.should_stop:
            break

        # Create lines between each pair of points
        for edge in G.edges:
            # Get the indices of the points that form the edge
            i, j = [i for i, point in enumerate(points_dict.values()) if np.array_equal(point, edge[0])][0], [i for i, point in enumerate(points_dict.values()) if np.array_equal(point, edge[1])][0]
            
            # Store the line in the lines dictionary
            max_key = max(lines_dict.keys(), key=lambda x: max(x)) if lines_dict else (0, 0)
            lines_dict[(i, j)] = np.linalg.norm(np.array(points_dict[i]) - np.array(points_dict[j]))

        # Rotate points and lines
        rotated_points_dict = {}
        rotated_lines_dict = {}
        original_num_points = len(points_dict)
        original_num_lines = len(lines_dict)

        for i, (key, value) in  enumerate(points_dict.items(), start=0):
            # Rotate the points
            x_rotated = value[0] * np.cos(rotation_angle) - value[1] * np.sin(rotation_angle)
            y_rotated = value[0] * np.sin(rotation_angle) + value[1] * np.cos(rotation_angle)
            z_rotated = value[2]  # Z coordinate remains the same after rotation around Z axis
            
            # Add the rotated points to the new dictionary
            rotated_points_dict[i + original_num_points] = np.array([x_rotated, y_rotated, z_rotated])

        for points, distance in lines_dict.items():
            # Update the lines
            point1_rotated = points[0] + original_num_points
            point2_rotated = points[1] + original_num_points
            
            # Add the rotated lines to the new dictionary
            rotated_lines_dict[(point1_rotated, point2_rotated)] = np.linalg.norm(rotated_points_dict[point1_rotated] - rotated_points_dict[point2_rotated])

        # Merge the original and rotated points and lines dictionaries
        points_dict.update(rotated_points_dict)
        lines_dict.update(rotated_lines_dict)

        # Create new lines that connect each original point with its corresponding rotated point
        for i in range(1, original_num_points):
            # Calculate the distance between the original point and the rotated point
            distance = np.linalg.norm(points_dict[i] - points_dict[i + original_num_points])
            
            # Add the new line to the lines dictionary
            lines_dict[(i, i + original_num_points)] = distance

        # Open a .geo file for writing
        output_file_path = f"model_{chosen_key}.geo"
        with open(output_file_path, "w") as f:
            f.write("Mesh.Algorithm = 6;\n")  # Set mesh algorithm
            f.write("Mesh.Format = 2.2;\n")  # Set mesh file format
            
            # Write the points to the .geo file
            for i, point in points_dict.items():
                if np.all(point == np.array([0.0, 0.0, 0.0])):  # Skip the dummy point
                    continue
                f.write(f"Point({i}) = {{{point[0]}, {point[1]}, {point[2]}, 1.0}};\n")
            
            # Initialize a graph
            H = nx.DiGraph()

            # Write the lines to the .geo file and add them to the graph
            for i, (points, distance) in enumerate(lines_dict.items(), start=1):
                f.write(f"Line({i}) = {{{points[0]}, {points[1]}}};\n")
                f.write(f"Transfinite Line {i} = 2 Using Progression 1;\n")
                H.add_edge(points[0], points[1], name=i)

            # Create a list of tuples where each tuple is (start_node, end_node, line_name)
            h_edges_list = [(u, v, d['name']) for u, v, d in H.edges(data=True)]
            print('H.edges list: ')
            print(h_edges_list)

            # Convert the directed graph to an undirected graph
            H = H.to_undirected()

            # Find all cycles of length 4 in the graph
            quad_cycles = find_quad_cycles(H)
            print('Quad point cycles of 4, undirected: ')
            print(quad_cycles)

            # Create line loops
            # Create a set to store unique line loops
            unique_line_loops = set()

            # Create a list to store all line loops
            all_line_loops = []

            for cycle in quad_cycles:
                # Get the lines that form the cycle
                lines = []
                for i in range(4):
                    start, end = cycle[i], cycle[(i+1)%4]
                    # Check if the edge is in h_edges_list
                    if (start, end) in [(u, v) for u, v, _ in h_edges_list]:
                        line_name = [d for u, v, d in h_edges_list if (u, v) == (start, end)][0]
                        lines.append(line_name)
                    elif (end, start) in [(u, v) for u, v, _ in h_edges_list]:
                        line_name = [d for u, v, d in h_edges_list if (u, v) == (end, start)][0]
                        lines.append(-line_name)
                    else:
                        print(f"Edge ({start}, {end}) not found in graph.")

                # Add lines to the list of all line loops
                all_line_loops.append(lines)

                # Create a tuple of sorted absolute line values
                lines_tuple = tuple(sorted(abs(line) for line in lines))
                # Add to the set of unique line loops
                unique_line_loops.add(lines_tuple)

            print('All_line_loops: ')   
            print(all_line_loops)    

            # Write unique line loops to the file
            surfaces = [] # Create a list of sets of lines that each unique surface consists of.
            for idx, loop in enumerate(unique_line_loops, start=1):
                # We need to restore the signs of the lines
                # This can be done by checking each line in the original list of line loops
                # This assumes that there are no two loops with the same absolute values but different signs
                loop_with_signs = None
                for lines in all_line_loops:
                    if set(map(abs, lines)) == set(loop):
                        loop_with_signs = lines
                f.write(f"Curve Loop({idx}) = {{{', '.join(map(str, loop_with_signs))}}};\n")
                f.write(f"Surface({idx}) = {{{idx}}};\n")
                f.write(f"Transfinite Surface {{{idx}}};\n")
                surfaces.append(list(loop_with_signs))
                

            # Now we build the graph.
            S = nx.Graph()
            for i, surface in enumerate(surfaces):
                S.add_node(i+1, lines=surface)

            # Add edges between nodes that share at least one line.
            for i in range(len(surfaces)):
                for j in range(i+1, len(surfaces)):
                    if len(set(map(abs, surfaces[i])).intersection(set(map(abs, surfaces[j])))) > 0:
                        S.add_edge(i+1, j+1)

            # Create a list to store the volumes
            volumes = []

            # Iterate over all sets of 6 nodes
            for nodes in combinations(S.nodes, 6):
                # Check if each node is connected to at least 3 other nodes in the set
                if all(sum(1 for neighbor in S.neighbors(node) if neighbor in nodes) >= 3 for node in nodes):
                    # If so, add the set of nodes to the list of volumes
                    volumes.append(nodes)

            # Write Surface Loops and Volumes to the .geo file
            for i, volume in enumerate(volumes, start=1):
                # Write Surface Loop
                f.write(f"Surface Loop({i}) = {{{', '.join(map(str, volume))}}};\n")
                # Write Volume
                f.write(f"Volume({i}) = {{{i}}};\n")
                f.write(f"Transfinite Volume {{{i}}};\n")
            f.write(f'Physical Volume("{chosen_key}") = {{{", ".join(map(str, range(1, len(volumes) + 1)))}}};\n')

            # Create clusters by duplicating volumes and rotating
            rotation_angle_off = 2 * np.pi / numClusters * numClusters_off
            volumes_str = "; ".join(f"Volume{{{volume}}}" for volume in range(1, len(volumes) + 1))       
            
            for i, val in enumerate(ClusterPattern, start=0):
                rotation_angle_rep = rotation_angle * i
                if val == 1:
                    if i == 0:
                        f.write(f"Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {rotation_angle_off}}} " +  f"{{ {volumes_str}; }}\n")
                    else:
                        f.write(f"Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {rotation_angle_rep}}} " +  f"{{ Duplicata {{ {volumes_str}; }} }}\n")
               
                # Check if the desired number of iterations is reached
                if i >= numClusters-1:
                    break
                
            # Join clusters
            f.write(f"Coherence;\n")

        # Printing
        print("Points: ")
        print(points_dict)
        print("Lines: ")
        print(lines_dict)
        print("H.edges: ")
        print(H.edges())
        print("Surfaces with lines: ")
        print(surfaces)
        print("Nodes of graph representing surfaces: ")
        print(S.nodes())
        print("Edges of graph represent connected nodes(surfaces): ")
        print(S.edges())
        print("Volumes: ")
        for volume in volumes:
                print(volume)
        # Turn off interactive mode
        plt.ioff()
        # Clear the plot
        plt.clf()
        # Close the figure window
        plt.close()
        return output_file_path

def start_combine(execution_thread):
    while True:
        if execution_thread.should_stop:
            break
        if file_paths:
            output_file_path = 'combined.geo'
            combine_geo_files(file_paths, output_file_path)
            print(f"Geo files combined within: {output_file_path}")

            # Clear the lists before adding new file paths
            file_paths.clear()
            file_names.clear()
        else:
            print("No geo files dropped.")

        # Update the text area after combining
        update_file_label_text()  # Update the label text
        label_text.set('Drag and Drop GEO Files Here')
        abort_button.place_forget()  # Hide the abort button
        break  # Exit the loop if execution is completed without aborting  


def start_execute(execution_thread):
    while True:
        if execution_thread.should_stop:
            break
        if file_paths:
            output_file_path = ''
            output_file_path = generate_geo_file(file_paths, output_file_path)
            print(f"Single geo file created: {output_file_path}")

            # Clear the lists before adding new file paths
            file_paths.clear()
            file_names.clear()
        else:
            print("No stp file dropped.")

        # Update the text area after combining
        update_file_label_text()  # Update the label text
        label_text.set('Drag and Drop STP File Here')
        abort_button.place_forget()  # Hide the abort button
        break  # Exit the loop if execution is completed without aborting  

def update_label_with_background():
    global label, label_text, width, height, bg_image, bg_photo, root
    # Load the background image
    bg_image = bg_image.resize((width, height), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    # Create a canvas
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack(fill='both', expand=True)

    # Create a label for the drag and drop area
    label_text = tk.StringVar()
    label_text.set('Drag and Drop STP File Here')
    label = tk.Label(root, textvariable=label_text, font=('Helvetica', 12, 'bold'), relief='flat', fg='#D4D4D4')
    label.place(relx=0.5, rely=0.2, anchor='center')

    # Set the label background image
    label.config(bg='black', image=bg_photo, compound='center', padx=90, pady=90)

def create_gui():
    global selected_code, execution_thread, code_switch, root, handle_drop, DND_FILES, canvas, bg_photo, bg_image, label, label_text, update_file_label_text, abort_button

    # Define the dimensions
    width = 300
    height = 300

    root = TkinterDnD.Tk()
    root.title('Create or Combine gmsh GEO Files')
    root.geometry(f'{width}x{height}')

    update_label_with_background()

    def update_file_label_text():
        global label_text, file_names
        label_text.set('\n'.join(file_names))  # Use file_names instead of file_paths

    def abort_execution():
        if execution_thread and execution_thread.is_alive():
            # Set a flag to stop the execution thread
            execution_thread.should_stop = True
            # Clear the terminal to indicate the abortion
            #clear_terminal()
            print("\n \nExecution aborted, close the terminal.\nExiting application...")

            # Clear the lists before adding new file paths
            file_paths.clear()
            file_names.clear()
            update_label_text()  # Update the label text
            abort_button.place_forget()  # Hide the abort button

            # Re-enable the file drop functionality
            enable_drop()

            # Delay the closing of the whole program for 5 seconds (5000 milliseconds)
            root.after(2000, root.destroy)

    def handle_drop(event):
        file = event.data.strip("{}")  # Remove {} from the file path
        print(f"Dropped file: {file}")  # Print the dropped file

        selected_option = code_switch.get()
        if selected_option == 'GEO Files Input':
            if file.lower().endswith('.geo'):
                file_name = os.path.basename(file)  # Get the file name
                file_paths.append(file)  # Add the full file path to the file_paths list
                file_names.append(file_name)  # Add the file name to the file_names list
                print(f"Added file: {file_name}")  # Print the added file name
                update_file_label_text()  # Update the label text
            else:
                print("Invalid file format. Only GEO files are allowed.")

        elif selected_option == 'STP File Input':
            if file.lower().endswith('.stp'):
                file_name = os.path.basename(file)  # Get the file name
                file_paths.clear()  # Clear the file_paths list
                file_paths.append(file)  # Add the full file path to the file_paths list
                file_names.clear()  # Clear the file_names list
                file_names.append(file_name)  # Add the file name to the file_names list
                print(f"Added file: {file_name}")  # Print the added file name
                update_file_label_text()  # Update the label text
            else:
                print("Invalid file format. Only STP files are allowed.")

    def enable_drop():
        global root, label, handle_drop, DND_FILES
        # Allow dropping files onto the label and canvas
        root.drop_target_register(DND_FILES)
        root.dnd_bind('<<Drop>>', handle_drop)
        label.drop_target_register(DND_FILES)
        label.dnd_bind('<<Drop>>', handle_drop)
    
    # Re-enable the file drop functionality
    enable_drop()
    
    def execute_code():
        global selected_code, execution_thread 
        selected_code = code_switch.get()
        execution_thread = StoppableThread()  # Initialize the thread first
        clear_terminal()
        if selected_code == 'GEO Files Input':
            execution_thread = StoppableThread(target=start_combine, args=(execution_thread,))
        elif selected_code == 'STP File Input':
            execution_thread = StoppableThread(target=start_execute, args=(execution_thread,))
        execution_thread.start()
        abort_button.place(relx=0.5, rely=0.3, anchor='center')  # Show the abort button


    def update_label_text():
        global label_text
        selected_option = code_switch.get()
        if selected_option == 'STP File Input':
            label_text.set('Drag and Drop STP Files Here')
        elif selected_option == 'GEO Files Input':
            label_text.set('Drag and Drop GEO Files Here')    

    def update_button_text(event):
        global label_text
        selected_option = code_switch.get()
        if selected_option == 'GEO Files Input':
            execute_button.config(text='Combine')
        elif selected_option == 'STP File Input':
            execute_button.config(text='Execute')
        update_label_text()  # Update the label text immediately

    # Create a switch button to select which code to execute
    code_switch = tk.StringVar()
    code_switch.set('STP File Input')  # Default value
    switch_button = ttk.Combobox(root, textvariable=code_switch, values=['STP File Input', 'GEO Files Input'])
    switch_button.place(relx=0.5, rely=0.8, anchor='center')

    # Create an execute button to execute the selected code
    execute_button = ttk.Button(root, text='Execute', command=execute_code)
    execute_button.place(relx=0.5, rely=0.9, anchor='center')

    # Create the "Abort" button and position it on the GUI
    abort_button = ttk.Button(root, text='Abort', command=abort_execution)
    abort_button.place(relx=0.5, rely=0.3, anchor='center')
    abort_button.place_forget()  # Hide the abort button initially

    # Updating text in switch button immediately after selection
    switch_button.bind('<<ComboboxSelected>>', update_button_text)  # Bind the selected event

    root.mainloop()

if __name__ == '__main__':
    create_gui()