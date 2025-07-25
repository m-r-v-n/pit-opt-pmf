############################################################
# PyMaxflow Pit Optimiser - Boykov–Kolmogorov (BK) Algorithm
#
#
#
#
#
############################################################

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import sin, cos, tan, radians
import maxflow
import os

# Sending nodes to Sink or Source
def sendNodes(BM, g, node_ids, BVal):
    """
    Adds source/sink capacities to PyMaxflow graph based on block economic value.

    Parameters:
    - BM: numpy array block model (rows: blocks, columns: attributes)
    - g: PyMaxflow Graph instance
    - node_ids: List of PyMaxflow node IDs for each block
    - BVal: Index of the column in BM containing economic value (profit/loss)
    """
    start_UPL = time.time()
    total_blocks = BM.shape[0]

    source_count = 0
    sink_count = 0

    for i in range(total_blocks):
        profit = BM[i, BVal]
        capacity = abs(round(profit, 2))

        if profit < 0:
            g.add_tedge(node_ids[i], 0, capacity)  # block → sink
            sink_count += 1
        else:
            g.add_tedge(node_ids[i], capacity, 0)  # source → block
            source_count += 1

    print(f"--> External arcs (source/sink) added in {np.round(time.time() - start_UPL, 2)} seconds")
    print(f"Total blocks processed: {total_blocks}")
    print(f"Connected to source: {source_count}, to sink: {sink_count}")

    return g

# Arc Precedence
def createArcPrecedence(BM,
                        idx,
                        xsize,ysize,zsize,
                        xmin,ymin,zmin,
                        xmax,ymax,zmax,
                        xcol,ycol,zcol,
                        slopecol,
                        num_blocks_above,
                        g,  # PyMaxflow graph
                        node_ids,  # list of PyMaxflow node IDs aligned with BM
                        minWidth):

    start_UPL = time.time()

    BM1 = BM[:, [idx, xcol, ycol, zcol]]
    block_to_value = {(x, y, z): value for value, x, y, z in BM1}

    BM2 = BM[:, [xcol, ycol, zcol, slopecol]]

    internal_arc = 0
    nodes_with_edge = 0

    for i, (x_i, y_i, z_i, angle_i) in enumerate(BM2, start=1):
        min_radius = minWidth / 2

        if z_i == zmax and min_radius == 0:
            continue

        cone_height = zsize * num_blocks_above
        cone_radius = cone_height / math.tan(math.radians(angle_i))
        search_x = int(np.ceil((cone_radius - (xsize/2)) / xsize))
        search_y = int(np.ceil((cone_radius - (ysize/2)) / ysize))

        x_range = range(-int(min(((x_i - xmin)/xsize), search_x)),
                        int(min(((xmax - x_i)/xsize) + 1, search_x + 1)))
        y_range = range(-int(min(((y_i - ymin)/ysize), search_y)),
                        int(min(((ymax - y_i)/ysize) + 1, search_y + 1)))
        z_range = range(0, int(min(((zmax - z_i)/zsize) + 1, num_blocks_above + 1)))

        block_coords = np.array([
            (x_i + j * xsize, y_i + k * ysize, z_i + l * zsize)
            for j in x_range
            for k in y_range
            for l in z_range
        ])

        if block_coords.size == 0:
            continue

        dists = np.sqrt((block_coords[:, 0] - x_i) ** 2 + (block_coords[:, 1] - y_i) ** 2)
        heights = block_coords[:, 2] - z_i
        with np.errstate(divide='ignore', invalid='ignore'):
            cone_radii = heights * cone_radius / cone_height

        if min_radius > 0:
            cone_radii = np.where(heights == 0, minWidth, cone_radii + min_radius)

        inside_indices = np.where(dists <= cone_radii)[0]
        inside_blocks = block_coords[inside_indices]

        connected = 0
        source_key = (x_i, y_i, z_i)
        if source_key not in block_to_value:
            continue
        source = int(block_to_value[source_key])

        for block in inside_blocks:
            block_key = tuple(block)
            if block_key not in block_to_value:
                continue

            target = int(block_to_value[block_key])
            if source == target:
                continue  # skip self-loop

            g.add_edge(node_ids[source], node_ids[target], float('inf'), 0)
            connected += 1
            internal_arc += 1

        if connected > 0:
            nodes_with_edge += 1
            arc_rate = np.around(internal_arc / (time.time() - start_UPL), 2)
            print(f"index = {i} node = {source} connected arcs = {connected} total arcs = {internal_arc} x = {x_i} y = {y_i} z = {z_i} angle = {angle_i} arc rate = {arc_rate}/s")

    print("\nPerformance:")
    print(f"--- Total Nodes Processed: {i}")
    print(f"--- Nodes With Edges: {nodes_with_edge}")
    print(f"--- Total Precedence Arcs: {internal_arc}")
    print(f"--- Average Arcs per Node: {np.around(internal_arc / nodes_with_edge, 0)} arcs/node")
    print(f"--- Precedence Arc Generation Rate: {np.around(internal_arc / (time.time() - start_UPL), 2)} arcs/s")
    print(f"--> Precedence Arc Generation time: {np.round(time.time() - start_UPL, 2)} seconds")

    return g

def PyMaxflow_UPL(BM,
                   idx,
                   xsize,ysize,zsize,
                   xmin,ymin,zmin,
                   xmax,ymax,zmax,
                   xcol,ycol,zcol,
                   slopecol,
                   num_blocks_above,
                   BVal,
                   pitLimit,
                   Cashflow,
                   minWidth):

    print("Process Start...")
    start_UPL = time.time()

    x_coords = BM[:, xcol]
    y_coords = BM[:, ycol]
    z_coords = BM[:, zcol]

    # create graph with enough vertices

    g = maxflow.Graph[float]()
    node_ids = g.add_nodes(len(BM))  # BM = your NumPy block model

    # Connecting nodes
    print("Sending Nodes")
    g = sendNodes(BM, g, node_ids, BVal)
    print("External Arcs done")

    print("Creating Precedence")

    g = createArcPrecedence(BM,
                            idx,
                            xsize,ysize,zsize,
                            xmin,ymin,zmin,
                            xmax,ymax,zmax,
                            xcol,ycol,zcol,
                            slopecol,
                            num_blocks_above,
                            g,  # PyMaxflow graph
                            node_ids,  # list of PyMaxflow node IDs aligned with BM
                            minWidth)

    # Optimisation using pseudoflow
    print("Solving Ultimate Pit Limit")
    solve_UPL = time.time()
    # 1. Solve UPL using PyMaxflow
    max_profit = g.maxflow()

    # 2. Get selected blocks (in pit = source side of cut)
    in_pit = [1 if g.get_segment(node_ids[i]) == 0 else 0 for i in range(len(node_ids))]

    # 3. Initialize pit limit and cashflow columns
    BM[:, pitLimit] = 0
    BM[:, Cashflow] = 0

    # 4. Assign 1 to pitLimit where block is in pit, and copy value to Cashflow
    for i in range(len(in_pit)):
        if in_pit[i] == 1:
            BM[i, pitLimit] = 1
            BM[i, Cashflow] = BM[i, BVal]

    # 5. Calculate total undiscounted cashflow
    cashFlow = "{:,.2f}".format(np.sum(BM[:, Cashflow]))

    # 6. Print performance info
    print("--> PyMaxflow Optimization time: --%s seconds " % np.round((time.time() - solve_UPL), 2))
    print("--> Total process time: --%s seconds " % np.round((time.time() - start_UPL), 2))
    print(f"Undiscounted Cashflow: ${cashFlow}")

    return BM

def main():
    print("Start")
    start_time = time.time()

########################################################################################

    # 1. Block model location
    filePath = 'marvin_pmf.csv'

    # 2. Block model size
    xsize = 30
    ysize = 30
    zsize = 30

    # 3. Column number of xyz coordinates. note that column number starts at 0
    xcol = 1
    ycol = 2
    zcol = 3

    # 4. Block search boundary parameters
    num_blocks_above = 6

    # 5. Minimum mining width for pit bottom consideration (this will be added to the radius of the search cone)
    minWidth = 0.0

    # 6. Column number of Block Value (column number starts at 0)
    BVal = 8

    # 7. Column numbler of Slope Angle
    slopecol = 5

    # 8. Column numbler of Index ID
    idx = 0

########################################################################################

    data = np.loadtxt(filePath, delimiter=',', skiprows=1) # Import Block Model

    x_col = data[:, xcol]
    y_col = data[:, ycol]
    z_col = data[:, zcol]

    xmin = x_col.min()
    xmax = x_col.max()

    ymin = y_col.min()
    ymax = y_col.max()

    zmin = z_col.min()
    zmax = z_col.max()

    nx = ((xmax - xmin) / xsize) + 1
    ny = ((ymax - ymin) / ysize) + 1
    nz = ((zmax - zmin) / zsize) + 1


    orig_col = data.shape[1]
    print(f"Original Column = {orig_col}")

    # Add two new columns for pitLimit and CashFlow
    n_rows = data.shape[0]
    col1 = np.zeros((n_rows, 1))
    col2 = np.zeros((n_rows, 1))
    data = np.hstack((data, col1, col2))

    # Store column numbers (indices) of new columns
    pitLimit = orig_col      # first new column
    CashFlow = orig_col + 1  # second new column
    print(f"UPL Column = {pitLimit}")
    print(f"Cashflow Column = {CashFlow}")

    BlockModel = data

    # Call PyMaxflow function
    BlockModel = PyMaxflow_UPL(BlockModel,
                                idx,
                                xsize,ysize,zsize,
                                xmin,ymin,zmin,
                                xmax,ymax,zmax,
                                xcol,ycol,zcol,
                                slopecol,
                                num_blocks_above,
                                BVal,
                                pitLimit,
                                CashFlow,
                                minWidth
                                )

    # Save Block Model
    base, ext = os.path.splitext(filePath)

    np.savetxt(
        fname=f"{base}_opt{ext}",
        X=BlockModel,
        fmt='%.3f',
        delimiter=',',
        header="id,X,Y,Z,tonne,slope,au_ppm,cu_pct,block_val,pit_limit,cash_flow",
        comments=''  # <- this removes the default '#' comment character
    )

if __name__ == "__main__":
    main()
