"""
VLSI Cell Placement Optimization Challenge
==========================================

CHALLENGE OVERVIEW:
You are tasked with implementing a critical component of a chip placement optimizer.
Given a set of cells (circuit components) with fixed sizes and connectivity requirements,
you need to find positions for these cells that:
1. Minimize total wirelength (wiring cost between connected pins)
2. Eliminate all overlaps between cells

YOUR TASK:
Implement the `overlap_repulsion_loss()` function to prevent cells from overlapping.
The function must:
- Be differentiable (uses PyTorch operations for gradient descent)
- Detect when cells overlap in 2D space
- Apply increasing penalties for larger overlaps
- Work efficiently with vectorized operations

SUCCESS CRITERIA:
After running the optimizer with your implementation:
- overlap_count should be 0 (no overlapping cell pairs)
- total_overlap_area should be 0.0 (no overlap)
- wirelength should be minimized
- Visualization should show clean, non-overlapping placement

GETTING STARTED:
1. Read through the existing code to understand the data structures
2. Look at wirelength_attraction_loss() as a reference implementation
3. Implement overlap_repulsion_loss() following the TODO instructions
4. Run main() and check the overlap metrics in the output
5. Tune hyperparameters (lambda_overlap, lambda_wirelength) if needed
6. Generate visualization to verify your solution

BONUS CHALLENGES:
- Improve convergence speed by tuning learning rate or adding momentum
- Implement better initial placement strategy
- Add visualization of optimization progress over time
"""

import os
from enum import IntEnum

import torch
import torch.optim as optim


# Feature index enums for cleaner code access
class CellFeatureIdx(IntEnum):
    """Indices for cell feature tensor columns."""
    AREA = 0
    NUM_PINS = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5


class PinFeatureIdx(IntEnum):
    """Indices for pin feature tensor columns."""
    CELL_IDX = 0
    PIN_X = 1  # Relative to cell corner
    PIN_Y = 2  # Relative to cell corner
    X = 3  # Absolute position
    Y = 4  # Absolute position
    WIDTH = 5
    HEIGHT = 6


# Configuration constants
# Macro parameters
MIN_MACRO_AREA = 100.0
MAX_MACRO_AREA = 10000.0

# Standard cell parameters (areas can be 1, 2, or 3)
STANDARD_CELL_AREAS = [1.0, 2.0, 3.0]
STANDARD_CELL_HEIGHT = 1.0

# Pin count parameters
MIN_STANDARD_CELL_PINS = 3
MAX_STANDARD_CELL_PINS = 6

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======= SETUP =======

def generate_placement_input(num_macros, num_std_cells):
    """Generate synthetic placement input data.

    Args:
        num_macros: Number of macros to generate
        num_std_cells: Number of standard cells to generate

    Returns:
        Tuple of (cell_features, pin_features, edge_list):
            - cell_features: torch.Tensor of shape [N, 6] with columns [area, num_pins, x, y, width, height]
            - pin_features: torch.Tensor of shape [total_pins, 7] with columns
              [cell_instance_index, pin_x, pin_y, x, y, pin_width, pin_height]
            - edge_list: torch.Tensor of shape [E, 2] with [src_pin_idx, tgt_pin_idx]
    """
    total_cells = num_macros + num_std_cells

    # Step 1: Generate macro areas (uniformly distributed between min and max)
    macro_areas = (
        torch.rand(num_macros) * (MAX_MACRO_AREA - MIN_MACRO_AREA) + MIN_MACRO_AREA
    )

    # Step 2: Generate standard cell areas (randomly pick from 1, 2, or 3)
    std_cell_areas = torch.tensor(STANDARD_CELL_AREAS)[
        torch.randint(0, len(STANDARD_CELL_AREAS), (num_std_cells,))
    ]

    # Combine all areas
    areas = torch.cat([macro_areas, std_cell_areas])

    # Step 3: Calculate cell dimensions
    # Macros are square
    macro_widths = torch.sqrt(macro_areas)
    macro_heights = torch.sqrt(macro_areas)

    # Standard cells have fixed height = 1, width = area
    std_cell_widths = std_cell_areas / STANDARD_CELL_HEIGHT
    std_cell_heights = torch.full((num_std_cells,), STANDARD_CELL_HEIGHT)

    # Combine dimensions
    cell_widths = torch.cat([macro_widths, std_cell_widths])
    cell_heights = torch.cat([macro_heights, std_cell_heights])

    # Step 4: Calculate number of pins per cell
    num_pins_per_cell = torch.zeros(total_cells, dtype=torch.int)

    # Macros: between sqrt(area) and 2*sqrt(area) pins
    for i in range(num_macros):
        sqrt_area = int(torch.sqrt(macro_areas[i]).item())
        num_pins_per_cell[i] = torch.randint(sqrt_area, 2 * sqrt_area + 1, (1,)).item()

    # Standard cells: between 3 and 6 pins
    num_pins_per_cell[num_macros:] = torch.randint(
        MIN_STANDARD_CELL_PINS, MAX_STANDARD_CELL_PINS + 1, (num_std_cells,)
    )

    # Step 5: Create cell features tensor [area, num_pins, x, y, width, height]
    cell_features = torch.zeros(total_cells, 6)
    cell_features[:, CellFeatureIdx.AREA] = areas
    cell_features[:, CellFeatureIdx.NUM_PINS] = num_pins_per_cell.float()
    cell_features[:, CellFeatureIdx.X] = 0.0  # x position (initialized to 0)
    cell_features[:, CellFeatureIdx.Y] = 0.0  # y position (initialized to 0)
    cell_features[:, CellFeatureIdx.WIDTH] = cell_widths
    cell_features[:, CellFeatureIdx.HEIGHT] = cell_heights

    # Step 6: Generate pins for each cell
    total_pins = num_pins_per_cell.sum().item()
    pin_features = torch.zeros(total_pins, 7)

    # Fixed pin size for all pins (square pins)
    PIN_SIZE = 0.1  # All pins are 0.1 x 0.1

    pin_idx = 0
    for cell_idx in range(total_cells):
        n_pins = num_pins_per_cell[cell_idx].item()
        cell_width = cell_widths[cell_idx].item()
        cell_height = cell_heights[cell_idx].item()

        # Generate random pin positions within the cell
        # Offset from edges to ensure pins are fully inside
        margin = PIN_SIZE / 2
        if cell_width > 2 * margin and cell_height > 2 * margin:
            pin_x = torch.rand(n_pins) * (cell_width - 2 * margin) + margin
            pin_y = torch.rand(n_pins) * (cell_height - 2 * margin) + margin
        else:
            # For very small cells, just center the pins
            pin_x = torch.full((n_pins,), cell_width / 2)
            pin_y = torch.full((n_pins,), cell_height / 2)

        # Fill pin features
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.CELL_IDX] = cell_idx
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_X] = (
            pin_x  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_Y] = (
            pin_y  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.X] = (
            pin_x  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.Y] = (
            pin_y  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.WIDTH] = PIN_SIZE
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.HEIGHT] = PIN_SIZE

        pin_idx += n_pins

    # Step 7: Generate edges with simple random connectivity
    # Each pin connects to 1-3 random pins (preferring different cells)
    edge_list = []
    avg_edges_per_pin = 2.0

    pin_to_cell = torch.zeros(total_pins, dtype=torch.long)
    pin_idx = 0
    for cell_idx, n_pins in enumerate(num_pins_per_cell):
        pin_to_cell[pin_idx : pin_idx + n_pins] = cell_idx
        pin_idx += n_pins

    # Create adjacency set to avoid duplicate edges
    adjacency = [set() for _ in range(total_pins)]

    for pin_idx in range(total_pins):
        pin_cell = pin_to_cell[pin_idx].item()
        num_connections = torch.randint(1, 4, (1,)).item()  # 1-3 connections per pin

        # Try to connect to pins from different cells
        for _ in range(num_connections):
            # Random candidate
            other_pin = torch.randint(0, total_pins, (1,)).item()

            # Skip self-connections and existing connections
            if other_pin == pin_idx or other_pin in adjacency[pin_idx]:
                continue

            # Add edge (always store smaller index first for consistency)
            if pin_idx < other_pin:
                edge_list.append([pin_idx, other_pin])
            else:
                edge_list.append([other_pin, pin_idx])

            # Update adjacency
            adjacency[pin_idx].add(other_pin)
            adjacency[other_pin].add(pin_idx)

    # Convert to tensor and remove duplicates
    if edge_list:
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_list = torch.unique(edge_list, dim=0)
    else:
        edge_list = torch.zeros((0, 2), dtype=torch.long)

    print(f"\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")

    return cell_features, pin_features, edge_list

# ======= OPTIMIZATION CODE (edit this part) =======

# ==== 2D PACKING ALGORITHMS ====
def skyline_packing_placement(cell_features, margin=0):
    """
    Simple skyline packing for initial placement with zero overlaps.
    Uses a greedy bottom-left heuristic to pack rectangles efficiently.
    
    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        margin: Minimum spacing between cells
        
    Returns:
        Updated cell_features with new (x, y) positions
    """
    N = cell_features.shape[0]
    if N == 0:
        return cell_features
    
    # Clone to avoid modifying original
    cell_features = cell_features.clone()
    
    # Extract dimensions
    widths = cell_features[:, CellFeatureIdx.WIDTH]
    heights = cell_features[:, CellFeatureIdx.HEIGHT]
    areas = cell_features[:, CellFeatureIdx.AREA]
    
    # Sort by area (largest first for better packing)
    sorted_indices = torch.argsort(areas, descending=True)
    
    # Calculate bin width from total area
    total_area = areas.sum().item()
    bin_width = (total_area * 1.3 * 1.6) ** 0.5  # Extra space to ensure everything fits
    
    # Skyline: list of (x_start, y_height, width) tuples
    skyline = [[0.0, 0.0, bin_width]]
    
    # Pack each cell
    for idx in sorted_indices:
        w = widths[idx].item() + 2 * margin
        h = heights[idx].item() + 2 * margin
        
        # Find best position: lowest y first, then leftmost x
        best_x = None
        best_y = float('inf')
        best_idx = -1
        
        for i in range(len(skyline)):
            seg_x, seg_y, seg_w = skyline[i]
            
            # Can't fit if starting position is too far right
            if seg_x + w > bin_width:
                continue
            
            # Find the maximum y across all segments this rect would cover
            y_max = seg_y
            x_end = seg_x + w
            
            for j in range(i, len(skyline)):
                if skyline[j][0] >= x_end:
                    break
                y_max = max(y_max, skyline[j][1])
            
            # Update best position
            if y_max < best_y or (y_max == best_y and seg_x < best_x):
                best_y = y_max
                best_x = seg_x
                best_idx = i
        
        # Fallback if no position found (should not happen with enough bin_width)
        if best_x is None:
            best_x = 0.0
            best_y = max(seg[1] for seg in skyline)
        
        # Build new skyline after placing rectangle
        x_rect_end = best_x + w
        new_skyline = []
        
        # Add new segment for the placed rectangle
        new_seg_added = False
        
        for seg_x, seg_y, seg_w in skyline:
            seg_x_end = seg_x + seg_w
            
            # Case 1: Segment entirely before rectangle
            if seg_x_end <= best_x:
                new_skyline.append([seg_x, seg_y, seg_w])
            
            # Case 2: Segment entirely after rectangle
            elif seg_x >= x_rect_end:
                if not new_seg_added:
                    new_skyline.append([best_x, best_y + h, w])
                    new_seg_added = True
                new_skyline.append([seg_x, seg_y, seg_w])
            
            # Case 3: Segment partially before rectangle (left side)
            elif seg_x < best_x < seg_x_end:
                # Keep the left part
                new_skyline.append([seg_x, seg_y, best_x - seg_x])
                
                # Add rectangle segment if not added yet
                if not new_seg_added:
                    new_skyline.append([best_x, best_y + h, w])
                    new_seg_added = True
                
                # If segment extends past rectangle, keep right part
                if seg_x_end > x_rect_end:
                    new_skyline.append([x_rect_end, seg_y, seg_x_end - x_rect_end])
            
            # Case 4: Segment partially after rectangle (right side)
            elif seg_x < x_rect_end < seg_x_end:
                if not new_seg_added:
                    new_skyline.append([best_x, best_y + h, w])
                    new_seg_added = True
                new_skyline.append([x_rect_end, seg_y, seg_x_end - x_rect_end])
            
            # Case 5: Segment completely covered by rectangle - skip it
        
        # Add rectangle segment if haven't yet (edge case)
        if not new_seg_added:
            new_skyline.append([best_x, best_y + h, w])
        
        # Merge adjacent segments with same height
        merged = []
        for seg in new_skyline:
            if merged and abs(merged[-1][1] - seg[1]) < 1e-9 and abs(merged[-1][0] + merged[-1][2] - seg[0]) < 1e-9:
                # Merge with previous
                merged[-1][2] += seg[2]
            else:
                merged.append(seg)
        
        skyline = merged
        
        # Store CENTER position of cell (not corner)
        cell_features[idx, CellFeatureIdx.X] = best_x + w/2 - margin
        cell_features[idx, CellFeatureIdx.Y] = best_y + h/2 - margin
    
    return cell_features

def shelf_packing_placement(cell_features, margin=0.0):
    """
    Shelf packing: simplified skyline that packs in horizontal rows.
    Good balance of simplicity and efficiency.
    
    Args:
        cell_features: [N, 6] tensor with cell properties
        margin: Spacing between cells
        
    Returns:
        Updated cell_features with positions
    """
    N = cell_features.shape[0]
    if N == 0:
        return cell_features
    
    cell_features = cell_features.clone()
    
    widths = cell_features[:, CellFeatureIdx.WIDTH]
    heights = cell_features[:, CellFeatureIdx.HEIGHT]
    areas = cell_features[:, CellFeatureIdx.AREA]
    
    # Sort by height (tallest first) for better shelf packing
    sorted_indices = torch.argsort(heights, descending=True)
    
    # Calculate target width
    total_area = areas.sum().item()
    target_width = (total_area * 1.25 * 1.5) ** 0.5
    
    # Shelf packing state
    shelf_x = 0.0        # Current x position in shelf
    shelf_y = 0.0        # Bottom of current shelf
    shelf_height = 0.0   # Height of tallest item in current shelf
    
    for idx in sorted_indices:
        w = widths[idx].item() + 2 * margin
        h = heights[idx].item() + 2 * margin
        
        # Start new shelf if this item doesn't fit
        if shelf_x + w > target_width and shelf_x > 0:
            shelf_x = 0.0
            shelf_y += shelf_height
            shelf_height = h
        else:
            shelf_height = max(shelf_height, h)
        
        # Place cell at center
        cell_features[idx, CellFeatureIdx.X] = shelf_x + w/2 - margin
        cell_features[idx, CellFeatureIdx.Y] = shelf_y + h/2 - margin
        
        # Advance x position
        shelf_x += w
    
    return cell_features

def rects_overlap(a, b, eps=1e-9):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx + eps or bx + bw <= ax + eps or
                ay + ah <= by + eps or by + bh <= ay + eps)

def contained(a, b, eps=1e-9):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax + eps >= bx and ay + eps >= by and ax + aw <= bx + bw + eps and ay + ah <= by + bh + eps

def merge_free_rects(free_rects, eps=1e-9):
    # simple O(n^2) merge of adjacent rectangles with same edge
    merged = True
    while merged:
        merged = False
        n = len(free_rects)
        i = 0
        while i < n:
            a = free_rects[i]
            j = i + 1
            while j < n:
                b = free_rects[j]
                # merge horizontally
                if abs(a[1] - b[1]) < eps and abs(a[3] - b[3]) < eps and (abs(a[0] + a[2] - b[0]) < eps or abs(b[0] + b[2] - a[0]) < eps):
                    x = min(a[0], b[0]); y = a[1]; w = max(a[0]+a[2], b[0]+b[2]) - x; h = a[3]
                    free_rects[i] = [x, y, w, h]; free_rects.pop(j); n -= 1; merged = True; continue
                # merge vertically
                if abs(a[0] - b[0]) < eps and abs(a[2] - b[2]) < eps and (abs(a[1] + a[3] - b[1]) < eps or abs(b[1] + b[3] - a[1]) < eps):
                    x = a[0]; y = min(a[1], b[1]); w = a[2]; h = max(a[1]+a[3], b[1]+b[3]) - y
                    free_rects[i] = [x, y, w, h]; free_rects.pop(j); n -= 1; merged = True; continue
                j += 1
            i += 1
    return free_rects

def maxrects_packing_placement(cell_features, margin=0.0, bin_growth_factor=2.0):
    if cell_features.shape[0] == 0:
        return cell_features
    cell_features = cell_features.clone()
    widths = (cell_features[:, 4].float() + 2 * margin).tolist()
    heights = (cell_features[:, 5].float() + 2 * margin).tolist()
    areas = cell_features[:, 0].float()
    order = torch.argsort(areas, descending=True).tolist()

    total_area = float(areas.sum().item())
    side = math.sqrt(total_area * 1.5) if total_area > 0 else 1.0
    bin_w = bin_h = max(1.0, side)

    free_rects = [[0.0, 0.0, bin_w, bin_h]]
    placed = []  # list of placed rects (x, y, w, h) in lower-left coords

    for idx in order:
        w, h = widths[idx], heights[idx]
        placed_ok = False

        # attempt placing; allow trying multiple candidates
        def try_place_once():
            nonlocal free_rects, placed, placed_ok, bin_w, bin_h
            # score candidates: (short_side_fit, area_fit, free_rect_index, fx, fy)
            candidates = []
            for i, fr in enumerate(free_rects):
                fx, fy, fw, fh = fr
                if w <= fw + 1e-9 and h <= fh + 1e-9:
                    short_side = min(fw - w, fh - h)
                    area_fit = fw * fh - w * h
                    candidates.append((short_side, area_fit, i, fx, fy, fw, fh))
            if not candidates:
                return False
            # sort by short_side then area
            candidates.sort(key=lambda x: (x[0], x[1]))
            for cand in candidates:
                _, _, fr_i, fx, fy, fw, fh = cand
                bx, by = fx, fy  # place at lower-left of free rect (can be improved)
                placed_rect = (bx, by, w, h)
                # collision check with existing placed rects
                collide = False
                for p in placed:
                    if rects_overlap(placed_rect, p):
                        collide = True; break
                if collide:
                    continue
                # commit place
                cell_features[idx, 2] = bx + w/2 - margin
                cell_features[idx, 3] = by + h/2 - margin
                placed.append(placed_rect)
                # split all free_rects by this placed_rect
                new_free = []
                for fr in free_rects:
                    fx, fy, fw, fh = fr
                    # no intersection -> keep
                    if bx >= fx + fw - 1e-9 or bx + w <= fx + 1e-9 or by >= fy + fh - 1e-9 or by + h <= fy + 1e-9:
                        new_free.append(fr)
                        continue
                    # produce fragments (clipped)
                    # left strip
                    if bx > fx + 1e-9:
                        new_free.append([fx, fy, bx - fx, fh])
                    # right strip
                    if bx + w < fx + fw - 1e-9:
                        new_free.append([bx + w, fy, fx + fw - (bx + w), fh])
                    # bottom strip
                    if by > fy + 1e-9:
                        left = max(fx, bx); right = min(fx + fw, bx + w)
                        if right - left > 1e-9:
                            new_free.append([left, fy, right - left, by - fy])
                        else:
                            new_free.append([fx, fy, fw, by - fy])
                    # top strip
                    if by + h < fy + fh - 1e-9:
                        left = max(fx, bx); right = min(fx + fw, bx + w)
                        if right - left > 1e-9:
                            new_free.append([left, by + h, right - left, fy + fh - (by + h)])
                        else:
                            new_free.append([fx, by + h, fw, fy + fh - (by + h)])
                # prune degenerate and contained
                cleaned = []
                for r in new_free:
                    rx, ry, rw, rh = r
                    if rw <= 1e-9 or rh <= 1e-9:
                        continue
                    skip = False
                    for o in new_free:
                        if r is o: continue
                        if contained(r, o):
                            skip = True; break
                    if not skip:
                        cleaned.append(r)
                # merge nearby/adjacent free rects
                free_rects = merge_free_rects(cleaned)
                placed_ok = True
                return True
            return False

        # try place; if fails expand bin and retry up to a few times
        attempts = 0
        while not placed_ok and attempts < 4:
            if try_place_once():
                break
            # expand bin and add new free area on the right and top
            attempts += 1
            old_w, old_h = bin_w, bin_h
            bin_w *= bin_growth_factor
            bin_h *= bin_growth_factor
            # add new free strips
            free_rects.append([0.0, old_h, bin_w, bin_h - old_h])
            free_rects.append([old_w, 0.0, bin_w - old_w, old_h])
            free_rects = merge_free_rects(free_rects)

        if not placed_ok:
            # as a fallback place off-grid to avoid abortion (shouldn't happen with growth)
            bx, by = 0.0, bin_h
            cell_features[idx, 2] = bx + w/2 - margin
            cell_features[idx, 3] = by + h/2 - margin
            placed.append((bx, by, w, h))
            bin_h += h
            free_rects.append([0.0, by + h, bin_w, bin_h - (by + h)])
            free_rects = merge_free_rects(free_rects)

    # final sanity: assert no overlaps
    for i in range(len(placed)):
        for j in range(i+1, len(placed)):
            if rects_overlap(placed[i], placed[j]):
                raise RuntimeError("Overlap detected after packing. This should not happen.")

    return cell_features

# ==== LOSS FUNCTIONS ====

def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    """Calculate loss based on total wirelength to minimize routing.

    This is a REFERENCE IMPLEMENTATION showing how to write a differentiable loss function.

    The loss computes the Manhattan distance between connected pins and minimizes
    the total wirelength across all edges.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges

    Returns:
        Scalar loss value
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Update absolute pin positions based on cell positions
    cell_positions = cell_features[:, 2:4]  # [N, 2]
    cell_indices = pin_features[:, 0].long()

    # Calculate absolute pin positions
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    # Get source and target pin positions for each edge
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    # Calculate smooth approximation of Manhattan distance
    # Using log-sum-exp approximation for differentiability
    alpha = 0.1  # Smoothing parameter
    dx = torch.abs(src_x - tgt_x)
    dy = torch.abs(src_y - tgt_y)

    # Smooth L1 distance with numerical stability
    smooth_manhattan = alpha * torch.logsumexp(
        torch.stack([dx / alpha, dy / alpha], dim=0), dim=0
    )

    # Total wirelength
    total_wirelength = torch.sum(smooth_manhattan)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


def overlap_repulsion_loss(cell_features, pin_features, edge_list):
    """Calculate loss to prevent cell overlaps.

    TODO: IMPLEMENT THIS FUNCTION

    This is the main challenge. You need to implement a differentiable loss function
    that penalizes overlapping cells. The loss should:

    1. Be zero when no cells overlap
    2. Increase as overlap area increases
    3. Use only differentiable PyTorch operations (no if statements on tensors)
    4. Work efficiently with vectorized operations

    HINTS:
    - Two axis-aligned rectangles overlap if they overlap in BOTH x and y dimensions
    - For rectangles centered at (x1, y1) and (x2, y2) with widths (w1, w2) and heights (h1, h2):
      * x-overlap occurs when |x1 - x2| < (w1 + w2) / 2
      * y-overlap occurs when |y1 - y2| < (h1 + h2) / 2
    - Use torch.relu() to compute positive overlaps: overlap_x = relu((w1+w2)/2 - |x1-x2|)
    - Overlap area = overlap_x * overlap_y
    - Consider all pairs of cells: use broadcasting with unsqueeze
    - Use torch.triu() to avoid counting each pair twice (only consider i < j)
    - Normalize the loss appropriately (by number of pairs or total area)

    RECOMMENDED APPROACH:
    1. Extract positions, widths, heights from cell_features
    2. Compute all pairwise distances using broadcasting:
       positions_i = positions.unsqueeze(1)  # [N, 1, 2]
       positions_j = positions.unsqueeze(0)  # [1, N, 2]
       distances = positions_i - positions_j  # [N, N, 2]
    3. Calculate minimum separation distances for each pair
    4. Use relu to get positive overlap amounts
    5. Multiply overlaps in x and y to get overlap areas
    6. Mask to only consider upper triangle (i < j)
    7. Sum and normalize

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information (not used here)
        edge_list: [E, 2] tensor with edges (not used here)

    Returns:
        Scalar loss value (should be 0 when no overlaps exist)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True)

    # TODO: Implement overlap detection and loss calculation here
    #
    # Your implementation should:
    # 1. Extract cell positions, widths, and heights
    # 2. Compute pairwise overlaps using vectorized operations
    # 3. Return a scalar loss that is zero when no overlaps exist
    #
    # Delete this placeholder and add your implementation:
    
    import torch.nn.functional as F

    # Cell data
    x = cell_features[:, CellFeatureIdx.X]
    y = cell_features[:, CellFeatureIdx.Y]
    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]
    area = cell_features[:, CellFeatureIdx.AREA]

    xi, yi, wi, hi, areai = x[:, None], y[:, None], w[:, None], h[:, None], area[:, None]
    xj, yj, wj, hj, areaj = x[None, :], y[None, :], w[None, :], h[None, :], area[None, :]

    dx = torch.abs(xi - xj)
    dy = torch.abs(yi - yj)

    # Minimum separations
    min_sep_x = 0.5 * (wi + wj)
    min_sep_y = 0.5 * (hi + hj)

    # Margins scaled by smaller dimension
    margin_scale = 0.1
    margin = margin_scale * torch.minimum(torch.minimum(wi, wj), torch.minimum(hi, hj))

    # Using softplus for better convergence near boundaries
    beta = 10.0
    ox = F.softplus(min_sep_x + margin - dx, beta=beta)
    oy = F.softplus(min_sep_y + margin - dy, beta=beta)

    overlap_area = ox * oy

    # Weighing by area
    pair_weight = (areai * areaj) ** 0.25
    overlap_pen = overlap_area * pair_weight

    # Only sum upper triangle portion
    loss = torch.triu(overlap_pen, diagonal=1).sum() / (N * (N - 1) // 2) # N is at least 2

    return loss

def density_loss_bins(cell_features, pin_features, edge_list, bins=10, target_density=0.6):
    """Scuffed bin-based density loss for now as I figure out how to implement it better
    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information (not used here)
        edge_list: [E, 2] tensor with edges (not used here)
        bins: Number of bins to group cells into
        target_density: The target density of each bin
    Returns:
        Scalar loss value
    """
    device = cell_features.device
    
    # Cell data
    x = cell_features[:, CellFeatureIdx.X]
    y = cell_features[:, CellFeatureIdx.Y]
    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]
    area = cell_features[:, CellFeatureIdx.AREA]
    
    # Get bounds with padding
    pad = 2.0
    xmin = x.min() - pad
    xmax = x.max() + pad
    ymin = y.min() - pad
    ymax = y.max() + pad
    
    # Bin sizes
    bin_w = (xmax - xmin) / bins
    bin_h = (ymax - ymin) / bins
    bin_area = bin_w * bin_h
    
    # Assign each cell to a bin, currently using center point
    bin_x = torch.clamp(((x - xmin) / bin_w).long(), 0, bins - 1)
    bin_y = torch.clamp(((y - ymin) / bin_h).long(), 0, bins - 1)
    bin_idx = bin_y * bins + bin_x  # Flatten to 1D
    
    # Accumulate area in each bin
    density = torch.zeros(bins * bins, device=device)
    density.scatter_add_(0, bin_idx, area)
    
    # Target capacity per bin
    capacity = target_density * bin_area
    
    # Penalize overflow
    overflow = torch.relu(density - capacity)
    
    # Mean square overflow
    return overflow.pow(2).sum() / (bins * bins)


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=None,
    lr=0.75,
    lambda_wirelength=0.0,
    lambda_overlap=1.0,
    lambda_density=3.0,
    verbose=True,
    log_interval=100,
):
    """Train the placement optimization using gradient descent.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Number of optimization iterations, will scale with input size if None
        lr: Learning rate for Adam optimizer
        lambda_wirelength: Weight for wirelength loss
        lambda_overlap: Weight for overlap loss
        lambda_density: Weight for density loss
        verbose: Whether to print progress
        log_interval: How often to print progress

    Returns:
        Dictionary with:
            - final_cell_features: Optimized cell positions
            - initial_cell_features: Original cell positions (for comparison)
            - loss_history: Loss values over time
    """
    # Scale epochs based on input size if not specified
    if num_epochs is None:
        N = cell_features.shape[0]
        if N <= 50:
            num_epochs = 20
            useDensity = False
        elif N <= 200:
            num_epochs = 50
            useDensity = False
        else:
            num_epochs = int(50 + (N - 200) * 0.6)
            useDensity = True

    # Conditionally define loss function
    if useDensity:
        def loss_fn(cell_features_current):
            wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
            )
            overlap_loss = overlap_repulsion_loss(
                cell_features_current, pin_features, edge_list
            )
            density_loss = density_loss_bins(
                cell_features_current, pin_features, edge_list
            )
            loss_history["wirelength_loss"].append(wl_loss.item())
            loss_history["overlap_loss"].append(overlap_loss.item())
            loss_history["density_loss"].append(overlap_loss.item())
            return lambda_wirelength * wl_loss + lambda_overlap * overlap_loss + lambda_density * density_loss 
    else:
        def loss_fn(cell_features_current):
            wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
            )
            overlap_loss = overlap_repulsion_loss(
                cell_features_current, pin_features, edge_list
            )
            loss_history["wirelength_loss"].append(wl_loss.item())
            loss_history["overlap_loss"].append(overlap_loss.item())
            loss_history["density_loss"].append(None)
            return lambda_wirelength * wl_loss + lambda_overlap * overlap_loss
    
    # Clone features and create learnable positions
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()
    
    cell_features = shelf_packing_placement(cell_features)

    # Make only cell positions require gradients
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    # Create optimizer
    optimizer = optim.Adam([cell_positions], lr=lr)

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
        "density_loss": [],
    }

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Create cell_features with current positions
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        # Combined loss
        total_loss = loss_fn(cell_features_current)

        # Backward pass
        total_loss.backward()

        # Gradient clipping to prevent extreme updates
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)

        # Update positions
        optimizer.step()

        # Record losses
        loss_history["total_loss"].append(total_loss.item())

        # Log progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Wirelength Loss: {loss_history["wirelength_loss"][-1]:.6f}")
            print(f"  Overlap Loss: {loss_history["overlap_loss"][-1].item():.6f}")

    # Create final cell features
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions.detach()

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }


# ======= FINAL EVALUATION CODE (Don't edit this part) =======

def calculate_overlap_metrics(cell_features):
    """Calculate ground truth overlap statistics (non-differentiable).

    This function provides exact overlap measurements for evaluation and reporting.
    Unlike the loss function, this does NOT need to be differentiable.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]

    Returns:
        Dictionary with:
            - overlap_count: number of overlapping cell pairs (int)
            - total_overlap_area: sum of all overlap areas (float)
            - max_overlap_area: largest single overlap area (float)
            - overlap_percentage: percentage of total area that overlaps (float)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "overlap_percentage": 0.0,
        }

    # Extract cell properties
    positions = cell_features[:, 2:4].detach().numpy()  # [N, 2]
    widths = cell_features[:, 4].detach().numpy()  # [N]
    heights = cell_features[:, 5].detach().numpy()  # [N]
    areas = cell_features[:, 0].detach().numpy()  # [N]

    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    overlap_areas = []

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2

            # Calculate overlap amounts
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)

            # Overlap occurs only if both x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                overlap_count += 1
                total_overlap_area += overlap_area
                max_overlap_area = max(max_overlap_area, overlap_area)
                overlap_areas.append(overlap_area)

    # Calculate percentage of total area
    total_area = sum(areas)
    overlap_percentage = (overlap_count / N * 100) if total_area > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "overlap_percentage": overlap_percentage,
    }


def calculate_cells_with_overlaps(cell_features):
    """Calculate number of cells involved in at least one overlap.

    This metric matches the test suite evaluation criteria.

    Args:
        cell_features: [N, 6] tensor with cell properties

    Returns:
        Set of cell indices that have overlaps with other cells
    """
    N = cell_features.shape[0]
    if N <= 1:
        return set()

    # Extract cell properties
    positions = cell_features[:, 2:4].detach().numpy()
    widths = cell_features[:, 4].detach().numpy()
    heights = cell_features[:, 5].detach().numpy()

    cells_with_overlaps = set()

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2

            # Calculate overlap amounts
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)

            # Overlap occurs only if both x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                cells_with_overlaps.add(i)
                cells_with_overlaps.add(j)

    return cells_with_overlaps


def calculate_normalized_metrics(cell_features, pin_features, edge_list):
    """Calculate normalized overlap and wirelength metrics for test suite.

    These metrics match the evaluation criteria in the test suite.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity

    Returns:
        Dictionary with:
            - overlap_ratio: (num cells with overlaps / total cells)
            - normalized_wl: (wirelength / num nets) / sqrt(total area)
            - num_cells_with_overlaps: number of unique cells involved in overlaps
            - total_cells: total number of cells
            - num_nets: number of nets (edges)
    """
    N = cell_features.shape[0]

    # Calculate overlap metric: num cells with overlaps / total cells
    cells_with_overlaps = calculate_cells_with_overlaps(cell_features)
    num_cells_with_overlaps = len(cells_with_overlaps)
    overlap_ratio = num_cells_with_overlaps / N if N > 0 else 0.0

    # Calculate wirelength metric: (wirelength / num nets) / sqrt(total area)
    if edge_list.shape[0] == 0:
        normalized_wl = 0.0
        num_nets = 0
    else:
        # Calculate total wirelength using the loss function (unnormalized)
        wl_loss = wirelength_attraction_loss(cell_features, pin_features, edge_list)
        total_wirelength = wl_loss.item() * edge_list.shape[0]  # Undo normalization

        # Calculate total area
        total_area = cell_features[:, 0].sum().item()

        num_nets = edge_list.shape[0]

        # Normalize: (wirelength / net) / sqrt(area)
        # This gives a dimensionless quality metric independent of design size
        normalized_wl = (total_wirelength / num_nets) / (total_area ** 0.5) if total_area > 0 else 0.0

    return {
        "overlap_ratio": overlap_ratio,
        "normalized_wl": normalized_wl,
        "num_cells_with_overlaps": num_cells_with_overlaps,
        "total_cells": N,
        "num_nets": num_nets,
    }


def plot_placement(
    initial_cell_features,
    final_cell_features,
    pin_features,
    edge_list,
    filename="placement_result.png",
):
    """Create side-by-side visualization of initial vs final placement.

    Args:
        initial_cell_features: Initial cell positions and properties
        final_cell_features: Optimized cell positions and properties
        pin_features: Pin information
        edge_list: Edge connectivity
        filename: Output filename for the plot
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot both initial and final placements
        for ax, cell_features, title in [
            (ax1, initial_cell_features, "Initial Placement"),
            (ax2, final_cell_features, "Final Placement"),
        ]:
            N = cell_features.shape[0]
            positions = cell_features[:, 2:4].detach().numpy()
            widths = cell_features[:, 4].detach().numpy()
            heights = cell_features[:, 5].detach().numpy()

            # Draw cells
            for i in range(N):
                x = positions[i, 0] - widths[i] / 2
                y = positions[i, 1] - heights[i] / 2
                rect = Rectangle(
                    (x, y),
                    widths[i],
                    heights[i],
                    fill=True,
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.add_patch(rect)

            # Calculate and display overlap metrics
            metrics = calculate_overlap_metrics(cell_features)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{title}\n"
                f"Overlaps: {metrics['overlap_count']}, "
                f"Total Overlap Area: {metrics['total_overlap_area']:.2f}",
                fontsize=12,
            )

            # Set axis limits with margin
            all_x = positions[:, 0]
            all_y = positions[:, 1]
            margin = 10
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError as e:
        print(f"Could not create visualization: {e}")
        print("Install matplotlib to enable visualization: pip install matplotlib")

# ======= MAIN FUNCTION =======

def main():
    """Main function demonstrating the placement optimization challenge."""
    print("=" * 70)
    print("VLSI CELL PLACEMENT OPTIMIZATION CHALLENGE")
    print("=" * 70)
    print("\nObjective: Implement overlap_repulsion_loss() to eliminate cell overlaps")
    print("while minimizing wirelength.\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate placement problem
    num_macros = 3
    num_std_cells = 50

    print(f"Generating placement problem:")
    print(f"  - {num_macros} macros")
    print(f"  - {num_std_cells} standard cells")

    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    # Initialize positions with random spread to reduce initial overlaps
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Calculate initial metrics
    print("\n" + "=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    initial_metrics = calculate_overlap_metrics(cell_features)
    print(f"Overlap count: {initial_metrics['overlap_count']}")
    print(f"Total overlap area: {initial_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {initial_metrics['max_overlap_area']:.2f}")
    print(f"Overlap percentage: {initial_metrics['overlap_percentage']:.2f}%")

    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION")
    print("=" * 70)

    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=True,
        log_interval=200,
    )

    # Calculate final metrics (both detailed and normalized)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_cell_features = result["final_cell_features"]

    # Detailed metrics
    final_metrics = calculate_overlap_metrics(final_cell_features)
    print(f"Overlap count (pairs): {final_metrics['overlap_count']}")
    print(f"Total overlap area: {final_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {final_metrics['max_overlap_area']:.2f}")

    # Normalized metrics (matching test suite)
    print("\n" + "-" * 70)
    print("TEST SUITE METRICS (for leaderboard)")
    print("-" * 70)
    normalized_metrics = calculate_normalized_metrics(
        final_cell_features, pin_features, edge_list
    )
    print(f"Overlap Ratio: {normalized_metrics['overlap_ratio']:.4f} "
          f"({normalized_metrics['num_cells_with_overlaps']}/{normalized_metrics['total_cells']} cells)")
    print(f"Normalized Wirelength: {normalized_metrics['normalized_wl']:.4f}")

    # Success check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    if normalized_metrics["num_cells_with_overlaps"] == 0:
        print("✓ PASS: No overlapping cells!")
        print("✓ PASS: Overlap ratio is 0.0")
        print("\nCongratulations! Your implementation successfully eliminated all overlaps.")
        print(f"Your normalized wirelength: {normalized_metrics['normalized_wl']:.4f}")
    else:
        print("✗ FAIL: Overlaps still exist")
        print(f"  Need to eliminate overlaps in {normalized_metrics['num_cells_with_overlaps']} cells")
        print("\nSuggestions:")
        print("  1. Check your overlap_repulsion_loss() implementation")
        print("  2. Change lambdas (try increasing lambda_overlap)")
        print("  3. Change learning rate or number of epochs")

    # Generate visualization
    plot_placement(
        result["initial_cell_features"],
        result["final_cell_features"],
        pin_features,
        edge_list,
        filename="placement_result.png",
    )

if __name__ == "__main__":
    main()
