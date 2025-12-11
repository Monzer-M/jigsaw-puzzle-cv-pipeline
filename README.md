---

# Gravity Jigsaw CV – Classical Computer Vision Jigsaw Solver

This project solves jigsaw-style puzzles **purely with classical image processing** (no machine learning).

It’s split into two main milestones:

* **MS1 – Preprocessing & Tile Generation**
  Take the scrambled puzzle images (2×2, 4×4, 8×8 grids), enhance them, compute edge maps, and cut them into tiles. Save all intermediate artifacts and a JSON metadata file describing each tile.

* **MS2 – Puzzle Reconstruction**
  Use only the MS1 outputs (tiles + metadata) and the **correct full images** to reconstruct each puzzle.
  Reconstruction works for 2×2, 4×4, and 8×8 puzzles.

The final pipeline uses **no deep learning**. Instead, it relies on:

* Contrast enhancement (CLAHE)
* Noise-preserving smoothing (bilateral filter)
* Unsharp masking for detail enhancement
* Adaptive thresholding & morphology (for masks)
* Canny edges (for strong edge-based descriptors)
* Hybrid border descriptors (color + edges)
* Global matching solved by the Hungarian algorithm (4×4, 8×8)
* A small set of **hand-discovered ID exceptions** for mislabeled ground-truth images

---

## 1. Dataset & Folder Structure

On Google Drive:

```text
/imgdataset
    /correct           # 110 ground-truth full images (0.png ... 109.png)
    /puzzle_2x2        # scrambled 2×2 puzzle images
    /puzzle_4x4        # scrambled 4×4 puzzle images
    /puzzle_8x8        # scrambled 8×8 puzzle images

/imgsProcessed         # MS1 output (created by MS1 notebook)
    /puzzle_2x2
        /0
            0_orig.png
            0_enhanced.png
            0_mask.png
            0_edges.png
            tiles_metadata.json
            /tiles
                0_r0_c0.png
                0_r0_c0_enh.png
                0_r0_c0_edge.png
                ...
    /puzzle_4x4
    /puzzle_8x8

/MS2_Solved            # MS2 output (created by FinalMS_image notebook)
    /puzzle_2x2
        /0
            scrambled.png   # reconstructed scramble from metadata
            solved.png      # best reconstruction (matching or metadata)
    /puzzle_4x4
    /puzzle_8x8
```

---

## 2. MS1 – Preprocessing & Tile Extraction

**Notebook:** `MS1.ipynb`

### 2.1 Goals

1. Clean up each scrambled puzzle image.
2. Generate several useful representations:

   * **Original** image
   * **Enhanced grayscale** (for strong edges & details)
   * **Binary mask** (from adaptive thresholding)
   * **Edge map** (Canny)
3. Cut the original into uniform tiles (2×2, 4×4, or 8×8) using the known grid.
4. Save per-tile variants:

   * color tile
   * enhanced tile
   * edge tile
5. Save a **`tiles_metadata.json`** file so MS2 knows exactly where each tile came from.

### 2.2 Helper Functions & Configuration

* `ensure(path)`
  Creates directories if needed – avoids “folder not found” errors when writing output.

* `list_images(folder)`
  Returns only files with image extensions (`.png`, `.jpg`, …). This ensures we never try to process random system files.

* `CFG` (configuration dictionary)
  All processing hyperparameters are centralized here:

  * `clahe_clip`, `clahe_tile` – CLAHE contrast enhancement
  * `bilateral_d`, `bilateral_sigma_color`, `bilateral_sigma_space` – bilateral filter
  * `unsharp_amount` – strength of unsharp masking
  * `th_block`, `th_C` – adaptive threshold parameters
  * `morph_kernel`, `morph_open_it`, `morph_close_it` – morphology settings
  * `canny_low`, `canny_high` – Canny thresholds

This design makes the pipeline easy to tune without hunting for magic numbers spread across the code.

### 2.3 Preprocessing Pipeline (per image)

For each scrambled puzzle:

1. **Read image (`img_bgr`)**
   Input is BGR (OpenCV).

2. **CLAHE grayscale**

   ```python
   clahe = clahe_grayscale(img_bgr, clipLimit=CFG['clahe_clip'],
                           tileGridSize=CFG['clahe_tile'])
   ```

   * Increases local contrast, especially in darker areas.
   * Helps reveal edges in regions that were too flat or low contrast.

3. **Bilateral filter + Unsharp mask**

   ```python
   unsharp = bilateral_and_unsharp(
       clahe,
       bilateral_d=CFG['bilateral_d'],
       bilateral_sigma_color=CFG['bilateral_sigma_color'],
       bilateral_sigma_space=CFG['bilateral_sigma_space'],
       unsharp_amount=CFG['unsharp_amount']
   )
   ```

   * Bilateral filter smooths noise while preserving edges.
   * Unsharp masking boosts mid-frequency details.
   * Result: clean yet sharp grayscale image, perfect for Canny and for visual matching.

4. **Adaptive Threshold + Morphology (Mask)**

   ```python
   th = adaptive_thresh(unsharp, blockSize=CFG['th_block'], C=CFG['th_C'])
   clean_mask = morphological_cleanup(
       th, kernel_size=CFG['morph_kernel'],
       open_iterations=CFG['morph_open_it'],
       close_iterations=CFG['morph_close_it']
   )
   ```

   * Adaptive thresholding computes a local threshold, handling changing illumination.
   * Opening removes tiny noise blobs; closing fills small holes.
   * The resulting mask is mainly diagnostic / visualization in MS1 (not used heavily in MS2) but can be extended for advanced segmentation.

5. **Canny Edge Detection**

   ```python
   edges = compute_canny(unsharp, low=CFG['canny_low'], high=CFG['canny_high'])
   ```

   * Produces a binary edge map emphasizing strong structural boundaries.
   * Later, in MS2, we use Canny-based descriptors along tile borders, because they’re robust to small color changes and focus on geometric structure.

6. **Grid-Based Tiling**

   We already know the grid size from the folder (`puzzle_2x2`, `puzzle_4x4`, `puzzle_8x8`).
   So instead of complicated contour detection, we use a simple **grid split**:

   ```python
   tiles_rgb   = split_into_grid(to_rgb(img_bgr), rows, cols)
   tiles_enh   = split_into_grid(unsharp, rows, cols)
   tiles_edges = split_into_grid(edges, rows, cols)
   ```

   where each item is `((row, col), (x1,y1,x2,y2), tile_image)`.

   This is far more robust than contour-based tiling and guaranteed to give us the correct tile count.

### 2.4 Per-Image Output (MS1)

In `imgsProcessed/puzzle_#x#/ID/`:

* `ID_orig.png` – original scrambled image
* `ID_enhanced.png` – unsharp grayscale
* `ID_mask.png` – threshold + morphology result
* `ID_edges.png` – Canny edges

In `imgsProcessed/puzzle_#x#/ID/tiles/`:

* `ID_r{r}_c{c}.png` – color tile
* `ID_r{r}_c{c}_enh.png` – enhanced tile
* `ID_r{r}_c{c}_edge.png` – edges tile

And a **metadata file**:

```json
[
  {
    "tile_name": "0_r0_c0.png",
    "row": 0,
    "col": 0,
    "bbox": [0, 0, 112, 112]
  },
  //...
]
```

This metadata is *critical* for MS2: it tells us exactly where every tile belongs in a correctly ordered version of the scrambled image.

---

## 3. MS2 – Puzzle Reconstruction

**Notebook:** `FinalMS_image.ipynb`

### 3.1 High-Level Idea

MS2 uses *only* MS1 outputs plus the **correct full images** from `/correct`.

For each puzzle:

1. Reconstruct the scrambled image from tiles using metadata.
   (This is solver B / fallback and also gives us tile geometry.)
2. Use the matching solver (solver A) to assign tiles to positions that best match the **correct image**.
3. Compare the solved result to the clean reference:

   * If the match is good (high confidence), accept solver A.
   * Otherwise, fall back to the metadata layout (which is at least a coherent image).
4. Save both `scrambled.png` and `solved.png` in a structured output folder.

### 3.2 Handling Misaligned IDs (Exceptions Table)

We discovered a **dataset issue**: for some indices, the scrambled puzzle ID did **not** match the correct image’s ID. For example:

* Scrambled 3×3 puzzle folder `3` actually belongs to `correct/4.png`, and vice versa.

To handle this, we created an **EXCEPTIONS** dictionary:

```python
EXCEPTIONS = {
    3: 4, 4: 3, 5: 6, 6: 5,
    31: 32, 32: 31,
    ...
    97: 98, 98: 97
}
```

When solving puzzle `pid`, we get its true clean reference via:

```python
true_id = EXCEPTIONS.get(pid, pid)
clean = cv2.imread(str(CORRECT_DIR / f"{true_id}.png"))
```

This ensures we always compare against the correct ground-truth image, even when the filenames in the original dataset don’t match.

### 3.3 Common helper functions (MS2)

* `load_tiles_exact(folder)`
  Reads `tiles_metadata.json` and loads *only* the tiles listed there.
  If any tile is missing or unreadable, the puzzle is **skipped** (for 8×8 we also skip puzzle 9 and 88 where the data is incomplete).

* `reconstruct_from_metadata(tiles, metadata)`
  Uses `row` and `col` from metadata to rebuild the scrambled image in its correct grid layout (works for all grid sizes N×N).

* `slice_clean_image(clean_img, N, th, tw)`
  Resize clean image to `N*tw × N*th` and slice it into `N×N` reference tiles.

* `rgb_border(tile, border)` / `edge_border(tile, border)`
  Extract border strips in RGB and edge-space.

* `hybrid_distance(a, b)`
  Our key similarity metric: a weighted combination of L1 distances between border descriptors at different band widths:

  ```python
  d1 = |RGB border 8px|
  d2 = |RGB border 16px|
  d3 = |Edge border 12px|
  d4 = |Edge border 20px|
  distance = 0.2*d1 + 0.3*d2 + 0.2*d3 + 0.3*d4
  ```

  This proved more robust than plain pixel MSE because:

  * Borders matter most for tile adjacency.
  * Combining color and edges balances texture vs. structure.
  * Multi-scale borders capture context while still focusing on tile edges.

* `hungarian_assignment(cost_matrix, tile_names, N)`
  Uses **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) to compute the globally optimal assignment of tiles to positions for 4×4 and 8×8 puzzles.

* `save_solver_output(puzzle_type, pid, scrambled_img, solved_img)`
  Writes `scrambled.png` and `solved.png` to `/MS2_Solved/puzzle_#x#/<pid>/`.

### 3.4 Solver A – Matching-Based Reconstruction

#### 2×2 solver

For each puzzle ID `pid`:

1. **Load tiles + metadata**, rebuild scrambled image (2×2).

2. Infer tile height/width from one tile.

3. Load the correct image (`correct[ true_id ]`), slice it into 4 reference tiles.

4. For every tile, compute its cost against each of the four positions using `hybrid_distance`.

5. Try all 4! = 24 permutations of positions to find the **minimal total cost**.

6. Build `solverA` image by placing each tile in its best position.

7. Compute confidence vs clean:

   ```python
   diff = mean(|solverA - clean_resized|)
   confidence = 1 - diff / 255
   ```

8. If `confidence >= 0.85`, accept solver A; otherwise fall back to metadata reconstruction.

#### 4×4 solver

Same structure, but:

* 16 tiles → 16 positions

* Build a **16×16 cost matrix**:

  ```python
  cost_matrix[i,j] = hybrid_distance(tile_i, ref_tile_j)
  ```

* Use **Hungarian algorithm** to find the optimal assignment (no need to brute-force 16! permutations).

* Rebuild `solverA` with the assigned positions.

* Confidence threshold is slightly stricter (`0.87`) because 4×4 is larger and incorrect placements can still look “okay”.

#### 8×8 solver

* 64 tiles → 64 positions
* Any brute-force search is impossible, so Hungarian matching is essential.
* We **skip** puzzles with incomplete tiles (9, 88) or missing metadata.
* Confidence threshold is even stricter (`0.90`) to avoid subtle but widespread misplacements.

### 3.5 Solver B – Metadata Fallback

Solver B simply uses the **metadata** layout:

* Reconstruct the scrambled image using each tile’s `(row, col)` position.
* No matching to the correct image.
* It’s a guaranteed coherent reconstruction (no overlapping, no empty slots) but not necessarily the solved puzzle; we use it only:

  * When the clean image is missing, or
  * When Solver A’s confidence is too low.

### 3.6 Running All Solvers

Each solver has a loop like:

```python
folders = sorted([f for f in PUZZLE_DIR_2x2.iterdir() if f.is_dir()],
                 key=lambda x: int(x.name))

for folder in folders:
    pid = int(folder.name)
    solved, method = solve_puzzle_2x2(pid)
    ...
```

The same pattern is used for 4×4 and 8×8, with statistics printed at the end:

* How many puzzles were solved by matching (A)
* How many fell back to metadata (B)
---

## 4. Design Decisions & Justification

### Why classical CV instead of deep learning?

* The dataset is relatively small (110 images).
* Tiles are **perfect grid splits**, not arbitrary irregular shapes.
* We know the exact size and number of pieces; the structure is constrained.
* Classical techniques (filters, edges, Hungarian matching) are transparent, explainable, and easier to debug for a small academic project.

### Why grid-based tiling instead of contour extraction?

We initially considered contour-based extraction of puzzle pieces. However:

* Tiles are perfect rectangles aligned with the image borders.
* Contour-based segmentation is overkill and error-prone under noise, especially near transparent or low-contrast boundaries.
* Grid-based splitting is deterministic and 100% aligned with the puzzle generation process.

So we removed contour-based segmentation and relied entirely on `split_into_grid`.

### Why CLAHE + bilateral + unsharp?

* CLAHE: fixes local contrast issues and prevents over-saturation (compared to global histogram equalization).
* Bilateral filter: denoises while preserving edges, so Canny and border comparisons are sharper.
* Unsharp masking: boosts fine details along edges without totally blowing up noise.

This combination gives strong structures for both raw RGB and edge-based descriptors.

### Why Canny edges for descriptors?

Earlier we tried **Sobel** and simple grayscale differences, but:

* Sobel responses were noisy and less stable across tiles with fine textures.
* Edges from Canny focus on more global, perceptually important contours.
* Edge-based borders are insensitive to changes in absolute color but sensitive to object boundaries, which is exactly what we want for matching adjacent tiles.

### Why border descriptors, not full tile descriptors?

We tried full-tile comparisons (MSE between entire tiles). This often failed because:

* Many regions of the images (e.g., blue sky, wooden backgrounds) look very similar away from the borders.
* Jigsaw solving depends mainly on **how neighboring tiles meet at the borders**, not on the global texture inside the tile.

So we switched to **border-only descriptors**, and further improved them by:

* Using **multiple border thicknesses** (8, 16, 12, 20 pixels).
* Combining **color** and **edges**.

This made neighboring tiles more discriminative.

### Why Hungarian algorithm?

Initial attempts for 4×4 and 8×8:

* Direct greedily picking the “best match” for each tile/edge created inconsistent layouts (collisions and cycles).
* Trying to build local adjacency graphs (tile A’s right edge best matches tile B’s left edge, etc.) produced valid pairwise matches but not consistent global grids.

The Hungarian algorithm solves this elegantly:

* Construct a cost matrix tile → position.
* Find the assignment that minimizes **total cost**, not local pairwise cost.
* Guarantees each tile is used exactly once and each position has exactly one tile.

This is perfect for fixed-grid puzzles.

### Why the confidence threshold & metadata fallback?

Even with a good cost function, some images are tricky:

* Large uniform areas (sky, walls, ground) where many tiles look similar.
* Subtle color variations that trick the border metric.

If we blindly trust the matching, some puzzles can be severely mis-solved.

So we:

1. Rebuild an image from the matching (`solverA`).
2. Compare it to the resized clean image via mean absolute pixel difference.
3. Accept only if confidence is high (≥ 0.85 / 0.87 / 0.90 depending on grid size).
4. Otherwise, fall back to the safe metadata reconstruction.

This protects the final output quality.

### Why the EXCEPTIONS mapping?

We discovered that for certain IDs, the scrambled puzzle and the correct image **did not correspond by filename** (e.g., puzzle `3` corresponds to correct image `4`). Without fixing this, the solver’s “accuracy” appears bad even when tile matching is correct relative to *its own* reference.

Instead of renaming all the dataset images, we added an explicit exceptions map:

```python
EXCEPTIONS = {3:4, 4:3, 5:6, 6:5, ...}
```

This keeps the raw dataset unchanged and centralizes the fix in the code.

---

## 5. Methods Tried & Why They Failed

During development we tried several approaches that **did not** work well enough. They’re worth describing because they motivated the final design.

### 5.1 Pairwise edge matching between tiles (no ground-truth image)

Idea:

* Ignore the correct images.
* For each tile edge (top/right/bottom/left), find its best matching complementary edge in another tile (e.g., right edge of A with left edge of B).
* Build a huge adjacency graph and try to place tiles based on local neighbor relationships only.

Why it failed:

* Many borders in these cartoon images are visually similar — e.g., large wood or grass regions.
* Local best matches are often **ambiguous or misleading**.
* You can get loops and contradictions (A’s right matches B’s left, B’s right also wants C’s left, etc.).
* 8×8 becomes combinatorially explosive without strong global constraints.

Verdict: cool idea, but too weak without a global reference image.

### 5.2 Using only edge tiles (`*_edge.png`) as input for matching

We generated Sobel/Canny edge tiles in MS1 and tried to:

* Match tiles using only their edge images.
* Even tried edge-only adjacency and full-tile edge matching.

Issues:

* Edges alone are binary and lose important color context.
* Many tiles (same type of trees, sky, etc.) have very similar edge structures.
* Edge-only matching produced incorrect assignments and low reconstruction quality.

We finally decided to **combine** color + edges instead of using edges alone.

### 5.3 Direct full-tile MSE to reference tiles

We tried:

```python
distance = mean(|tile - ref_tile|)
```

* Works for some easy images but breaks badly when two tiles have similar interior textures but different borders.
* Jigsaw solving cares about how tiles meet at boundaries, not interior similarity.
* Leading to swapped tiles with similar inside regions but wrong border alignment.

Hence, we moved to **border-only** hybrid descriptors.

### 5.4 Brute-force or greedy assignment for 4×4 / 8×8

* For 4×4 we initially tried greedy strategies (iteratively placing the best tile-position pair).
* For 8×8 brute-force is obviously impossible (64!).
* Greedy approaches would get stuck early — good initial choices block the optimal global layout.

Switching to **Hungarian algorithm** gave us a principled optimal assignment for any N×N grid.

---

## 6. How to Run

### Requirements

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* Matplotlib
* SciPy (for Hungarian algorithm: `scipy.optimize.linear_sum_assignment`)
* Google Colab (or local env with Google Drive mounted similarly)

### Steps

1. **Run MS1 (`MS1.ipynb`)**

   * Mount Google Drive.
   * Set `DATASET_ROOT` and `OUTPUT_ROOT` correctly.
   * Run all cells.
   * This will populate `/imgsProcessed/puzzle_2x2`, `/puzzle_4x4`, `/puzzle_8x8`.

2. **Run MS2 (`FinalMS_image.ipynb`)**

   * Mount Drive again.
   * Check that `BASE_DIR` and `CORRECT_DIR` paths are correct.
   * Run all helper cells and then:

     * the 2×2 solver-all loop,
     * the 4×4 solver-all loop,
     * the 8×8 solver-all loop.
   * Results land in `/MS2_Solved/puzzle_2x2`, `/puzzle_4x4`, `/puzzle_8x8`.

---

## 7. References / Inspiration

These resources informed the methods and design choices (no direct copying of code, only concepts):

* OpenCV Documentation – Contrast Limited Adaptive Histogram Equalization (CLAHE), bilateral filtering, unsharp masking, Canny edges, etc.
* SciPy Documentation – `scipy.optimize.linear_sum_assignment` for the Hungarian algorithm used in tile–position matching.
* Research on jigsaw puzzle solving (conceptual inspiration for global matching):

  * Gallagher, A. C. “Jigsaw puzzles with pieces of unknown orientation.” CVPR 2012.
  * Paikin, G., & Tal, A. “Solving Jigsaw Puzzles with Loop Constraints.” CVPR 2015.

---
