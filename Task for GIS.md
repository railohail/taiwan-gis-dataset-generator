# Taiwan County Detection with Hierarchical DETR

## Goal
Improve instance segmentation accuracy for Taiwan administrative regions using hierarchical detection.

---

## Phase 1: County-Level Detection (Completed)

| Model | Accuracy (mAP) | Notes |
|-------|----------------|-------|
| Baseline | 30.5% | RF-DETR default |
| + Noise Aug | **39.0%** | Best result (+27.7%) |
| DINOv3 | 31.6% | No pretrain |

**Conclusion:** Noise augmentation is the most effective improvement so far.

---

## Phase 2: Hierarchical Detection (In Progress)

### Approach
Extend from county-only (L0) to dual-level hierarchical annotations:
- **L0 (County 縣市):** 22 classes
- **L1 (Township 鄉鎮區):** 368 classes

Parent-child relationships - Townships link to parent counties via `COUNTYID`

---

## Data Structure Overview

### 1. Source Data (Input)

**Shapefiles (Vector Boundaries)**
```
datasets/shapefile/
├── county/
│   └── COUNTY_MOI_1130718.shp     # 22 counties
│       Columns: COUNTYID, COUNTYNAME, COUNTYCODE, geometry
│       Example: COUNTYID="K", COUNTYNAME="苗栗縣"
│
└── township/
    └── TOWN_MOI_1140318.shp       # 368 townships
        Columns: TOWNID, TOWNNAME, TOWNCODE, COUNTYID, geometry
        Example: TOWNID="K01", TOWNNAME="苗栗市", COUNTYID="K"
```

**Raster Maps (TIF Images)**
```
datasets/MAPDATA/
├── miaoli/
│   ├── 10005_001_1_modified.tif
│   └── ...
├── hsinchu/
├── taichung/
└── ... (22 district folders)
```

---

### 2. Hierarchy Mapping

```
datasets/taiwan_admin_hierarchy.json
```

```json
{
  "metadata": {
    "num_counties": 22,
    "num_townships": 368
  },

  "counties": {
    "K": {
      "class_id": 12,
      "name": "苗栗縣",
      "english": "Miaoli County",
      "code": "10005",
      "type": "縣"
    }
  },

  "townships": {
    "K01": {
      "class_id": 45,
      "name": "苗栗市",
      "english": "Miaoli City",
      "parent_county_id": "K",
      "parent_class_id": 12
    }
  }
}
```

**ID Naming Convention**

| Level | ID Format | Example | Description |
|-------|-----------|---------|-------------|
| County | Single letter | `K` | 22 unique letters |
| Township | Letter + 2 digits | `K01` | First char = parent county |

---

### 3. Output Structure (YOLO Hierarchical)

```
yolo_hierarchical/
├── classes_l0.txt              # 22 county class names
├── classes_l1.txt              # 368 township class names
├── hierarchy.json              # Copy of admin hierarchy
├── dataset.yaml                # Training config
│
├── train/
│   ├── images/
│   │   └── miaoli_001_clean.png
│   └── labels/
│       ├── miaoli_001_clean_l0.txt    # County annotations
│       └── miaoli_001_clean_l1.txt    # Township annotations
│
└── valid/
    └── ...
```

**Annotation File Format (YOLO Segmentation)**
```
# L0 file (county): miaoli_001_clean_l0.txt
12 0.234 0.156 0.267 0.189 0.301 0.223 ...
   ↑  └─────────── normalized polygon coords ──────────┘
   class_id (county)

# L1 file (township): miaoli_001_clean_l1.txt
45 0.234 0.156 0.245 0.167 ...    # Miaoli City
9  0.301 0.223 0.312 0.234 ...    # Sanwan Township
11 0.267 0.189 0.278 0.200 ...    # Nanzhuang Township
```

---

### 4. Class Hierarchy Relationship

```
L0 (County)                    L1 (Townships)
─────────────                  ───────────────
class_id: 12                   class_id: 45  (K01 - 苗栗市)
name: 苗栗縣         ───────►  class_id: 9   (K13 - 三灣鄉)
COUNTYID: K                    class_id: 11  (K14 - 南庄鄉)
                               ... (18 townships total)
```

**Key Relationships**

| From | To | Link Field | Example |
|------|-----|------------|---------|
| Township → County | `parent_county_id` | `K01.parent_county_id = "K"` |
| Township → County Class | `parent_class_id` | `K01.parent_class_id = 12` |

---

### 5. Processing Pipeline Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   TIF Raster    │     │  County Shapefile │     │Township Shapefile│
│  (map images)   │     │   (22 polygons)   │     │ (368 polygons)  │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌────────────────────────────────────────────────────────────────────┐
│                    HierarchyManager                                │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  county_level: HierarchyLevel                               │  │
│  │    - gdf: GeoDataFrame (22 rows)                           │  │
│  │    - id_to_class: {"K": 12, "A": 7, ...}                   │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │  township_level: HierarchyLevel                             │  │
│  │    - gdf: GeoDataFrame (368 rows)                          │  │
│  │    - id_to_class: {"K01": 45, "K13": 9, ...}               │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │  township_to_parent: {"K01": "K", "K13": "K", ...}         │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Per-Image Output                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │  image.png   │  │  image_l0.txt    │  │  image_l1.txt        │ │
│  │              │  │  (1 county poly) │  │  (N township polys)  │ │
│  └──────────────┘  └──────────────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

### 6. Township Dropout Augmentation

Simulates partial/incomplete maps by removing edge townships:

```
Original County (L0)          After Dropout
┌─────────────────┐           ┌─────────────────┐
│ ┌───┬───┬───┐   │           │ ┌───┬───┬───┐   │
│ │T01│T02│T03│   │           │ │   │T02│   │   │  ← T01, T03 dropped
│ ├───┼───┼───┤   │    ───►   │ ├───┼───┼───┤   │
│ │T04│T05│T06│   │           │ │T04│T05│T06│   │
│ └───┴───┴───┘   │           │ └───┴───┴───┘   │
└─────────────────┘           └───────┬─────────┘
                                      │
                              County geometry
                              also gets holes!
```

**Config options:**
- `edge_only: true` - Only drop townships on county border
- `cluster_drop: true` - Dropped townships must be adjacent
- `create_county_holes: true` - Subtract dropped townships from L0 geometry

---

## Target Model

**H-DETR** (Hierarchical DETR) - leverages parent-child relationships for improved detection of fine-grained administrative boundaries.

### Expected Benefits
- Better township-level segmentation through county context
- Improved robustness to partial/cropped regions
- Hierarchical loss functions can leverage structure
