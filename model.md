# H-DETR: Hierarchical Detection Transformer for Administrative Boundary Segmentation

## Overview

H-DETR extends RF-DETR to perform **hierarchical instance segmentation** on administrative boundaries. Instead of predicting flat classes, the model predicts a nested hierarchy where lower-level regions (townships/鄉鎮) must be spatially contained within higher-level regions (counties/縣市).

### Key Contributions

1. **Hierarchical Query Design** - Separate query sets for each administrative level with cross-level attention
2. **Hierarchy Consistency Loss** - Novel loss function ensuring spatial nesting constraints
3. **Parent-Child Attention Module** - Township queries attend to county features for contextual reasoning
4. **Bottom-Up Evidence Aggregation** - Township predictions reinforce county predictions ("I see 大安區 → This is 台北市")

---

## Architecture Diagram

```
                         Input Map Image
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DINOv2 Backbone                                     │
│                    (Frozen or Fine-tuned, from RF-DETR)                       │
│                                                                               │
│   - DINOv2-Base (86M params) or DINOv2-Small (22M params)                    │
│   - Outputs multi-scale features F = {F1, F2, F3, F4}                        │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Transformer Encoder                                    │
│                                                                               │
│   - 6 encoder layers with deformable attention                               │
│   - Multi-scale feature fusion                                               │
│   - Output: encoder_memory (H×W×C)                                           │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│     County Decoder (L0)       │   │    Township Decoder (L1)      │
│                               │   │                               │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │ query_feat_l0           │  │   │  │ query_feat_l1           │  │
│  │ (N_county × hidden_dim) │  │   │  │ (N_town × hidden_dim)   │  │
│  │ N_county = 50 queries   │  │   │  │ N_town = 300 queries    │  │
│  └─────────────────────────┘  │   │  └───────────┬─────────────┘  │
│              │                │   │              │                │
│              ▼                │   │              ▼                │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │ 6× Decoder Layers       │  │   │  │ Cross-Level Attention   │◄─┼──┐
│  │  - Self-Attention       │  │   │  │ (attend to L0 features) │  │  │
│  │  - Cross-Attention      │  │   │  └───────────┬─────────────┘  │  │
│  │  - FFN                  │  │   │              │                │  │
│  └───────────┬─────────────┘  │   │              ▼                │  │
│              │                │   │  ┌─────────────────────────┐  │  │
│              │                │   │  │ 6× Decoder Layers       │  │  │
│              │                │   │  │  - Self-Attention       │  │  │
│              │                │   │  │  - Cross-Attention      │  │  │
│              │                │   │  │  - FFN                  │  │  │
│              │                │   │  └───────────┬─────────────┘  │  │
└──────────────┼────────────────┘   └──────────────┼────────────────┘  │
               │                                   │                   │
               │              ┌────────────────────┘                   │
               │              │                                        │
               ▼              ▼                                        │
┌──────────────────────────────────────────────────────────────────┐   │
│                        Output Heads                               │   │
│                                                                   │   │
│  County (L0) Heads:                                              │   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │   │
│  │ class_embed_l0  │  │ bbox_embed_l0   │  │ mask_embed_l0   │   │   │
│  │ → 22 classes    │  │ → 4 coords      │  │ → H×W mask      │───┼───┘
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                   │
│  Township (L1) Heads:                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ class_embed_l1  │  │ bbox_embed_l1   │  │ mask_embed_l1   │   │
│  │ → 368 classes   │  │ → 4 coords      │  │ → H×W mask      │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                   │
│  Hierarchy Head:                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ parent_head: Linear(hidden_dim → 22)                        │ │
│  │ Predicts which county each township belongs to              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. Hierarchical Query Embeddings

```python
class HierarchicalQueries(nn.Module):
    """
    Separate learnable queries for each administrative level.
    Township queries are enhanced with county context via cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_county_queries: int = 50,      # More than 22 counties for flexibility
        n_township_queries: int = 300,   # More than 368 townships for flexibility
    ):
        super().__init__()

        # Level 0: County queries
        self.query_feat_l0 = nn.Embedding(n_county_queries, hidden_dim)
        self.refpoint_embed_l0 = nn.Embedding(n_county_queries, 4)  # (cx, cy, w, h)

        # Level 1: Township queries
        self.query_feat_l1 = nn.Embedding(n_township_queries, hidden_dim)
        self.refpoint_embed_l1 = nn.Embedding(n_township_queries, 4)

        # Initialize reference points uniformly across image
        nn.init.uniform_(self.refpoint_embed_l0.weight, 0, 1)
        nn.init.uniform_(self.refpoint_embed_l1.weight, 0, 1)
```

### 2. Cross-Level Attention Module

```python
class CrossLevelAttention(nn.Module):
    """
    Allows township queries to attend to county decoder outputs.
    This enables bottom-up reasoning: township features inform county predictions.

    Key insight: "I see 大安區 (Da'an) → This must be 台北市 (Taipei)"
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Project attention weights to parent prediction
        self.parent_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        township_queries: Tensor,   # (B, N_town, C)
        county_features: Tensor,    # (B, N_county, C)
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            enhanced_queries: Township queries with county context
            attn_weights: (B, N_town, N_county) - which county each township attends to
        """
        attended, attn_weights = self.cross_attn(
            query=township_queries,
            key=county_features,
            value=county_features,
        )

        # Residual connection
        enhanced_queries = self.norm(township_queries + self.dropout(attended))

        return enhanced_queries, attn_weights
```

### 3. Hierarchical LWDETR (Main Model)

```python
class HierarchicalLWDETR(nn.Module):
    """
    Hierarchical Detection Transformer for nested administrative boundaries.

    Extends RF-DETR's LWDETR to predict two levels:
    - Level 0: Counties (縣市) - 22 classes
    - Level 1: Townships (鄉鎮區) - 368 classes

    With explicit hierarchy modeling via cross-level attention and
    consistency constraints.
    """

    def __init__(
        self,
        backbone: nn.Module,                    # DINOv2
        transformer: nn.Module,                  # From RF-DETR
        num_classes_l0: int = 22,               # Taiwan counties
        num_classes_l1: int = 368,              # Taiwan townships
        num_queries_l0: int = 50,
        num_queries_l1: int = 300,
        hidden_dim: int = 256,
        num_feature_levels: int = 4,
        aux_loss: bool = True,
        with_mask: bool = True,                  # Instance segmentation
    ):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss
        self.with_mask = with_mask

        # === Query Embeddings ===
        self.query_feat_l0 = nn.Embedding(num_queries_l0, hidden_dim)
        self.query_feat_l1 = nn.Embedding(num_queries_l1, hidden_dim)
        self.refpoint_embed_l0 = nn.Embedding(num_queries_l0, 4)
        self.refpoint_embed_l1 = nn.Embedding(num_queries_l1, 4)

        # === Cross-Level Attention ===
        self.cross_level_attn = CrossLevelAttention(hidden_dim)

        # === Level 0 (County) Heads ===
        self.class_embed_l0 = nn.Linear(hidden_dim, num_classes_l0)
        self.bbox_embed_l0 = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # === Level 1 (Township) Heads ===
        self.class_embed_l1 = nn.Linear(hidden_dim, num_classes_l1)
        self.bbox_embed_l1 = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # === Hierarchy Head ===
        # Predicts parent county for each township query
        self.parent_head = nn.Linear(hidden_dim, num_classes_l0)

        # === Mask Heads (for segmentation) ===
        if with_mask:
            self.mask_head_l0 = MaskHeadSmallConv(
                hidden_dim, hidden_dim, hidden_dim
            )
            self.mask_head_l1 = MaskHeadSmallConv(
                hidden_dim, hidden_dim, hidden_dim
            )

    def forward(self, samples: NestedTensor) -> Dict[str, Tensor]:
        """
        Args:
            samples: Batched images with masks

        Returns:
            Dictionary containing:
            - pred_logits_l0: (B, N_county, 22) county class logits
            - pred_boxes_l0: (B, N_county, 4) county boxes
            - pred_masks_l0: (B, N_county, H, W) county masks
            - pred_logits_l1: (B, N_town, 368) township class logits
            - pred_boxes_l1: (B, N_town, 4) township boxes
            - pred_masks_l1: (B, N_town, H, W) township masks
            - pred_parent: (B, N_town, 22) parent county prediction
            - cross_level_attn: (B, N_town, N_county) attention weights
        """
        # Extract features
        features, pos_embeds = self.backbone(samples)

        # Encode
        srcs, masks = self._prepare_encoder_inputs(features)
        encoder_output = self.transformer.encoder(srcs, masks, pos_embeds)

        # === Decode Level 0 (Counties) ===
        bs = samples.tensors.shape[0]
        query_l0 = self.query_feat_l0.weight.unsqueeze(0).expand(bs, -1, -1)
        refpoint_l0 = self.refpoint_embed_l0.weight.unsqueeze(0).expand(bs, -1, -1)

        county_features, county_references = self.transformer.decoder(
            query_l0, refpoint_l0, encoder_output, ...
        )

        # === Cross-Level Enhancement ===
        query_l1 = self.query_feat_l1.weight.unsqueeze(0).expand(bs, -1, -1)
        enhanced_query_l1, cross_attn_weights = self.cross_level_attn(
            query_l1, county_features[-1]  # Use final layer county features
        )

        # === Decode Level 1 (Townships) ===
        refpoint_l1 = self.refpoint_embed_l1.weight.unsqueeze(0).expand(bs, -1, -1)

        township_features, township_references = self.transformer.decoder(
            enhanced_query_l1, refpoint_l1, encoder_output, ...
        )

        # === Output Predictions ===
        outputs = {}

        # County outputs
        outputs['pred_logits_l0'] = self.class_embed_l0(county_features[-1])
        outputs['pred_boxes_l0'] = self.bbox_embed_l0(county_features[-1]).sigmoid()

        # Township outputs
        outputs['pred_logits_l1'] = self.class_embed_l1(township_features[-1])
        outputs['pred_boxes_l1'] = self.bbox_embed_l1(township_features[-1]).sigmoid()

        # Hierarchy outputs
        outputs['pred_parent'] = self.parent_head(township_features[-1])
        outputs['cross_level_attn'] = cross_attn_weights

        # Mask outputs
        if self.with_mask:
            outputs['pred_masks_l0'] = self.mask_head_l0(
                county_features[-1], encoder_output, ...
            )
            outputs['pred_masks_l1'] = self.mask_head_l1(
                township_features[-1], encoder_output, ...
            )

        # Auxiliary outputs from intermediate layers
        if self.aux_loss:
            outputs['aux_outputs_l0'] = self._get_aux_outputs(
                county_features[:-1], self.class_embed_l0, self.bbox_embed_l0
            )
            outputs['aux_outputs_l1'] = self._get_aux_outputs(
                township_features[:-1], self.class_embed_l1, self.bbox_embed_l1
            )

        return outputs
```

---

## Loss Functions

### 4. Hierarchical Set Criterion

```python
class HierarchicalSetCriterion(nn.Module):
    """
    Loss computation for hierarchical detection.

    Total Loss = L_county + L_township + λ_hier * L_hierarchy + λ_parent * L_parent

    Where:
    - L_county: Standard DETR loss for county predictions
    - L_township: Standard DETR loss for township predictions
    - L_hierarchy: Spatial consistency loss (townships inside counties)
    - L_parent: Cross-entropy for parent county prediction
    """

    def __init__(
        self,
        num_classes_l0: int = 22,
        num_classes_l1: int = 368,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        losses: List[str] = ['labels', 'boxes', 'masks'],
        hierarchy_weight: float = 2.0,
        parent_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes_l0 = num_classes_l0
        self.num_classes_l1 = num_classes_l1
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.hierarchy_weight = hierarchy_weight
        self.parent_weight = parent_weight

    def loss_hierarchy_consistency(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices_l0: List[Tuple[Tensor, Tensor]],
        indices_l1: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """
        Ensures township masks are spatially contained within parent county masks.

        For each predicted township:
        1. Get its predicted parent county (from parent_head or ground truth)
        2. Get the county's predicted mask
        3. Penalize township mask pixels that fall outside county mask

        Loss = Σ max(0, M_township - M_parent_county)

        This enforces: township ⊆ parent_county spatially
        """
        pred_masks_l0 = outputs['pred_masks_l0']  # (B, N_county, H, W)
        pred_masks_l1 = outputs['pred_masks_l1']  # (B, N_town, H, W)
        pred_parent = outputs['pred_parent']       # (B, N_town, 22)

        total_loss = 0.0
        num_townships = 0

        for batch_idx, (idx_l0, idx_l1) in enumerate(zip(indices_l0, indices_l1)):
            # Get matched predictions
            pred_idx_l0, tgt_idx_l0 = idx_l0
            pred_idx_l1, tgt_idx_l1 = idx_l1

            # Get target parent information
            target_parents = targets[batch_idx]['parent_ids']  # Township → County mapping

            for pred_t_idx, tgt_t_idx in zip(pred_idx_l1, tgt_idx_l1):
                # Get township mask
                township_mask = pred_masks_l1[batch_idx, pred_t_idx].sigmoid()

                # Get ground truth parent county for this township
                parent_county_tgt_idx = target_parents[tgt_t_idx]

                # Find which prediction corresponds to parent county
                parent_in_pred = (tgt_idx_l0 == parent_county_tgt_idx).nonzero()
                if len(parent_in_pred) == 0:
                    continue  # Parent county not in this image's predictions

                parent_pred_idx = pred_idx_l0[parent_in_pred[0]]
                county_mask = pred_masks_l0[batch_idx, parent_pred_idx].sigmoid()

                # Violation: township pixels outside parent county
                # ReLU ensures we only penalize violations, not containment
                violation = torch.relu(township_mask - county_mask)
                total_loss += violation.sum()
                num_townships += 1

        if num_townships == 0:
            return torch.tensor(0.0, device=pred_masks_l0.device)

        return total_loss / num_townships

    def loss_parent_classification(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices_l1: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """
        Cross-entropy loss for predicting which county each township belongs to.

        Ground truth comes from administrative hierarchy:
        e.g., 大安區 → 台北市 (class 0)
              鳳山區 → 高雄市 (class 1)
        """
        pred_parent = outputs['pred_parent']  # (B, N_town, 22)

        total_loss = 0.0
        num_townships = 0

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices_l1):
            if len(pred_idx) == 0:
                continue

            # Get predictions for matched townships
            pred_parent_logits = pred_parent[batch_idx, pred_idx]  # (N_matched, 22)

            # Get ground truth parent county class
            target_parent_classes = targets[batch_idx]['parent_classes'][tgt_idx]

            loss = F.cross_entropy(pred_parent_logits, target_parent_classes)
            total_loss += loss * len(pred_idx)
            num_townships += len(pred_idx)

        if num_townships == 0:
            return torch.tensor(0.0, device=pred_parent.device)

        return total_loss / num_townships

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Compute all losses.
        """
        losses = {}

        # === Level 0 (County) Losses ===
        indices_l0 = self.matcher(outputs, targets, level=0)
        for loss_name in self.losses:
            loss_fn = getattr(self, f'loss_{loss_name}')
            losses.update(loss_fn(outputs, targets, indices_l0, level=0))

        # === Level 1 (Township) Losses ===
        indices_l1 = self.matcher(outputs, targets, level=1)
        for loss_name in self.losses:
            loss_fn = getattr(self, f'loss_{loss_name}')
            l1_losses = loss_fn(outputs, targets, indices_l1, level=1)
            losses.update({f'{k}_l1': v for k, v in l1_losses.items()})

        # === Hierarchy Consistency Loss ===
        losses['loss_hierarchy'] = self.loss_hierarchy_consistency(
            outputs, targets, indices_l0, indices_l1
        ) * self.hierarchy_weight

        # === Parent Classification Loss ===
        losses['loss_parent'] = self.loss_parent_classification(
            outputs, targets, indices_l1
        ) * self.parent_weight

        return losses
```

---

## Training Configuration

### 5. Hyperparameters

```yaml
# h_detr_config.yaml

model:
  name: "H-DETR"
  backbone: "dinov2_base"
  hidden_dim: 256
  num_feature_levels: 4

  # Query configuration
  num_queries_l0: 50      # County queries (> 22 for flexibility)
  num_queries_l1: 300     # Township queries (> 368 for flexibility)

  # Decoder configuration
  num_decoder_layers: 6
  num_heads: 8
  dim_feedforward: 2048
  dropout: 0.1

hierarchy:
  num_classes_l0: 22      # Taiwan counties
  num_classes_l1: 368     # Taiwan townships

  # Parent-child mapping loaded from JSON
  hierarchy_file: "taiwan_admin_hierarchy.json"

loss:
  # Standard DETR losses
  class_weight: 2.0
  bbox_weight: 5.0
  giou_weight: 2.0
  mask_weight: 5.0
  dice_weight: 5.0

  # Hierarchy-specific losses (OUR CONTRIBUTION)
  hierarchy_weight: 2.0   # Spatial containment loss
  parent_weight: 1.0      # Parent classification loss

training:
  # Curriculum learning
  phase1:
    epochs: 20
    description: "County-only training"
    freeze_l1: true

  phase2:
    epochs: 20
    description: "Add township, freeze county"
    freeze_l0: true

  phase3:
    epochs: 30
    description: "Joint training with hierarchy loss"
    freeze_l0: false
    freeze_l1: false

  phase4:
    epochs: 10
    description: "End-to-end fine-tuning"
    lr_multiplier: 0.1

optimizer:
  name: "AdamW"
  lr: 1e-4
  backbone_lr: 1e-5
  weight_decay: 1e-4

scheduler:
  name: "MultiStepLR"
  milestones: [60, 80]
  gamma: 0.1

data:
  train_batch_size: 4
  val_batch_size: 8
  num_workers: 4
  image_size: 640
```

---

## Data Format

### 6. Hierarchical COCO Annotation Format

```json
{
  "info": {
    "description": "Taiwan Administrative Boundaries - Hierarchical",
    "version": "1.0",
    "hierarchy_levels": ["county", "township"]
  },

  "categories_l0": [
    {"id": 0, "name": "台北市", "english": "Taipei City"},
    {"id": 1, "name": "高雄市", "english": "Kaohsiung City"},
    ...
  ],

  "categories_l1": [
    {"id": 0, "name": "中正區", "parent_id": 0, "english": "Zhongzheng"},
    {"id": 1, "name": "大安區", "parent_id": 0, "english": "Da'an"},
    {"id": 2, "name": "鳳山區", "parent_id": 1, "english": "Fengshan"},
    ...
  ],

  "hierarchy": {
    "0": [0, 1, 2, 3, ...],     // 台北市 → [中正區, 大安區, ...]
    "1": [100, 101, 102, ...],  // 高雄市 → [鳳山區, ...]
    ...
  },

  "images": [...],

  "annotations": [
    {
      "id": 1,
      "image_id": 100,
      "category_id": 0,
      "category_level": 0,
      "segmentation": [[...]],
      "bbox": [x, y, w, h],
      "area": 50000,
      "children": [101, 102, 103]
    },
    {
      "id": 101,
      "image_id": 100,
      "category_id": 5,
      "category_level": 1,
      "segmentation": [[...]],
      "bbox": [x, y, w, h],
      "area": 5000,
      "parent_id": 1,
      "parent_class": 0
    }
  ]
}
```

---

## Inference Pipeline

### 7. Hierarchical Post-Processing

```python
class HierarchicalPostProcessor:
    """
    Post-process H-DETR outputs with hierarchy enforcement.
    """

    def __init__(
        self,
        score_threshold_l0: float = 0.5,
        score_threshold_l1: float = 0.3,
        enforce_hierarchy: bool = True,
    ):
        self.score_threshold_l0 = score_threshold_l0
        self.score_threshold_l1 = score_threshold_l1
        self.enforce_hierarchy = enforce_hierarchy

    def __call__(self, outputs: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Process outputs into final predictions with hierarchy enforcement.
        """
        # Filter county predictions
        county_scores = outputs['pred_logits_l0'].softmax(-1)
        county_scores, county_labels = county_scores.max(-1)
        county_keep = county_scores > self.score_threshold_l0

        # Filter township predictions
        township_scores = outputs['pred_logits_l1'].softmax(-1)
        township_scores, township_labels = township_scores.max(-1)
        township_keep = township_scores > self.score_threshold_l1

        if self.enforce_hierarchy:
            # Enforce spatial containment
            township_masks = outputs['pred_masks_l1'][township_keep]
            county_masks = outputs['pred_masks_l0'][county_keep]
            pred_parents = outputs['pred_parent'][township_keep].argmax(-1)

            # Clip township masks to parent county boundaries
            for i, (t_mask, parent_idx) in enumerate(zip(township_masks, pred_parents)):
                if parent_idx < len(county_masks):
                    parent_mask = county_masks[parent_idx]
                    township_masks[i] = t_mask * parent_mask

        return {
            'counties': {
                'labels': county_labels[county_keep],
                'scores': county_scores[county_keep],
                'boxes': outputs['pred_boxes_l0'][county_keep],
                'masks': outputs['pred_masks_l0'][county_keep],
            },
            'townships': {
                'labels': township_labels[township_keep],
                'scores': township_scores[township_keep],
                'boxes': outputs['pred_boxes_l1'][township_keep],
                'masks': township_masks,
                'parent_counties': pred_parents,
            }
        }
```

---

## Evaluation Metrics

### 8. Hierarchical mAP

```python
def compute_hierarchical_metrics(predictions, ground_truth):
    """
    Compute metrics for hierarchical segmentation.

    Returns:
        - mAP_l0: Mean AP for county level
        - mAP_l1: Mean AP for township level
        - mAP_hier: Hierarchical mAP (only counts if parent is also correct)
        - consistency_rate: % of townships correctly inside their county
    """
    metrics = {}

    # Standard mAP per level
    metrics['mAP_l0'] = compute_coco_map(
        predictions['counties'], ground_truth['counties']
    )
    metrics['mAP_l1'] = compute_coco_map(
        predictions['townships'], ground_truth['townships']
    )

    # Hierarchical mAP: township counts only if parent county is correct
    correct_townships = 0
    total_townships = 0

    for pred_t, gt_t in match_predictions(predictions['townships'], ground_truth['townships']):
        total_townships += 1

        # Check if parent county prediction matches ground truth
        pred_parent = pred_t['parent_county']
        gt_parent = gt_t['parent_county']

        if pred_parent == gt_parent:
            # Also verify spatial containment
            if is_contained(pred_t['mask'], predictions['counties'][pred_parent]['mask']):
                correct_townships += 1

    metrics['mAP_hier'] = correct_townships / max(total_townships, 1)

    # Consistency rate: spatial containment check
    contained = 0
    for pred_t in predictions['townships']:
        parent_mask = predictions['counties'][pred_t['parent_county']]['mask']
        if is_contained(pred_t['mask'], parent_mask, threshold=0.95):
            contained += 1

    metrics['consistency_rate'] = contained / max(len(predictions['townships']), 1)

    return metrics
```

---

## Ablation Study Design

| Experiment | L0 Decoder | L1 Decoder | Cross-Attn | Hier Loss | Parent Loss |
|------------|------------|------------|------------|-----------|-------------|
| Baseline (RF-DETR) | ✓ | - | - | - | - |
| Flat Multi-class | ✓ (390 classes) | - | - | - | - |
| Two-Head (no hier) | ✓ | ✓ | - | - | - |
| + Cross-Attention | ✓ | ✓ | ✓ | - | - |
| + Parent Loss | ✓ | ✓ | ✓ | - | ✓ |
| + Hierarchy Loss | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Full H-DETR** | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Expected Results (Hypothesis)

| Model | mAP@L0 | mAP@L1 | mAP_hier | Consistency |
|-------|--------|--------|----------|-------------|
| RF-DETR (county only) | ~75% | - | - | - |
| Flat 390-class | ~60% | ~45% | ~30% | ~70% |
| H-DETR (ours) | ~78% | ~65% | ~60% | ~95% |

**Key claims:**
1. Hierarchical modeling improves both L0 and L1 performance
2. Cross-level attention provides context for fine-grained prediction
3. Hierarchy loss ensures spatial consistency
4. Parent prediction enables interpretable reasoning

---

## File Structure (Implementation)

```
H-DETR/
├── rfdetr/
│   ├── models/
│   │   ├── hierarchical.py          # NEW: HierarchicalLWDETR
│   │   ├── cross_attention.py       # NEW: CrossLevelAttention
│   │   ├── lwdetr.py                # Modified: base class
│   │   └── transformer.py           # Unchanged
│   ├── losses/
│   │   ├── criterion.py             # Modified: base criterion
│   │   └── hierarchical_loss.py     # NEW: HierarchicalSetCriterion
│   ├── data/
│   │   └── hierarchical_dataset.py  # NEW: dataset with parent info
│   └── engine.py                    # Modified: training loop
├── configs/
│   └── h_detr_taiwan.yaml
├── tools/
│   └── eval_hierarchical.py         # NEW: evaluation script
└── README.md
```

---

## References

- RF-DETR: https://github.com/roboflow/rf-detr
- DETR: End-to-End Object Detection with Transformers (Carion et al., 2020)
- Deformable DETR (Zhu et al., 2021)
- DINOv2 (Oquab et al., 2023)
- LW-DETR (Chen et al., 2024)
