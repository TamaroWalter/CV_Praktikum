import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn

from data_loading import extract_patch_features_rgb_sobel, load_union_boundary_map


# ---------- MLP ----------
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)

def compute_sobel_maps(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy

# ---------- evaluation (restliche 90%) ----------
def evaluate_on_rest(pred_mask, gt_union, train_mask):
    eval_mask = ~train_mask
    y_true = gt_union[eval_mask].astype(np.int32)
    y_pred = pred_mask[eval_mask].astype(np.int32)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    print("Evaluation on remaining pixels:")
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    return precision, recall, f1, accuracy

def best_threshold_on_rest(prob_edge, gt_union, train_mask, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.50, 19)  # 0.05..0.50

    eval_mask = ~train_mask
    y_true = gt_union[eval_mask].astype(np.int32)
    p = prob_edge[eval_mask].astype(np.float32)

    best = None
    for t in thresholds:
        y_pred = (p >= t).astype(np.int32)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        if best is None or f1 > best["f1"]:
            best = {"thr": float(t), "f1": float(f1), "precision": float(precision), "recall": float(recall),
                    "TP": int(TP), "FP": int(FP), "FN": int(FN)}

    print(f"[thr sweep] best thr={best['thr']:.3f}  F1={best['f1']:.4f}  "
          f"P={best['precision']:.4f}  R={best['recall']:.4f}  TP={best['TP']} FP={best['FP']} FN={best['FN']}")
    return best


# ---------- inference: full probability mask ----------
@torch.no_grad()
def predict_prob_mask(model, img_bgr, patch, device, gx, gy, batch_pixels=65536):
    """
    Returns prob_edge map (H,W) float32 in [0,1]
    """
    model.eval()
    H, W, _ = img_bgr.shape

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y = ys.reshape(-1)
    x = xs.reshape(-1)

    prob = np.zeros((H * W,), dtype=np.float32)

    for start in range(0, len(y), batch_pixels):
        end = min(len(y), start + batch_pixels)
        X = extract_patch_features_rgb_sobel(img_bgr, gx, gy, y[start:end], x[start:end], patch)  # (B,D)
        mu = X.mean(axis=0, keepdims=True).astype(np.float32)
        sigma = (X.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        X = (X - mu) / sigma
        X = torch.from_numpy(X).to(device)
        logits = model(X)
        p = torch.softmax(logits, dim=1)[:, 1]  # p(edge)
        prob[start:end] = p.cpu().numpy()

    return prob.reshape(H, W)


# ---------- main: train one image (no DataLoader) ----------
def train_one_image(
    img_bgr,
    union_label,
    patch=5,
    train_frac=0.10,
    neg_pos_ratio=10,
    epochs=10,
    lr=1e-3,
    device="cpu",
    batch_size=8192,
    seed=0,
    out_dir="out_demo",
    base_name="image"
):
    """
    Trains on <=10% pixels with pos/neg sampling using neg_pos_ratio.
    Produces prob map, pred mask via threshold, evaluates on remaining pixels,
    and saves output images.
    Returns: model, pred_mask, prob_edge, train_mask
    """
    gx, gy = compute_sobel_maps(img_bgr)
    rng = np.random.default_rng(seed)
    H, W = union_label.shape
    N = H * W
    max_train = int(np.floor(train_frac * N))

    pos = np.argwhere(union_label == 1)
    neg = np.argwhere(union_label == 0)

    # ----- choose train coords within budget -----
    if len(pos) == 0:
        idx = rng.choice(N, size=max_train, replace=False)
        y_train = idx // W
        x_train = idx % W
        Y = np.zeros((max_train,), dtype=np.int64)
    else:
        # aim: n_neg ≈ neg_pos_ratio * n_pos and total <= max_train
        # solve n_pos + neg_pos_ratio*n_pos <= max_train  => n_pos <= max_train/(1+ratio)
        max_pos = max_train // (1 + neg_pos_ratio)
        n_pos = min(len(pos), max(1, max_pos))
        pos_sel = pos[rng.choice(len(pos), size=n_pos, replace=False)]

        n_neg = min(len(neg), max_train - n_pos)
        n_neg = min(n_neg, n_pos * neg_pos_ratio)  # target ratio, capped by budget
        neg_sel = neg[rng.choice(len(neg), size=n_neg, replace=False)]

        coords = np.vstack([pos_sel, neg_sel])
        y_train = coords[:, 0]
        x_train = coords[:, 1]
        Y = np.concatenate([
            np.ones((len(pos_sel),), dtype=np.int64),
            np.zeros((len(neg_sel),), dtype=np.int64),
        ])

        perm = rng.permutation(len(Y))
        y_train, x_train, Y = y_train[perm], x_train[perm], Y[perm]

    train_mask = np.zeros((H, W), dtype=bool)
    train_mask[y_train, x_train] = True

    # debug stats
    print(f"union unique values: {np.unique(union_label)[:10]}  ... max= {union_label.max()}")
    print(f"train samples: {len(Y)}  positives: {int(Y.sum())}  negatives: {len(Y)-int(Y.sum())}  pos_rate={Y.mean():.4f}")
    print(f"global union positives in image: {int(union_label.sum())}  total pixels: {union_label.size}  global_pos_rate={union_label.mean():.4f}")

    # ----- build train features once -----
    X = extract_patch_features_rgb_sobel(img_bgr, gx, gy, y_train, x_train, patch)
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    sigma = (X.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    X = (X - mu) / sigma
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    # ----- train MLP -----
    feat_dim = patch*patch*3 + 2*patch*patch
    model = MLP(feat_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    M = len(Y)
    for ep in range(1, epochs + 1):
        order = rng.permutation(M)
        total_loss = 0.0

        for start in range(0, M, batch_size):
            end = min(M, start + batch_size)
            idx = order[start:end]

            xb = X_t[idx].to(device)
            yb = Y_t[idx].to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item()) * (end - start)

        print(f"epoch {ep}/{epochs} loss={total_loss/M:.4f}")

    # ----- inference: probability + threshold -----
    prob_edge = predict_prob_mask(model, img_bgr, patch, device=device, gx=gx, gy=gy)
    best = best_threshold_on_rest(prob_edge, union_label, train_mask)
    best_thr = best["thr"]
    pred_mask = (prob_edge >= best_thr).astype(np.uint8)

    # optional: zusätzlich eine feste Schwelle speichern
    # pred_mask_fixed = (prob_edge >= threshold).astype(np.uint8)

    print(f"prob stats: min={prob_edge.min():.6f} max={prob_edge.max():.6f} mean={prob_edge.mean():.6f}")
    evaluate_on_rest(pred_mask, union_label, train_mask)

    # ----- save outputs -----
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, f"{base_name}_gt_union.png"),
                (union_label.astype(np.uint8) * 255))

    cv2.imwrite(os.path.join(out_dir, f"{base_name}_prob_edge.png"),
                (np.clip(prob_edge, 0, 1) * 255).astype(np.uint8))

    cv2.imwrite(os.path.join(out_dir, f"{base_name}_pred_bestthr{best_thr:.3f}.png"),
                (pred_mask * 255).astype(np.uint8))

    train_vis = (train_mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_train_pixels.png"), train_vis)

    return model, pred_mask, prob_edge, train_mask


# ---------- convenience wrapper: run on a BSDS test image ----------
def run_on_bsds_image(
    root="BSDS500/BSDS500/data/",
    split="test",
    image_name="3063.jpg",
    patch=5,
    train_frac=0.10,
    neg_pos_ratio=10,
    epochs=10,
    lr=1e-3,
    device="cpu",
    threshold=0.25,
    out_dir="out_demo",
    seed=0
):
    img_dir = os.path.join(root, "images", split)
    gt_dir  = os.path.join(root, "groundTruth", split)

    img_path = os.path.join(img_dir, image_name if image_name.endswith(".jpg") else image_name + ".jpg")
    base = os.path.splitext(os.path.basename(img_path))[0]
    gt_path = os.path.join(gt_dir, base + ".mat")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    union = load_union_boundary_map(gt_path)  # 0/1

    model, pred, prob, train_mask = train_one_image(
        img_bgr=img,
        union_label=union,
        patch=patch,
        train_frac=train_frac,
        neg_pos_ratio=neg_pos_ratio,
        epochs=epochs,
        lr=lr,
        device=device,
        batch_size=8192,
        seed=seed,
        out_dir=out_dir,
        base_name=base
    )

    print("saved to:", out_dir)
    print("image:", img_path)
    return model, pred, prob, union, train_mask

run_on_bsds_image(
    root="BSDS500/BSDS500/data/",
    split="test",
    image_name="2018.jpg",
    patch=5,
    train_frac=0.10,
    neg_pos_ratio=10,
    epochs=10,
    device="cpu",
)



