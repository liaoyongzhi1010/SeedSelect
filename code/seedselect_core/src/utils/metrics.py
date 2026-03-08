import numpy as np


def psnr(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def ssim(img1, img2):
    from skimage.metrics import structural_similarity
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    return structural_similarity(img1, img2, channel_axis=-1, data_range=1.0)


def lpips_metric(img1, img2, net='alex'):
    import torch
    import lpips as lp
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = lp.LPIPS(net=net).to(device)
    t1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t2 = torch.from_numpy(img2.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t1 = t1.to(device)
    t2 = t2.to(device)
    with torch.no_grad():
        val = loss_fn(t1, t2).item()
    return val


def chamfer_fscore(mesh_pred, mesh_gt, num_samples=16000, fscore_thresh=0.2):
    import trimesh
    import open3d as o3d

    pts_pred = mesh_pred.sample(num_samples)
    pts_gt = mesh_gt.sample(num_samples)

    pcd_pred = o3d.geometry.PointCloud()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pts_pred)
    pcd_gt.points = o3d.utility.Vector3dVector(pts_gt)

    # Compute distances
    dist_pred_to_gt = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    dist_gt_to_pred = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))

    cd = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()

    precision = (dist_pred_to_gt < fscore_thresh).mean()
    recall = (dist_gt_to_pred < fscore_thresh).mean()
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return cd, fscore
