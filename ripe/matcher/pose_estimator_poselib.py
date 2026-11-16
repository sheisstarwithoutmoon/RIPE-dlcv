import poselib
import torch


class PoseLibRelativePoseEstimator:
    """PoseLibRelativePoseEstimator estimates the fundamental matrix using poselib library.
    It uses the poselib's estimate_fundamental function to compute the fundamental matrix and inliers based on the provided points.
    Args:
        None
    """

    def __init__(self):
        pass

    def __call__(self, pts0, pts1, inl_th):
        F, info = poselib.estimate_fundamental(
            # float64: recent PoseLib builds reject float32 arrays (same fix as for Glue Factory)
            pts0.detach().float().cpu().numpy().astype("float64"),
            pts1.detach().float().cpu().numpy().astype("float64"),
            {
                "max_epipolar_error": inl_th,
            },
        )

        success = F is not None
        if success:
            inliers = info.pop("inliers")
            inliers = torch.tensor(inliers, dtype=torch.bool, device=pts0.device)
        else:
            inliers = torch.zeros(pts0.shape[0], dtype=torch.bool, device=pts0.device)

        return F, inliers
