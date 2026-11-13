# Fix for the MegaDepth-1500 evaluation in the RIPE fork of Glue Factory
# (https://github.com/JohannesK14/glue-factory).
#
# The relative-pose estimator wrapper calls poselib.estimate_relative_pose with the old
# signature. Recent PoseLib versions expect
#   (pts0: float64[m,2], pts1: float64[m,2], cam0, cam1, ransac_opt: dict, bundle_opt: dict)
# so the call fails with "incompatible function arguments" after feature extraction finished.
# We cast the points to float64 and pass an empty bundle-adjustment options dict.
#
# Usage: python scripts/patch_gluefactory_poselib.py /content/glue-factory

import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/glue-factory")
p = root / "gluefactory/robust_estimators/relative_pose/poselib.py"
s = p.read_text()

if 'astype("float64")' in s:
    print("already patched")
    sys.exit(0)

s = s.replace(
    "pts0.numpy(),\n            pts1.numpy(),",
    'pts0.numpy().astype("float64"),\n            pts1.numpy().astype("float64"),',
)
s = s.replace(
    "**OmegaConf.to_container(self.conf.options),\n            },\n        )",
    "**OmegaConf.to_container(self.conf.options),\n            },\n            {},\n        )",
)
assert 'astype("float64")' in s and "},\n            {},\n        )" in s, "patch did not apply, file layout changed"
p.write_text(s)
print(s[s.find("M, info") : s.find("success =")])
