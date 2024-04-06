import numpy as np
import numpy as np
from visual_tools import draw_clouds_with_boxes
import open3d as o3d
import yaml
import sys
import pathlib

def dataloader(cloud_path , boxes_path):
    pcd = o3d.io.read_point_cloud(cloud_path)
    org_cloud = np.asarray(pcd.points)
    cloud = np.zeros((org_cloud.shape[0], 1))
    cloud = np.hstack((org_cloud, cloud))
    boxes = np.loadtxt(boxes_path).reshape(-1,7)
    return cloud,boxes

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " path_to_data")
        quit()
    for pcd_file in pathlib.Path(sys.argv[1]).glob("*.pcd"):
        pcd_path = pcd_file
        bb_path = pcd_file.with_suffix(".txt")
        cloud, boxes = dataloader(str(pcd_path), str(bb_path))
        draw_clouds_with_boxes(cloud, boxes)
