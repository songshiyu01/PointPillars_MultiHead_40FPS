import numpy as np
import numpy as np
from visual_tools import draw_clouds_with_boxes
import open3d as o3d
import yaml
import sys

def dataloader(cloud_path , boxes_path):
    pcd = o3d.io.read_point_cloud(cloud_path)
    org_cloud = np.asarray(pcd.points)
    cloud = np.zeros((org_cloud.shape[0], 1))
    cloud = np.hstack((org_cloud, cloud))
    boxes = np.loadtxt(boxes_path).reshape(-1,7)
    return cloud,boxes

if __name__ == "__main__":
   pre_path = "/root/OpenPCDet/data/safeai_cache/v1.2-safeai-cache/samples/LIDAR_TOP_PLATE/"
   pcd_path = pre_path + sys.argv[1] + ".pcd"
   bb_path = pre_path + sys.argv[1] + ".txt"
   cloud, boxes = dataloader(pcd_path, bb_path)
   draw_clouds_with_boxes(cloud ,boxes)