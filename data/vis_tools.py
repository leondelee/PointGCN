# Author: llw
import open3d as o3d


def vis_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud('test.ply', pcd)
    pcd_load = o3d.io.read_point_cloud('test.ply')
    o3d.visualization.draw_geometries([pcd_load])