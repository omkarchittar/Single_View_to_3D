import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image
import numpy as np
import imageio
from PIL import Image, ImageDraw

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--vis_freq', default=100, type=str)
    parser.add_argument('--batch_size', default=8, type=str)
    parser.add_argument('--num_workers', default=2, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=10000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.5, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)
    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)
    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'output/eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        print(voxels_src.shape)
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.0)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics


def evaluate_model(args, voxel_size = 32, device = 'cuda', duration = 200,):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        color = [0, 0.7, 0.7]

        # TODO:
        # if (step % args.vis_freq) == 0:
            # visualization block
        img_step = step % 100 
        num = step // 100 

        if img_step == 0:
# ---------- VOXEL ------------
            if args.type == "vox":

                rgb_image = images_gt.squeeze()
                rgb_image = (rgb_image*255).byte().cpu().numpy()
                plt.imsave(f"output/RGB_voxel_{num}.png", rgb_image)
            
            #-------------------- ground truth mesh -------------------------------
                renderer = get_mesh_renderer(image_size=args.image_size)
                mesh = mesh_gt
                textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0), device = 'cuda')
                textures = textures * torch.tensor([0.7, 0.7, 1], device = 'cuda')
                mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
                mesh = mesh.to(device)

                # Initialize an empty list to store rendered images
                renders = []
                for theta in range(0, 360, 10):
                    R = torch.tensor([
                        [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                        [0.0, 1.0, 0.0],
                        [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
                    ])
                    T = torch.tensor([[0, 0, 2]])  # Move the camera to the side

                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=T, fov=60, device=device)
                    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device='cuda')
                    rend = renderer(mesh, cameras=cameras, lights=lights)
                    rend = rend[0, ..., :3].detach().cpu().numpy()  # (N, H, W, 3)
                    renders.append(rend)

                images = []
                for i, r in enumerate(renders):
                    image = Image.fromarray((r * 255).astype(np.uint8))
                    images.append(np.array(image))
                imageio.mimsave(f"output/SingletoVox_gtmesh_{num}.gif", images, duration=duration, loop = 0)

            
            # ------------------- voxel ground truth ---------------------------
                vox_gt = feed_dict['voxels'].to(args.device)

                # get the renderer
                renderer = get_mesh_renderer(image_size = 512)

                mesh_vgt = pytorch3d.ops.cubify(vox_gt[0], thresh=0.5)
                textures = torch.ones_like(mesh_vgt.verts_list()[0].unsqueeze(0))
                textures = textures * torch.tensor([0.7, 0.7, 1], device='cuda')
                mesh_vgt.textures=pytorch3d.renderer.TexturesVertex(textures)
                mesh_vgt = mesh_vgt.to(device)

                # Initialize an empty list to store rendered images
                renders = []
                for theta in range(0, 360, 10):
                    R = torch.tensor([
                        [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                        [0.0, 1.0, 0.0],
                        [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
                    ])
                    T = torch.tensor([[0, 0, 3]])   # Move the camera to the side

                    # prepare the camera:
                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R = R.unsqueeze(0), T = T, fov=60, device='cuda')
                    lights = pytorch3d.renderer.PointLights(location = [[0, 0, -4]], device = 'cuda')

                    rend = renderer(mesh_vgt, cameras=cameras, lights=lights)
                    rend = rend[0, ..., :3].cpu().detach().numpy().clip(0,1) # (B, H, W, 4) -> (H, W, 3)
                    renders.append(rend)

                images = []
                for i, r in enumerate(renders):
                    image = Image.fromarray((r * 255).clip(0,255).astype(np.uint8))
                    images.append(np.array(image))
                imageio.mimsave(f"output/SingletoVox_gtvox_{num}.gif", images, duration=duration, loop = 0)

            
            # ------------------- voxel prediction ---------------------------
                # get the renderer
                renderer = get_mesh_renderer(image_size = 512)

                mesh_vpred = pytorch3d.ops.cubify(predictions[0], thresh=0.0)
                textures = torch.ones_like(mesh_vpred.verts_list()[0].unsqueeze(0))
                textures = textures * torch.tensor([0.7, 0.7, 1], device='cuda')
                mesh_vpred.textures=pytorch3d.renderer.TexturesVertex(textures)
                mesh_vpred = mesh_vpred.to(device)
                # Initialize an empty list to store rendered images
                renders = []
                for theta in range(0, 360, 10):
                    R = torch.tensor([
                        [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                        [0.0, 1.0, 0.0],
                        [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
                    ])
                    T = torch.tensor([[0, 0, 3]])   # Move the camera to the side

                    # prepare the camera:
                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R = R.unsqueeze(0), T = T, fov=60, device='cuda')
                    lights = pytorch3d.renderer.PointLights(location = [[0, 0, -4]], device = 'cuda')

                    rend = renderer(mesh_vpred, cameras=cameras, lights=lights)
                    rend = rend[0, ..., :3].cpu().detach().numpy().clip(0,1) # (B, H, W, 4) -> (H, W, 3)
                    renders.append(rend)

                images = []
                for i, r in enumerate(renders):
                    image = Image.fromarray((r * 255).clip(0,255).astype(np.uint8))
                    images.append(np.array(image))
                imageio.mimsave(f"output/SingletoVox_pred_{num}.gif", images, duration=duration, loop = 0)


# ---------- POINT CLOUD ------------
            if args.type == "point":

                rgb_image = images_gt.squeeze()
                rgb_image = (rgb_image*255).byte().cpu().numpy()
                plt.imsave(f"output/RGB_pointcloud_{num}.png", rgb_image)

            #-------------------- ground truth mesh -------------------------------
                renderer = get_mesh_renderer(image_size=args.image_size)
                vertices = mesh_gt.verts_list()
                faces = mesh_gt.faces_list()
                vertices = vertices[0]  # (N_v, 3) -> (N_v, 3)
                faces = faces[0]  # (N_f, 3) -> (N_f, 3)    
                vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
                faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
                textures = torch.ones_like(vertices)  # (1, N_v, 3)
                # textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
                textures = textures * torch.tensor(color)
                mesh = pytorch3d.structures.Meshes(
                    verts=vertices,
                    faces=faces,
                    textures=pytorch3d.renderer.TexturesVertex(textures),
                )
                mesh = mesh.to(device)

                # Initialize an empty list to store rendered images
                renders = []
                for theta in range(0, 360, 10):
                    R = torch.tensor([
                        [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                        [0.0, 1.0, 0.0],
                        [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
                    ])
                    T = torch.tensor([[0, 0, 2]])  # Move the camera to the side
                    
                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=T, fov=60, device=device)
                    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
                    rend = renderer(mesh, cameras=cameras, lights=lights)
                    rend = rend[0, ..., :3].detach().cpu().numpy()  # (N, H, W, 3)
                    renders.append(rend)

                images = []
                for i, r in enumerate(renders):
                    image = Image.fromarray((r * 255).astype(np.uint8))
                    images.append(np.array(image))
                imageio.mimsave(f"output/SingletoPC_gtmesh_{num}.gif", images, duration=duration, loop = 0)

            
            # ------------------- point cloud prediction ---------------------------                
                pointcloud_tgt = predictions[0].to(device)

                points = pointcloud_tgt
                color = (points - points.min()) / (points.max() - points.min())

                sphere_point_cloud = pytorch3d.structures.Pointclouds(points=[points], 
                                                                    features=[color],)

                renders = []
                for theta in range(0, 360, 10):
                    R = torch.tensor([
                        [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                        [0.0, 1.0, 0.0],
                        [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
                    ])
                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=[[0, 0, 2]], device=device)
                    renderer = get_points_renderer(image_size=args.image_size, device=device)
                    rend = renderer(sphere_point_cloud, cameras=cameras)
                    rend = rend[0, ..., :3].cpu().detach().numpy()
                    renders.append(rend)

                images = []
                for i, r in enumerate(renders):
                    image = Image.fromarray((r * 255).astype(np.uint8))
                    draw = ImageDraw.Draw(image)
                    images.append(np.array(image))
                imageio.mimsave(f"output/SingletoPC_pred_{num}.gif", images, duration=4, loop = 0)   # change here

            
            # ------------------- point cloud ground truth ---------------------------       
                gt_points = sample_points_from_meshes(mesh_gt, args.n_points)         
                pointcloud_tgt = gt_points[0].to(device)

                points = pointcloud_tgt
                color = (points - points.min()) / (points.max() - points.min())

                sphere_point_cloud = pytorch3d.structures.Pointclouds(points=[points], 
                                                                    features=[color],)

                renders = []
                for theta in range(0, 360, 10):
                    R = torch.tensor([
                        [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                        [0.0, 1.0, 0.0],
                        [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
                    ])
                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=[[0, 0, 2]], device=device)
                    renderer = get_points_renderer(image_size=args.image_size, device=device)
                    rend = renderer(sphere_point_cloud, cameras=cameras)
                    rend = rend[0, ..., :3].cpu().detach().numpy()
                    renders.append(rend)

                images = []
                for i, r in enumerate(renders):
                    image = Image.fromarray((r * 255).astype(np.uint8))
                    draw = ImageDraw.Draw(image)
                    images.append(np.array(image))
                imageio.mimsave(f"output/SingletoPC_gt_{num}.gif", images, duration=4, loop = 0)   # change here


# ---------- MESH ------------
            if args.type == "mesh":

                rgb_image = images_gt.squeeze()
                rgb_image = (rgb_image*255).byte().cpu().numpy()
                plt.imsave(f"output/RGB_mesh_{num}.png", rgb_image)

            # ------------------- mesh prediction --------------------------- 
                renderer = get_mesh_renderer(image_size=args.image_size) 
                vertices = predictions.verts_list()
                faces = predictions.faces_list()
                vertices = vertices[0].to(device)  # (N_v, 3) -> (N_v, 3)
                faces = faces[0].to(device)  # (N_f, 3) -> (N_f, 3)    
                vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
                faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
                textures = torch.ones_like(vertices)  # (1, N_v, 3)
                textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
                mesh = pytorch3d.structures.Meshes(
                    verts=vertices,
                    faces=faces,
                    textures=pytorch3d.renderer.TexturesVertex(textures),
                )
                mesh = mesh.to(device)

                # Initialize an empty list to store rendered images
                renders = []
                for theta in range(0, 360, 10):
                    R = torch.tensor([
                        [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                        [0.0, 1.0, 0.0],
                        [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
                    ])
                    T = torch.tensor([[0, 0, 2]])  # Move the camera to the side
                    
                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=T, fov=60, device=device)
                    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
                    rend = renderer(mesh, cameras=cameras, lights=lights)
                    rend = rend[0, ..., :3].detach().cpu().numpy()  # (N, H, W, 3)
                    renders.append(rend)

                images = []
                for i, r in enumerate(renders):
                    image = Image.fromarray((r * 255).astype(np.uint8))
                    images.append(np.array(image))
                imageio.mimsave(f"output/SingletoMesh_pred_{num}.gif", images, duration=duration, loop = 0)

            
            # ------------------- ground truth mesh ---------------------------  
                renderer = get_mesh_renderer(image_size=args.image_size)
                vertices = mesh_gt.verts_list()
                faces = mesh_gt.faces_list()
                vertices = vertices[0].to(device)  # (N_v, 3) -> (N_v, 3)
                faces = faces[0].to(device)  # (N_f, 3) -> (N_f, 3)    
                vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
                faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
                textures = torch.ones_like(vertices)  # (1, N_v, 3)
                textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
                mesh = pytorch3d.structures.Meshes(
                    verts=vertices,
                    faces=faces,
                    textures=pytorch3d.renderer.TexturesVertex(textures),
                )
                mesh = mesh.to(device)

                # Initialize an empty list to store rendered images
                renders = []
                for theta in range(0, 360, 10):
                    R = torch.tensor([
                        [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                        [0.0, 1.0, 0.0],
                        [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
                    ])
                    T = torch.tensor([[0, 0, 2]])  # Move the camera to the side
                    
                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=T, fov=60, device=device)
                    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
                    rend = renderer(mesh, cameras=cameras, lights=lights)
                    rend = rend[0, ..., :3].detach().cpu().numpy()  # (N, H, W, 3)
                    renders.append(rend)

                images = []
                for i, r in enumerate(renders):
                    image = Image.fromarray((r * 255).astype(np.uint8))
                    images.append(np.array(image))
                imageio.mimsave(f"output/SingletoMesh_gtmesh_{num}.gif", images, duration=duration, loop = 0)
      
        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args, voxel_size = 32, device = 'cuda', duration = 200,)
