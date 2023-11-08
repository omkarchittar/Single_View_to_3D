from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
import torch.nn.functional as F
import numpy as np
    
# Pix2Vox
class VoxDecoder(nn.Module):
    def __init__(self):
        super(VoxDecoder, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(4),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(4, 1, kernel_size=1),
            nn.BatchNorm3d(1)
        )

    def forward(self, feats):
        vox = feats.view((-1, 64, 2, 2, 2))
        vox = self.layer1(vox)
        vox = self.layer2(vox)
        vox = self.layer3(vox)
        vox = self.layer4(vox)
        vox = self.layer5(vox)
        return vox

class PointDecoder(nn.Module):
    def __init__(self, point_size):
        super(PointDecoder, self).__init__()
        self.point_size = point_size
        self.layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.point_size*3),
        )

    def forward(self, feats):
        points = self.layer(feats)
        points = points.reshape((-1, self.point_size, 3))
        return points

class MeshDecoder(nn.Module):
    def __init__(self, vert_size):
        super(MeshDecoder, self).__init__()
        self.vert_size = vert_size
        self.layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.vert_size * 3)
        )

    def forward(self, feats):
        meshes = self.layer(feats)
        meshes = meshes.view(-1, self.vert_size, 3)
        return meshes


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 1 x 32 x 32 x 32
            # pass
            # TODO: 
            self.decoder = VoxDecoder()     
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = PointDecoder(self.n_point)         
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  # 2562
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            self.decoder = MeshDecoder(mesh_pred.verts_packed().shape[0])            

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)           
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)           
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)       
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super(ResnetBlockFC, self).__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class ImplicitMLPDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    # sample_mode = "bilinear"
    def __init__(self, dim=3, c_dim=512,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=1):
        super(ImplicitMLPDecoder, self).__init__()
        print('Implicit Local Decoder...')
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.fc_p = nn.Linear(3, 256)
    
        self.xyz_grid = self.build_grid([32,32,32])

        self.fc_c = nn.ModuleList([
            nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        self.out_dim = out_dim
        
        self.actvn = F.relu

        self.padding = padding

    def build_grid(self,resolution):
        ranges = [np.linspace(0., res-1., num=res) for res in resolution]
        
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], resolution[2], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        grid = torch.from_numpy(grid).cuda()
        
        # grid = grid.permute(0,-1,1,2,3)        
        return grid

    def forward(self, featmap):
        # pcl is None
        # pcl_mem is point cloud in mem coordinate
        # c_plane is the 3d feature grid
        B = featmap.shape[0]
        
        pcl_mem = self.xyz_grid

        pcl_mem_ = pcl_mem.reshape([1,-1,3]).repeat([B,1,1])
        
        pcl_norm = (pcl_mem_/32) -0.5

        c = featmap.unsqueeze(dim=1).repeat(1,pcl_norm.shape[1],1)

        
        net = self.fc_p(pcl_norm)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net)).permute(0,2,1)
        out = out.reshape(B, self.out_dim, 32, 32, 32)
        return out