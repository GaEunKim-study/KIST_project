import argparse
import imageio
import scipy.misc
from models import hmr, SMPL
import config, constants
import torch
from torchvision.transforms import Normalize
import numpy as np
from utils.renderer import Renderer
from PIL import Image
import cv2
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import ipdb
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=str, help='Path to network checkpoint')
parser.add_argument('--img_path', required=True, type=str, help='Testing image path')
parser.add_argument('--mode', type=str, default='video', choices=['image', 'video'], help='image or video')
parser.add_argument('--output_dir',default="/home/gaeunter/PycharmProjects/pythonProject/RSC-Net/examples/", type=str, help='Testing image path')

def size_to_scale(size):
    if size >= 224:
        scale = 0
    elif 128 <= size < 224:
        scale = 1
    elif 64 <= size < 128:
        scale = 2
    elif 40 <= size < 64:
        scale = 3
    else:
        scale = 4
    return scale


def get_render_results(vertices, cam_t, renderer):
    #rendered_people_view_1 = renderer.visualize(vertices, cam_t, torch.ones((images.size(0), 3, 224, 224)).long() * 255)
    rendered_people_view_1 = renderer.visualize(vertices, cam_t, torch.ones((1, 3, 224, 224)).long() * 255)
    return rendered_people_view_1

def img2video():
    cap = cv2.VideoCapture(img_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])
    videoWrite = cv2.VideoWriter(output_dir + 'video_pose' + '.mp4', fourcc, fps, size)
    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)
    videoWrite.release()

def show3Dpose(vals, ax):
    I = np.array([1, 5, 6, 1, 1, 8, 10, 8, 13, 2, 3, 0])  # start points
    J = np.array([5, 6, 7, 2, 8, 10, 11, 13, 14, 3, 4, 1])  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z= [np.array([vals[:,:,j].cpu()[0][I[i]], vals[:,:,j].cpu()[0][J[i]]] ) for j in range(3)]
        ax.plot(x, z, -y, lw=2, color = (1, 0, 0))
    RADIUS = 1

    xroot, yroot, zroot = vals[0][0][0].cpu(), vals[0][0][1].cpu(), vals[0][0][2].cpu()

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_zlim3d([-RADIUS - yroot, RADIUS - yroot])

    ax.set_aspect('equal')
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)
def get_pose3D():
    args = parser.parse_args()
    img_path = args.img_path
    checkpoint_path = args.checkpoint
    output_dir=args.output_dir
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 351
    args.pad = (args.frames - 1) // 2
    args.n_joints, args.out_joints = 17, 17

    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    hmr_model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(checkpoint_path)
    hmr_model.load_state_dict(checkpoint, strict=False)
    hmr_model.eval()
    hmr_model.to(device)
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    img_renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES,
                            faces=smpl_neutral.faces)
    #capture
    cap = cv2.VideoCapture(img_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    print('\nGenerating 3D pose...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        im_size = img.shape
        im_size = im_size[0]
        im_scale = size_to_scale(im_size)
        img_up =np.array(Image.fromarray(img).resize([224, 224]))
        img_up = np.transpose(img_up.astype('float32'), (2, 0, 1)) / 255.0
        img_up = normalize_img(torch.from_numpy(img_up).float())
        images = img_up[None].to(device)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera, _ = hmr_model(images, scale=im_scale)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joint = pred_output.joints
            pred_cam_t = torch.stack([pred_camera[:, 1],
                                      pred_camera[:, 2],
                                      2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
                                     dim=-1)
        view_1= get_render_results(pred_vertices, pred_cam_t, img_renderer)
        view_1 = view_1[0].permute(1, 2, 0).numpy()

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title("Input", fontsize=12)
        ax.imshow(img)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title("Reconstruction", fontsize=12)
        show3Dpose(pred_joint, ax)
        output_dir_point = output_dir + 'pose/'
        os.makedirs(output_dir_point, exist_ok=True)
        plt.savefig(output_dir_point + str(('%04d' % i)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')

        """
        tmp = img_path.split('.')
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05)
        ax = plt.subplot(gs[0], projection='3d')
        output_dir_point = output_dir + 'pose/'
        os.makedirs(output_dir_point, exist_ok=True)
        plt.savefig(output_dir_point + str(('%04d' % i)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
    output_dir_point = output_dir + 'pose/'
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_point, '*.png')))
    for i in tqdm(range(len(image_3d_dir))):
        image_3d = plt.imread(image_3d_dir[i])
        font_size = 12

        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, img)
        ax.set_title("Input", fontsize=font_size)
        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = 12)
        output_dir_3D = output_dir + 'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d' % i)) + '_pose.png', dpi=200, bbox_inches='tight')
        """
    print('Generating 3D pose successful!')


if __name__ == '__main__':
    args = parser.parse_args()
    img_path = args.img_path
    checkpoint_path = args.checkpoint
    mode = args.mode
    output_dir=args.output_dir
    if mode == "image":
        normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        hmr_model = hmr(config.SMPL_MEAN_PARAMS)
        checkpoint = torch.load(checkpoint_path)
        hmr_model.load_state_dict(checkpoint, strict=False)
        hmr_model.eval()
        hmr_model.to(device)

        smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
        img_renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl_neutral.faces)

        img = imageio.imread(img_path)
        im_size = img.shape[0]
        im_scale = size_to_scale(im_size)
        img_up =np.array(Image.fromarray(img).resize([224, 224]))
        img_up = np.transpose(img_up.astype('float32'), (2, 0, 1)) / 255.0
        img_up = normalize_img(torch.from_numpy(img_up).float())
        images = img_up[None].to(device)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera, _ = hmr_model(images, scale=im_scale)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

            pred_cam_t = torch.stack([pred_camera[:, 1],
                                      pred_camera[:, 2],
                                      2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
                                     dim=-1)

        view_1, view_2 = get_render_results(pred_vertices, pred_cam_t, img_renderer)
        view_1 = view_1[0].permute(1, 2, 0).numpy()
        tmp = img_path.split('.')
        name_1 = '.'.join(tmp[:-2] + [tmp[-2] + '_view1'] + ['jpg'])
        imageio.imwrite(name_1, (view_1 * 255).astype(np.uint8))
    else:
        get_pose3D()
        img2video()





