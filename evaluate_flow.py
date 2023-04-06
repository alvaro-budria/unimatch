import numpy as np
import torch
import torch.nn.functional as F

from unimatch.unimatch import UniMatch

from glob import glob


@torch.no_grad()
def flow_unimatch( image1,
                   image2,
                   resume='pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth',
                   inference_dir='task_1_2_2/unimatch',
                   padding_factor=32,
                   upsample_factor=4,
                   feature_channels=128,
                   num_scales=2,
                   attn_splits_list=[2,8],
                   corr_radius_list=[-1,4],
                   prop_radius_list=[-1,1],
                   reg_refine=True,
                   num_reg_refine=6,
                   inference_size=None,
                   attn_type='swin',
                   num_head=1,
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   pred_bidir_flow=False,
                   ):
    """ Inference on a directory or a video """

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UniMatch(feature_channels=feature_channels,
                     num_scales=num_scales,
                     upsample_factor=upsample_factor,
                     num_head=num_head,
                     ffn_dim_expansion=ffn_dim_expansion,
                     num_transformer_layers=num_transformer_layers,
                     reg_refine=reg_refine,
                     task='flow').to(device)
    print(model)

    if resume:
        print('Load checkpoint: %s' % resume)

        loc = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(resume, map_location=loc)

        model.load_state_dict(checkpoint['model'], strict=True)

    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fixed_inference_size = inference_size
    transpose_img = False

    filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))
    print('%d images found' % len(filenames))

    image1 = np.array(image1).astype(np.uint8)
    image2 = np.array(image2).astype(np.uint8)

    if len(image1.shape) == 2:  # gray image
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))
    else:
        image1 = image1[..., :3]
        image2 = image2[..., :3]

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)
    print('image1.shape: ', image1.shape)

    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

    # resize to nearest size or specified size
    inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)

    results_dict = model(image1, image2,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task='flow',
                            pred_bidir_flow=pred_bidir_flow,
                            )

    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

    # resize back
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)

    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

    print('Done!')
    return flow, 'extra'
