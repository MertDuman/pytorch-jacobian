import torch
import torch.autograd as AG
import torch.nn.functional as F
import numpy as np

from typing import Union


def get_batch_jacobian_vmap_jacrev(*x, model, Nin=None, Nout=None, chunk_size=None, batch_size=None, flatten=False, target_outputs=None, target_inputs=0, input_vmap_dims=0, yshape=None):
    '''
    Calculates the batch Jacobian, following: https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html

    Parameters:
    *x
        Series of batch of inputs to the model, of shape (N, ...).
    model
        Model to calculate the Jacobian of. If your model takes multiple inputs use target_inputs and input_vmap_dims.
    Nin
        Number of differentiated inputs. We'll try to infer this from target_inputs if passed. Otherwise, defaults to 1.
    Nout
        Number of differentiated outputs. We'll try to infer this from target_outputs OR yshape if passed. Otherwise, defaults to 1.
    chunk_size
        Tells pytorch to calculate the Jacobian chunk_size rows at a time. Useful for large Jacobian matrices that don't fit in memory.
    batch_size
        Tells pytorch to process this many batches at a time.
    flatten
        Whether to flatten the Jacobian into a (N, num_fy, num_fx) tensor. If False, the Jacobian will be (N, *yshape, *xshape).
    target_outputs
        Calculate the Jacobian for these outputs. If None, calculate the Jacobian for all outputs.
    target_inputs
        Same as torch.func.jacrev's argnums. Specifies which input(s) to differentiate wrt. Can be a tuple of indices.
    input_vmap_dims
        Same as torch.vmap's in_dims. Specifies which dimensions of the input to vectorize. If your model takes multiple inputs, pass a tuple of integers and use None if the input is not vectorized.
    yshape
        Shape of your model's differentiated outputs (without the batch dimension). Used to reshape the Jacobian. If None, assumes xshape == yshape. If your model has multiple outputs, pass a tuple of shapes.
    '''
    wrapper = model
    def target_wrapper(*args, **kwargs):
        outs = model(*args, **kwargs)
        return tuple(outs[i] for i in target_outputs)
    if target_outputs is not None:
        wrapper = target_wrapper

    if Nin is None:
        try: Nin = len(target_inputs)
        except TypeError: Nin = 1
    
    if Nout is None:
        try: Nout = len(target_outputs)
        except TypeError:
            try: Nout = len(yshape)
            except TypeError: Nout = 1

    jacobian = torch.vmap(torch.func.jacrev(wrapper, chunk_size=chunk_size, argnums=target_inputs), chunk_size=batch_size, in_dims=input_vmap_dims)(*x)

    if flatten:
        if Nout > 1:    # <- multiple outputs
            jacobians = []
            for i in range(len(jacobian)):
                if Nin > 1:     # <- multiple outputs & inputs
                    jacobians.append([])
                    for j in range(len(jacobian[i])):
                        N, num_fx = x[j].shape[0], np.prod(x[j].shape[1:])
                        num_fy = num_fx if yshape is None else np.prod(yshape[i])
                        jacobians[i].append(jacobian[i][j].view(N, num_fy, num_fx))
                    jacobians[i] = tuple(jacobians[i])
                else:           # <- multiple outputs & single input
                    N, num_fx = x[i].shape[0], np.prod(x[i].shape[1:])
                    num_fy = num_fx if yshape is None else np.prod(yshape[i])
                    jacobians.append(jacobian[i].view(N, num_fy, num_fx))
            jacobian = tuple(jacobians)
        elif Nin > 1:   # <- single output & multiple inputs
            jacobians = []
            for i in range(len(jacobian)):
                N, num_fx = x[i].shape[0], np.prod(x[i].shape[1:])
                num_fy = num_fx if yshape is None else np.prod(yshape)
                jacobians.append(jacobian[i].view(N, num_fy, num_fx))
            jacobian = tuple(jacobians) 
        else:   # <- single output & single input
            N, num_fx = x[0].shape[0], np.prod(x[0].shape[1:])
            num_fy = num_fx if yshape is None else np.prod(yshape)
            jacobian = jacobian.view(N, num_fy, num_fx)
    return jacobian




def window_image(img: torch.Tensor, window_size: int, stride: int = None, image_pad: Union[int, tuple] = None, patch_pad: Union[int, tuple] = None, pad_mode: str = 'constant') -> torch.Tensor:
    """
    Window an image into patches of size window_size x window_size

    Parameters:
    img
        Image to be windowed, of shape (N, C, H, W)
    window_size
        Size of the window
    stride
        Stride of the window, defaults to window_size
    image_pad
        Pad the image before windowing. image_pad is a 4-tuple of pads for: (left, right, top, bottom). Or a single int, in which case, the same pad is applied to all 4 sides.
    patch_pad
        Patches will leak to one another by pad amount.
        patch_pad is a 4-tuple of pads for: (left, right, top, bottom). Or a single int, in which case, the same pad is applied to all 4 sides.
        This helps creatings windows of all sizes, and gets rid of the need for window_size/stride arithmetic to do so.
    pad_mode
        Same as torch.nn.functional.pad's mode. Defaults to 'constant', possible values are 'constant', 'reflect', 'replicate' or 'circular'.
    Returns:
    patches
        windowed image, of shape (N, num_patches, C, window_size, window_size)
    """
    assert image_pad is None or isinstance(image_pad, int) or len(image_pad) == 4, "pad must be int, or 4-tuple"
    assert patch_pad is None or isinstance(patch_pad, int) or len(patch_pad) == 4, "pad must be int, or 4-tuple"
    image_pad = 0 if image_pad is None else image_pad
    patch_pad = 0 if patch_pad is None else patch_pad
    stride = window_size if stride is None else stride
    
    image_pad = (image_pad, image_pad, image_pad, image_pad) if isinstance(image_pad, int) else image_pad
    patch_pad = (patch_pad, patch_pad, patch_pad, patch_pad) if isinstance(patch_pad, int) else patch_pad
    comb_pad = tuple(a + b for a, b in zip(image_pad, patch_pad))
    img = F.pad(img, comb_pad, mode=pad_mode)

    window_size_H = window_size + patch_pad[2] + patch_pad[3]
    window_size_W = window_size + patch_pad[0] + patch_pad[1]

    N, C, _, _ = img.shape
    patches = img.unfold(2, window_size_H, stride).unfold(3, window_size_W, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(N, -1, C, window_size_H, window_size_W)
    return patches

def unwindow_image(patches: torch.Tensor, window_size: int, H: int, W: int, stride: int = None, image_pad: Union[int, tuple] = None, patch_pad: Union[int, tuple] = None) -> torch.Tensor:
    """
    Unwindow patches of size window_size x window_size into an image of size H x W

    Parameters:
        patches
            Windowed image, of shape (N, num_patches, C, window_size, window_size)
        window_size
            Size of the window
        H
            Height of the original image
        W
            Width of the original image
        stride
            Stride of the window, defaults to window_size
        image_pad
            Pad applied to the image before windowing
        patch_pad
            Patches leak to one another by pad amount.
            patch_pad is a 4-tuple of pads for: (left, right, top, bottom). Or a single int, in which case, the same pad is applied to all 4 sides.
            Use the same patch_pad as used in window_image
    Returns:
        img
            image of shape (N, C, H, W)
    """
    assert image_pad is None or isinstance(image_pad, int) or len(image_pad) == 4, "pad must be int, or 4-tuple"
    assert patch_pad is None or isinstance(patch_pad, int) or len(patch_pad) == 4, "pad must be int, or 4-tuple"
    image_pad = 0 if image_pad is None else image_pad
    patch_pad = 0 if patch_pad is None else patch_pad
    stride = window_size if stride is None else stride
    assert stride <= window_size, "can't reconstruct image unless stride <= window_size"
    assert window_size % stride == 0, "can't reconstruct image unless stride divides window_size (i could fix this but i'm lazy)"
    step = window_size // stride

    N, _, C, _, _ = patches.shape  # can get window_size from patches too..

    image_pad = (image_pad, image_pad, image_pad, image_pad) if isinstance(image_pad, int) else image_pad
    H = H + image_pad[2] + image_pad[3]
    W = W + image_pad[0] + image_pad[1]
    patch_pad = (patch_pad, patch_pad, patch_pad, patch_pad) if isinstance(patch_pad, int) else patch_pad
    window_size_H = window_size + patch_pad[2] + patch_pad[3]
    window_size_W = window_size + patch_pad[0] + patch_pad[1]

    H_patches = (H + patch_pad[2] + patch_pad[3] - window_size_H) // stride + 1
    H_residue = (H + patch_pad[2] + patch_pad[3] - window_size_H) % stride
    W_patches = (W + patch_pad[0] + patch_pad[1] - window_size_W) // stride + 1
    W_residue = (W + patch_pad[0] + patch_pad[1] - window_size_W) % stride

    print(H_patches, W_patches, H_residue, W_residue)
    
    # Step 1: separate H_patches and W_patches
    patches = patches.view(N, H_patches, W_patches, C, window_size_H, window_size_W)
    # Step 2: remove overlaps and patch padding
    patches_rec = patches[:, ::step, ::step, :, patch_pad[2]:window_size_H-patch_pad[3], patch_pad[0]:window_size_W-patch_pad[1]]
    # Step 3: reshape to original image size (but we might not be able to fully reconstruct the original if window_size doesn't divide the image size)
    img = patches_rec.permute(0, 3, 1, 4, 2, 5).contiguous().view(N, C, H - H_residue, W - W_residue)

    did_W_recon = False
    # Step 4: Due to patch_pad, we may be able to reconstruct some of the image that was lost due to stride
    #         E.g. image size = 30, window_size = stride = 20, patch_pad = (0,100,0,0). Patch pad lets us capture the would-be missing 10 pixels
    if (patch_pad[1] > 0) and W_residue != 0:
        W_saved = min(patch_pad[1], W_residue)
        W_recon = patches[:, :, -1, :, patch_pad[2]:window_size_H-patch_pad[3], patch_pad[0]+window_size:patch_pad[0]+window_size+W_saved].permute(0, 2, 1, 3, 4).contiguous().view(N, C, H - H_residue, W_saved)
        img = torch.cat((img, W_recon), dim=3)
        did_W_recon = True

    if (patch_pad[3] > 0) and H_residue != 0:
        H_saved = min(patch_pad[3], H_residue)
        H_recon = patches[:, -1, :, :, patch_pad[2]+window_size:patch_pad[2]+window_size+H_saved, patch_pad[0]:window_size_W-patch_pad[1]].permute(0, 2, 1, 3, 4).contiguous().view(N, C, H_saved, W - W_residue)
        
        # We need to append the corner patch to the bottom right corner if we reconstructed the right side
        if did_W_recon:
            corner_patch = patches[:, -1, -1, :, patch_pad[2]+window_size:patch_pad[2]+window_size+H_saved, patch_pad[0]+window_size:patch_pad[0]+window_size+W_saved].view(N, C, H_saved, W_saved)
            H_recon = torch.cat((H_recon, corner_patch), dim=3)
        img = torch.cat((img, H_recon), dim=2)

    # Step 5: remove image padding
    img = img[:, :, image_pad[2]:H - image_pad[3], image_pad[0]:W - image_pad[1]]
    return img


def get_batch_diag_jacobian_vmap_jacrev_windowed(x, model, window_size=32, pad=None, batch_size=16, window_batch_size=16**2, flatten=False, target_output=None):
    '''
    Estimates the batch Jacobian, assuming x is an image and the model's receptive field is smaller than pad. 
    Instead of calculating the Jacobian for the whole image, this calculates the diagonal of the Jacobian for small patches of the image.

    This is MUCH faster than any other method, and has less memory requirements. However, it is only an estimate of the Jacobian. One obvious
    drawback is that some models have very large receptive fields, e.g. transformers, or they use image-wide pooling, e.g. global average pooling.

    Parameters:
    x
        Images to calculate the Jacobian of, of shape (N, C, H, W).
    model
        Model to calculate the Jacobian of.
    window_size
        Size of the window to use.
    pad
        Patches will leak to one another by pad amount. This is useful to fix issues with the patch boundaries when calculating the Jacobian.
        Pad is a 4-tuple of pads for: (left, right, top, bottom). Or a single int, in which case, the same pad is applied to all 4 sides.
        With pad, there will certainly be some issues with the boundary of the whole image, but you can accurately calculate the Jacobian for every other pixel.
    batch_size
        Tells pytorch to process this many batches at a time.
    window_batch_size
        After windowing the image, we end up with a lot of patches. Tells pytorch to process this many patches at a time.
    flatten
        Whether to flatten the Jacobian into a (N, num_fy, num_fx) tensor. If False, the Jacobian will be (N, *yshape, *xshape).
    target_output
        If the model has multiple outputs, calculate the Jacobian for this output. If None, the model must have a single output.
    '''
    from functools import partial
    assert pad is None or isinstance(pad, int) or len(pad) == 4, "pad must be int, or 4-tuple"
    pad = 0 if pad is None else pad
    pad = (pad, pad, pad, pad) if isinstance(pad, int) else pad
    window_size_H = window_size + pad[2] + pad[3]
    window_size_W = window_size + pad[0] + pad[1]

    patches = window_image(x, window_size, patch_pad=pad)
    num_patches = patches.shape[1]
    N, C, H, W = x.shape
    num_fx = np.prod(x.shape[1:])
    w_num_fx = C * window_size_H * window_size_W

    def early_diag(x, model):
        w_jacobian = torch.func.jacrev(model)(x)
        if target_output is not None:
            w_jacobian = w_jacobian[target_output]
        w_jacobian = w_jacobian.view(-1, w_num_fx, w_num_fx)
        return torch.diagonal(w_jacobian, dim1=-2, dim2=-1)     # (N, w_num_fx), i.e. flattened and target_output'd
    
    partial_early_diag = partial(early_diag, model=model)

    w_jacobian_diag = torch.vmap(torch.func.vmap(partial_early_diag, chunk_size=window_batch_size), chunk_size=batch_size)(patches)
    # This won't work if num_fy != num_fx
    w_jacobian_diag = w_jacobian_diag.view(N, num_patches, C, window_size_H, window_size_W)
    unw_jacobian_diag = unwindow_image(w_jacobian_diag, window_size, H, W, patch_pad=pad)
    if flatten:
        unw_jacobian_diag = unw_jacobian_diag.view(N, num_fx)
    return unw_jacobian_diag



def get_full_jacobian_parallel(x, model, chunk_size=16**2, flatten=False, target_output=None):
    ''' This is the parallel way to compute the Jacobian. Differentiates one output w.r.t one input. '''
    N = x.shape[0]
    y = model(x)
    if target_output is not None:
        y = y[target_output]
    xshape1 = x.shape[1:]
    yshape1 = y.shape[1:]
    num_fx = np.prod(xshape1)
    num_fy = np.prod(yshape1)
    y = y.view(y.shape[0], -1)

    num_steps = int(np.ceil(num_fy / chunk_size))
    jacobian = torch.zeros((N, num_fy, num_fx), device=x.device)
    for i in range(num_steps):
        start = i * chunk_size
        end   = min(num_fy, (i + 1) * chunk_size)
        row_idx = torch.arange(end - start)
        diag_idx = torch.arange(start, end)
        
        F = torch.zeros(end - start, num_fy, device=x.device)
        F[row_idx, diag_idx] = 1
        F = F.unsqueeze(1).repeat(1, N, 1)
        
        jpartial = AG.grad(y, x, F, create_graph=True, is_grads_batched=True)[0]
        jpartial = jpartial.view(end - start, N, num_fx)
        jacobian[:, start:end, :] = jpartial.transpose(0, 1)

    if not flatten:
        jacobian = jacobian.view(N, *yshape1, *xshape1)
    return jacobian

def get_diag_jacobian_parallel(x, model, chunk_size=16**2, flatten=False, target_output=None):
    ''' This is the parallel way to compute the diagonal of a Jacobian. Differentiates one output w.r.t one input. '''
    N = x.shape[0]
    y = model(x)
    if target_output is not None:
        y = y[target_output]
    xshape1 = x.shape[1:]
    yshape1 = y.shape[1:]
    num_fx = np.prod(xshape1)
    num_fy = np.prod(yshape1)
    y = y.view(y.shape[0], -1)
    min_f = min(num_fx, num_fy)
    min_shape1 = min(xshape1, yshape1)

    num_steps = int(np.ceil(min_f / chunk_size))
    jacobian = torch.zeros((N, min_f), device=x.device)
    for i in range(num_steps):
        start = i * chunk_size
        end   = min(min_f, (i + 1) * chunk_size)
        row_idx = torch.arange(end - start)
        diag_idx = torch.arange(start, end)
        
        F = torch.zeros(end - start, num_fy, device=x.device)
        F[row_idx, diag_idx] = 1
        F = F.unsqueeze(1).repeat(1, N, 1)
        
        jpartial = AG.grad(y, x, F, create_graph=True, is_grads_batched=True)[0]
        jpartial = jpartial.view(end - start, N, num_fx)
        jacobian[:, start:end] = jpartial[row_idx, :, diag_idx].transpose(0, 1)

    if not flatten:
        jacobian = jacobian.view(N, *min_shape1)
    return jacobian



def get_full_jacobian(x, model, flatten=False, target_output=None):
    ''' This is the simplest way to compute the Jacobian, memory efficient but very slow. Differentiates one output w.r.t one input. '''
    N = x.shape[0]
    y = model(x)
    if target_output is not None:
        y = y[target_output]
    xshape1 = x.shape[1:]
    yshape1 = y.shape[1:]
    num_fx = np.prod(xshape1)
    num_fy = np.prod(yshape1)
    y = y.view(y.shape[0], -1)

    # Jacobian is an N x Fy x Fx tensor, row j is dy_j / dx_i
    jacobian = torch.zeros((N, num_fy, num_fx), device=x.device)
    for f in range(num_fy):
        grd = torch.zeros_like(y)
        grd[:, f] = 1
        # Row-by-row computation of the Jacobian
        grads, = AG.grad(y, x, grad_outputs=grd, create_graph=True)
        jacobian[:, f, :] = grads.view(N, num_fx)

    if not flatten:
        jacobian = jacobian.view(jacobian.shape[0], *yshape1, *xshape1)
    return jacobian

def get_diag_jacobian(x, model, flatten=False, target_output=None):
    ''' This is the simplest way to compute the diagonal of a Jacobian, memory efficient but very slow. Differentiates one output w.r.t one input. '''
    N = x.shape[0]
    y = model(x)
    if target_output is not None:
        y = y[target_output]
    xshape1 = x.shape[1:]
    yshape1 = y.shape[1:]
    num_fx = np.prod(xshape1)
    num_fy = np.prod(yshape1)
    y = y.view(y.shape[0], -1)
    min_f = min(num_fx, num_fy)
    min_shape1 = min(xshape1, yshape1)

    jacobian = torch.zeros((N, min_f), device=x.device)
    for f in range(min_f):
        grd = torch.zeros_like(y)
        grd[:, f] = 1
        grads, = AG.grad(y, x, grad_outputs=grd, create_graph=True)
        jacobian[:, f] = grads.view(N, num_fx)[:, f]

    if not flatten:
        jacobian = jacobian.view(jacobian.shape[0], *min_shape1)
    return jacobian