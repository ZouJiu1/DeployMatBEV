# https://github.com/weiyangdaren/Fisheye3DOD/blob/main/models/bevdet/ops/bev_pool/bev_pool.py
import torch

from . import bev_pool_ext

class QuickCumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept, ) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None

# https://github.com/weiyangdaren/Fisheye3DOD/blob/main/models/bevdet/ops/bev_pool/bev_pool.py
class QuickCumsumMeanCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_mean_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_mean_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None

class QuickCumsumMean(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[1:] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[0] = interval_starts[0] + 1

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        interval_lengths = interval_lengths.unsqueeze(-1)
        x = x / interval_lengths

        # save kept for backward
        ctx.save_for_backward(kept)
        ctx.other = interval_lengths

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept = ctx.saved_tensors
        interval_lengths = ctx.other
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        gradx /= interval_lengths

        val = gradx[back]

        return val, None, None

def bev_pool_origin(feats, coords, B, D, H, W, mean_pool=False):
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (W * D * B) + coords[:, 1] * (D * B) +
        coords[:, 2] * B + coords[:, 3])
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]
    if mean_pool:
        x = QuickCumsumMeanCuda.apply(feats, coords, ranks, B, D, H, W)
    else:
        x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x

def bev_pool(feats, coords, ranks, B, D, H, W, mean_pool=False):
    if mean_pool:
        x = QuickCumsumMeanCuda.apply(feats.clone(), coords.clone(), ranks.clone(), B, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        out, geom_feats= QuickCumsumMean.apply(feats.clone(), coords.clone(), ranks.clone())
        C = feats.size()[1]
        output = torch.zeros((B, D, H, W, C), dtype=out.dtype, device=out.device)
        output[geom_feats[:, 3], geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1], :] = out
        output = output.permute(0, 4, 1, 2, 3).contiguous()
    else:
        x = QuickCumsumCuda.apply(feats.clone(), coords.clone(), ranks.clone(), B, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        out, geom_feats = QuickCumsum.apply(feats.clone(), coords.clone(), ranks.clone())
        C = feats.size()[1]
        output = torch.zeros((B, D, H, W, C), dtype=torch.float32, device=feats.device)
        output[geom_feats[:, 3], geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1], :] = out
        output = output.permute(0, 4, 1, 2, 3).contiguous()

    error = torch.max(torch.abs(output - x))
    errorMean = torch.mean(torch.abs(output - x))
    www = (output == x).all()
    mask = output != x
    tmp = x[mask]
    tmp2 = output[mask]
    
    return x

if __name__ == "__main__":
    mean_pool = False
    ranks = torch.tensor([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, \
                          6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, \
                          8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], dtype = torch.int64).cuda()
    length = len(ranks)
    feats = torch.ones((length, 2), dtype=torch.float32).cuda()
    notEqual = torch.where( ranks[:-1] != ranks[1:] )[0].tolist() + [length - 1]
    coords = torch.ones((length, 4), dtype=torch.int64).cuda()
    cnt = 0
    for i in range(length):
        if i in notEqual:
            coords[i, :] = torch.tensor([cnt/3, cnt%3, 0, 0], dtype=torch.int64, device='cuda')
            cnt += 1
    B = D = 1
    H = 3
    W = 3

    bev_pool(feats, coords, ranks, B, D, H, W, mean_pool=False)
    bev_pool(feats, coords, ranks, B, D, H, W, mean_pool=True)
