import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
try:
    import torch_kmeans
except ImportError:
    torch_kmeans = None




class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
            return x


class Stem(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_last"),
        )

    def forward(self, x):
        x = self.nn(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            LayerNorm(dim // 4, eps=1e-6, data_format="channels_last"),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4),
            LayerNorm(dim // 4, eps=1e-6, data_format="channels_last"),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.nn(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + self.drop_path(x)
        x = F.relu(x)
        return x


def cluster(x, n_cluster=8):
    x_kmeans = torch_kmeans.KMeans(n_clusters=n_cluster, seed=0, verbose=False)(x)
    cluster_centers = x_kmeans.centers.expand(x.shape[1], -1, -1).to(x.device)
    x_ = x.squeeze(0).unsqueeze(1).expand(-1, cluster_centers.shape[-2], -1)

    dist = torch.exp(-((x_ - cluster_centers).norm(p=2, dim=-1)))

    # dist_prob = torch.log_softmax(dist, dim=-1).unsqueeze(0)
    dist_prob = (dist / torch.sum(dist, dim=-1).unsqueeze(-1)).unsqueeze(0)
    # labels = x_kmeans.labels.squeeze(0).to(x.device)

    return dist_prob


class Tail(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.nn = nn.Sequential(
            LayerNorm(in_dim, eps=1e-6, data_format="channels_last"),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        x = self.nn(x)
        return x


class UNet_predict_angle(nn.Module):
    def __init__(self, C_in, C_out, depths=[3, 4, 6, 3], stem_dims=[256, 256, 512], dims=[256, 256, 256, 512],
                 decode_out_dims=[256, 256, 256, 128], drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.C_in = C_in
        self.dims = dims
        self.class_num = C_out
        self.stem_dims = stem_dims

        self.stems = nn.ModuleList()
        for i in range(len(depths)):
            if i != (len(depths) - 1):
                stem = Stem(C_in, self.stem_dims[i])
            else:
                stem = Stem(dims[-2], dims[-1])
            self.stems.append(stem)

        self.encoders = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            encode_layer = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for
                  j in range(depths[i])],
            )
            self.encoders.append(encode_layer)
            cur += depths[i]

        self.decoders = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths) + 1):
            if i != 0 and i != len(depths):
                decode_layer = nn.Sequential(
                    *[Block(dim=dims[-1], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                      for j in range(depths[i-1])],
                    Tail(dims[-1], decode_out_dims[i]),
                )
                cur += depths[i-1]
            elif i == 0:
                decode_layer = nn.Sequential(Tail(dims[-1], decode_out_dims[i]))
            elif i == len(depths):
                decode_layer = nn.Sequential(
                    *[Block(dim=(decode_out_dims[-1] * 4), drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[-1])],
                    Tail((decode_out_dims[-1] * 4), decode_out_dims[-1]),
                )
                cur += depths[i - 1]
            self.decoders.append(decode_layer)

        self.decoders_cat = nn.ModuleList()
        for i in range(len(depths) - 1):
            cat_layer = nn.Sequential(Tail(decode_out_dims[i], decode_out_dims[-1]))
            self.decoders_cat.append(cat_layer)

        self.last_layer = nn.Sequential(
            LayerNorm(128, eps=1e-6, data_format="channels_last"),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, C_out),
        )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_in, mass=None, faces=None):

        if x_in.shape[-1] != self.C_in:
            raise ValueError("Input channels must be {}".format(self.C_in))

        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False

        else:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        encoder_feat = list()
        stem_0 = self.stems[0](x_in)
        feat_0 = self.encoders[0](stem_0)
        encoder_feat.append(feat_0)
        for i in range(1, len(self.stem_dims)):
            # stem_i = self.stems[i](x_in)
            feat_i = self.encoders[i](encoder_feat[i - 1])
            encoder_feat.append(feat_i)
        stem_i = self.stems[-1](encoder_feat[-1])
        feat_i = self.encoders[-1](stem_i)

        decode_feat = list()
        feat_0 = self.decoders[0](feat_i)
        decode_feat.append(feat_0)
        for i in range(1, (len(self.stem_dims) + 1)):
            decode_input = torch.cat((encoder_feat[-i], decode_feat[i - 1]), dim=-1)
            feat_i = self.decoders[i](decode_input)
            decode_feat.append(feat_i)

        decode_cat_feat = list()
        for i in range(len(self.dims) - 1):
            feat_i = self.decoders_cat[i](decode_feat[i])
            decode_cat_feat.append(feat_i)

        # x = decode_feat[-1]
        x = torch.cat((decode_feat[-1], decode_cat_feat[2], decode_cat_feat[1], decode_cat_feat[0]), dim=-1)
        x_decoder = self.decoders[-1](x)

        x = self.last_layer(x_decoder)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x = x.squeeze(0)

        return x
