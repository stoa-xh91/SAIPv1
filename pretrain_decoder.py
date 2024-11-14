import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from vision_transformer import CrossAttentionBlock

def interpolate_pos_encoding(x, pos_embed, npatch, with_cls_token=True):
        # npatch = x.shape[1] - 1 if with_cls_token else x.shape[1]
        N = pos_embed.shape[1] - 1 if with_cls_token else pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        if with_cls_token:
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
        else:
            patch_pos_embed = pos_embed
        dim = x.shape[-1]
        #  h0 = h // self.patch_embed.patch_size
        #  w0 = w // self.patch_embed.patch_size
        #  w0, h0 = w0 + 0.1, h0 + 0.1
        L = npatch
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, N, 1, dim).permute(0, 3, 1, 2),
            scale_factor=(L / N, 1),
            mode='bicubic',
        )
        assert L == patch_pos_embed.shape[2]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if with_cls_token:
            patch_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        return patch_pos_embed

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim, num_patches, patch_size, in_chans=3, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, **args):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.rand(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.rand(1, num_patches + 1, decoder_embed_dim), requires_grad=True)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True)

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        pose_embed = interpolate_pos_encoding(x, self.decoder_pos_embed, x.shape[1]-1, True)
        # add pos embed
        x = x + pose_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
class CSMDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim=65536, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **args):
        super().__init__()
       
        self.decoder_norm = nn.LayerNorm(embed_dim)
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(embed_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(embed_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward_head(self, x):
        x = self.decoder_embed(x)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = x[:, 0, :]
        return x
    
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    
class CSRDecoder(MAEDecoder):
    def __init__(self, **args):
        super().__init__(**args)

    def forward(self, vis_x, masked_x, ids_restore):
        # embed tokens
        x = torch.cat([vis_x, masked_x], dim=1)
        x = self.decoder_embed(x)
        vis_x = x[:, :vis_x.shape[1], :] 
        masked_x = x[:, vis_x.shape[1]:, :]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(masked_x.shape[0], ids_restore.shape[1] + 1 - masked_x.shape[1], 1)
        m_x_ = torch.cat([masked_x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        m_x_ = torch.gather(m_x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, masked_x.shape[2]))  # unshuffle
        m_x_ = torch.cat([masked_x[:, :1, :], vis_x[:, :1, :], m_x_], dim=1)  # append visual tokens and masked tokens
        
        pos_embed = interpolate_pos_encoding(m_x_, self.decoder_pos_embed, m_x_.shape[1]-1, with_cls_token=True)
        # add pos embed
        # x = torch.cat([masked_x[:, :1, :], m_x_], dim=1)  # append cls token
        x = m_x_ + pos_embed

        # apply Transformer blocks
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token and vis tokens
        x = x[:, 2:, :]

        return x
    
    
class CSSDecoder(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim =512, mask_encoder_depth=2, mask_decoder_depth=4,
                 decoder_num_heads=4, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, **args):
        
        super().__init__()
        
        self.q_decoder_embed = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.kv_decoder_embed = nn.Linear(embed_dim, hidden_dim, bias=True)

        self.self_atten_blocks = nn.ModuleList([
            Block(hidden_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(mask_encoder_depth)])
        self.cross_atten_block = CrossAttentionBlock(hidden_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)

        self.kv_decoder_norm = norm_layer(hidden_dim)
        self.q_decoder_norm = norm_layer(hidden_dim)
        self.kernel_pred = nn.Linear(hidden_dim, hidden_dim, bias=True)

        mask_head = []
        for i in range(mask_decoder_depth):
            if i == 0:
                mask_head.append(nn.Conv2d(embed_dim, hidden_dim, 1, 1, 0))
            else:
                mask_head.append(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1))
            mask_head.append(nn.GELU())

        self.mask_head = nn.Sequential(*mask_head)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ref_x):

        b, c, h, w = ref_x.shape
        _ref_x = ref_x.view(b, c, -1).permute(0, 2, 1).contiguous()
        kernel_x = self.q_decoder_embed(x)
        # mask kernel
        for blk in self.self_atten_blocks:
            kernel_x = blk(kernel_x)
        # pop cls token
        kernel_x = kernel_x[:, 0, :].unsqueeze(1)
        tgt_x = self.kv_decoder_embed(_ref_x)
        tgt_x = self.kv_decoder_norm(tgt_x)
        
        y, atten = self.cross_atten_block(kernel_x, tgt_x, tgt_x)
        mask_token = self.q_decoder_norm(y)
        mask_kernel = self.kernel_pred(mask_token)

        # # mask head
        mask_x = self.mask_head(ref_x)
        
        mask_x = mask_x.flatten(2)
        
        logits = torch.bmm(mask_kernel, mask_x)

        return logits, atten
    
class AlignDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim=768, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **args):
        super().__init__()
        
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(embed_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(embed_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        self.decoder_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.decoder_norm(self.last_layer(x))
        return x
    
class ViTAlignDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim=768, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **args):
        super().__init__()
        
       
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(embed_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(embed_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        self.decoder_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.decoder_norm(self.last_layer(x))
        return x
    
class MAEDecoder(nn.Module):
    def __init__(self, embed_dim, num_patches, patch_size, in_chans=3, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, **args):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.rand(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.rand(1, num_patches + 1, decoder_embed_dim), requires_grad=True)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True)

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        pose_embed = interpolate_pos_encoding(x, self.decoder_pos_embed, x.shape[1]-1, True)
        # add pos embed
        x = x + pose_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
