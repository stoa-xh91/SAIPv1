from functools import partial

import torch
import torch.nn as nn
import random
from vision_transformer import Block
from pretrain_decoder import CSSDecoder, CSRDecoder, CSMDecoder, interpolate_pos_encoding
from tinyvit import tinyvit_5m

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MaskedVisionTransformer(nn.Module):
    """ Masked Autoregressor with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=(16,16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.rand(1, num_patches + 1, embed_dim), requires_grad=True) 

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio):
        # embed patches
        # h, w = x.size(2), x.size(3)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed[:,1:,:]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore



class SAIPViT(MaskedVisionTransformer):
    """ Masked Autoregressor with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=(16,16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, pretrained='',**args):
        super().__init__(img_size, patch_size, in_chans,
                 embed_dim, depth, num_heads,
                 mlp_ratio, norm_layer)

        # --------------------------------------------------------------------------
        self.out_index = [6]
        self.inter_norm = norm_layer(embed_dim)
        self.anchor_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.anchor_patches = self.anchor_size[0] * self.anchor_size[1]
        self.initialize_weights(pretrained)

    def initialize_weights(self, pretrained=''):
        # initialization
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        if pretrained != '':
            self.init_from_pretrain(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_from_pretrain(self, pretrained=None):
        checkpoint_model = torch.load(pretrained, map_location='cpu')
        if 'model' in checkpoint_model:
            param_dict = checkpoint_model['model']
        elif 'state_dict' in checkpoint_model:
            param_dict = checkpoint_model['state_dict']
        elif 'student' in checkpoint_model: ### for dino
            param_dict = checkpoint_model["student"]
        else:
            param_dict = checkpoint_model
        param_dict = {k.replace("backbone.", ""): v for k, v in param_dict.items()}
        param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
        count=0
        for k, v in param_dict.items():
            if k not in self.state_dict().keys():
                continue
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                print('shape resize from :{}: param_dict{} to self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
                b, l, d = self.state_dict()[k].size()
                pos_emb = v[:,1:,:]
                pos_emb = torch.nn.functional.interpolate(pos_emb.unsqueeze(0), size=(l-1,d),mode='bilinear')[0]
                v = torch.cat([v[:,0,:].unsqueeze(1), pos_emb], dim=1)
                param_dict[k] = v
            try:
                self.state_dict()[k].copy_(v)
                count +=1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))
        msg = self.load_state_dict(param_dict, strict=False)
        print(msg)
        
    def get_mask(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        L = self.anchor_patches
        N = x.shape[0]
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 1 is keep, 0 is remove
        mask = torch.zeros([N, L], device=x.device)
        mask[:, :len_keep] = 1.
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask
    
    def random_masking_with_ref_anchor(self, masked_x, mask_ratio, hw_size):
        
        B, L, D = masked_x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        base_mask = self.get_mask(masked_x, mask_ratio)
        base_mask = base_mask.view(B, 1, self.anchor_size[0], self.anchor_size[1])
        mask = torch.nn.functional.interpolate(base_mask, hw_size)
        mask = mask.flatten(1)
        ids_shuffle = torch.argsort(1-mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(masked_x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, [base_mask, 1-mask], ids_restore
    
    def forward(self, x, mask_ratio, with_anchor_mask=False, get_intermediate_only=False):
        # embed patches
        
        ph, pw = x.size(2)//self.patch_size[0], x.size(3)//self.patch_size[1]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = interpolate_pos_encoding(x, self.pos_embed, x.shape[1])
        x = x + pos_embed[:,1:,:]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.:
            if with_anchor_mask:
                x, mask, ids_restore = self.random_masking_with_ref_anchor(x, mask_ratio, (ph, pw))
            else:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        _outs = []
        
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            if i in self.out_index:
                _outs.append(self.inter_norm(x))
                if get_intermediate_only:
                    return _outs, mask, ids_restore, attn
            
        _outs.append(self.norm(x))
        
        return _outs, mask, ids_restore, attn



class CSLViTWrapper(nn.Module):
    
    def __init__(self, **args):
        super(CSLViTWrapper, self).__init__()
        self.norm_pix_loss = args['norm_pix_loss']
        self.scales = [0.75, 0.875, 1.125, 1.25]#, 1.5]
        self.backbone = SAIPViT(**args)
        self.csr_decoder = CSRDecoder(num_patches=self.backbone.anchor_patches, **args)
        self.csm_decoder = CSMDecoder(**args)
        self.css_decoder = CSSDecoder(**args)
        self.eps=1e-3
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.backbone.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, hw_size):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.backbone.patch_embed.patch_size[0]
        h, w = hw_size
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def forward(self, base_imgs, meta, is_student=True):
        
        bs = base_imgs.shape[0]
        target_s = random.choice(self.scales)
        ref_inputs = meta['ref_img']
        css_mask = meta['gt_mask']
        msc_imgs = meta['region_imgs']
        num_msc_imgs = len(msc_imgs)
        # msc_imgs_pre_half = msc_imgs[:num_msc_imgs//2]
        # msc_imgs_last_half = msc_imgs[num_msc_imgs//2:]

        base_inputs = torch.cat([base_imgs, meta['aug_img']])
        # r_cl_inputs = torch.cat(region_imgs)

        ph, pw = ref_inputs.size(2)//self.backbone.patch_size[0], ref_inputs.size(3)//self.backbone.patch_size[1]
        b_ph, b_pw = base_imgs.shape[2]//self.backbone.patch_size[0], base_imgs.shape[3]//self.backbone.patch_size[1]
        css_inst_mask = torch.nn.functional.interpolate(css_mask[0].unsqueeze(1), (ph, pw))
        css_roi_mask = torch.nn.functional.interpolate(css_mask[1].unsqueeze(1), (ph, pw))
        outputs = {}

        if is_student:

            b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)
            
            # Image level learning
            # 1.1 learning invariant representation under cross scales
            # base_local_feats, _, _, _ = self.backbone(torch.cat(msc_imgs_pre_half), 0.)
            msc_outs = []
            base_local_feats, _, _, _ = self.backbone(torch.cat(msc_imgs[:num_msc_imgs-2]), 0.)
            msc_outs.append(base_local_feats[1][:, 0, :])
            for i, s_img in enumerate(msc_imgs):
                if i < num_msc_imgs-2:
                    continue
                target_s = random.choice(self.scales)
                csi_inputs = torch.nn.functional.interpolate(s_img, scale_factor=target_s)
                
                csi_feats, _, _, _ = self.backbone(csi_inputs, 0.)
                msc_outs.append(csi_feats[1][:, 0, :])

                if i == len(msc_imgs)-2:
                    css_region_feats = csi_feats[0]

            img_level_tokens = torch.cat([b_feats[1][:, 0, :]]+msc_outs)
            csm_outs = self.csm_decoder(img_level_tokens)
            outputs['csm_preds'] = csm_outs

            # Pixel level learning:
            # 2.1 learning to reconstruct masked details at another scale 
            csr_inputs = torch.nn.functional.interpolate(meta['aug_img'], scale_factor=1.25)
            c_ph, c_pw = csr_inputs.shape[2]//self.backbone.patch_size[0], csr_inputs.shape[3]//self.backbone.patch_size[1]

            masked_feats, mask, ids_restore, _ = self.backbone(csr_inputs, meta['mask_ratio'], with_anchor_mask=True, get_intermediate_only=True)
            csr_outs = self.csr_decoder(b_feats[0][:bs], masked_feats[0], ids_restore)

            csr_target = self.patchify(csr_inputs)
            if self.norm_pix_loss:
                mean = csr_target.mean(dim=-1, keepdim=True)
                var = csr_target.var(dim=-1, keepdim=True)
                csr_target = (csr_target - mean) / (var + 1.e-6)**.5

            csr_loss = (csr_outs - csr_target) ** 2
            csr_loss = csr_loss.mean(dim=-1)  # [N, L], mean loss per patch

            csr_loss = (csr_loss * mask[1]).sum() / mask[1].sum()  # mean loss on removed patches
            outputs['csr_preds'] = self.unpatchify(csr_outs, (c_ph, c_pw))
            outputs['csr_loss'] = csr_loss
            
            # 2.2 learning to match instance to reference region under cross scale setting
            # features of reference region come from 'teacher' model 
            r_feats = meta['ref_feats']
            inter_b_feats = b_feats[0]
            _r_feats = r_feats[:, 1:, :].view(bs, ph, pw, -1).permute(0, 3, 1, 2).contiguous()

            css_inst_preds, _ = self.css_decoder(inter_b_feats[:bs], _r_feats)
            css_inst_outs = torch.sigmoid(css_inst_preds.flatten(1))
            css_inst_mask = css_inst_mask.flatten(1)
            
            css_inst_loss = torch.nn.functional.binary_cross_entropy_with_logits(css_inst_preds.reshape(-1), css_inst_mask.reshape(-1), reduction='mean')

            css_roi_preds, _ = self.css_decoder(css_region_feats, _r_feats)
            css_roi_outs = torch.sigmoid(css_roi_preds.flatten(1))
            css_roi_mask = css_roi_mask.flatten(1)
            css_roi_loss = torch.nn.functional.binary_cross_entropy_with_logits(css_roi_preds.reshape(-1), css_roi_mask.reshape(-1), reduction='mean')
            
            outputs['css_inst_loss'] = css_inst_loss
            outputs['css_roi_loss'] = css_roi_loss
            outputs['css_inst_preds'] = css_inst_outs.reshape(-1, ph, pw)
            outputs['css_roi_preds'] = css_roi_outs.reshape(-1, ph, pw)
            outputs['last_atten'] = last_atten[0][:bs, :, 0, 1:].view(bs, -1, b_ph, b_pw)
            outputs['qkv_atten'] =  last_atten

        else:
            
            r_feats, _, _, _ = self.backbone(ref_inputs, 0.)
            
            b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)

            img_level_tokens = b_feats[1][:, 0, :]
            csm_outs = self.csm_decoder(img_level_tokens)

            outputs['ref_feats'] = r_feats[0]
            outputs['csm_preds'] = csm_outs
            outputs['qkv_atten'] =  last_atten
        
        return outputs

class CSRSViTWrapper(nn.Module):
    """
    CSS+CSR
    """
    def __init__(self, **args):
        super(CSRSViTWrapper, self).__init__()
        self.norm_pix_loss = args['norm_pix_loss']
        self.scales = [0.75, 0.875, 1.125, 1.25]#, 1.5]
        self.backbone = SAIPViT(**args)
        self.csr_decoder = CSRDecoder(num_patches=self.backbone.anchor_patches, **args)
        self.dino_decoder = CSMDecoder(**args)
        self.eps=1e-3
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.backbone.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, hw_size):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.backbone.patch_embed.patch_size[0]
        h, w = hw_size
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def forward(self, base_imgs, meta, is_student=True):
        
        bs = base_imgs.shape[0]
        target_s = random.choice(self.scales)
        msc_imgs = meta['region_imgs']
        num_msc_imgs = len(msc_imgs)

        base_inputs = torch.cat([base_imgs, meta['aug_img']])

        b_ph, b_pw = base_imgs.shape[2]//self.backbone.patch_size[0], base_imgs.shape[3]//self.backbone.patch_size[1]
        
        outputs = {}

        if is_student:

            b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)
            
            # Image level learning
            # 1.1 learning invariant representation under cross scales
            msc_outs = []
            base_local_feats, _, _, _ = self.backbone(torch.cat(msc_imgs[:num_msc_imgs-2]), 0.)
            msc_outs.append(base_local_feats[1][:, 0, :])
            for i, s_img in enumerate(msc_imgs):
                if i < num_msc_imgs-2:
                    continue
                target_s = random.choice(self.scales)
                csi_inputs = torch.nn.functional.interpolate(s_img, scale_factor=target_s)
                
                csi_feats, _, _, _ = self.backbone(csi_inputs, 0.)
                msc_outs.append(csi_feats[1][:, 0, :])

            img_level_tokens = torch.cat([b_feats[1][:, 0, :]]+msc_outs)
            dino_outs = self.dino_decoder(img_level_tokens)
            outputs['dino_preds'] = dino_outs

            # Pixel level learning:
            # 2.1 learning to reconstruct masked details at another scale 
            csr_inputs = torch.nn.functional.interpolate(meta['aug_img'], scale_factor=1.25)
            c_ph, c_pw = csr_inputs.shape[2]//self.backbone.patch_size[0], csr_inputs.shape[3]//self.backbone.patch_size[1]

            masked_feats, mask, ids_restore, _ = self.backbone(csr_inputs, meta['mask_ratio'], with_anchor_mask=True, get_intermediate_only=True)
            csr_outs = self.csr_decoder(b_feats[0][:bs], masked_feats[0], ids_restore)

            csr_target = self.patchify(csr_inputs)
            if self.norm_pix_loss:
                mean = csr_target.mean(dim=-1, keepdim=True)
                var = csr_target.var(dim=-1, keepdim=True)
                csr_target = (csr_target - mean) / (var + 1.e-6)**.5

            csr_loss = (csr_outs - csr_target) ** 2
            csr_loss = csr_loss.mean(dim=-1)  # [N, L], mean loss per patch

            csr_loss = (csr_loss * mask[1]).sum() / mask[1].sum()  # mean loss on removed patches
            outputs['csr_preds'] = self.unpatchify(csr_outs, (c_ph, c_pw))
            outputs['csr_loss'] = csr_loss
            
            
            outputs['last_atten'] = last_atten[0][:bs, :, 0, 1:].view(bs, -1, b_ph, b_pw)
            outputs['qkv_atten'] =  last_atten

        else:
            
            
            b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)


            img_level_tokens = b_feats[1][:, 0, :] 
            dino_outs = self.dino_decoder(img_level_tokens)

            outputs['dino_preds'] = dino_outs
            outputs['qkv_atten'] =  last_atten
        
        return outputs
    


class CSLTinyViTWrapper(nn.Module):
   
    def __init__(self, **args):
        super(CSLTinyViTWrapper, self).__init__()
        self.norm_pix_loss = args['norm_pix_loss']
        self.scales = [0.75, 0.875, 1, 1.125]#, 1.5]
        self.patch_size = args['patch_size']
        self.backbone = tinyvit_5m(**args)
        self.dino_decoder = DINODecoder(embed_dim=self.backbone.final_dims, **args)
        self.csm_decoder = CSMDecoder(embed_dim=self.backbone.stage_dims[-1], **args)
        self.eps=1e-3
    
    def forward(self, base_imgs, meta, is_student=True):
        
        bs, _, bh, bw = base_imgs.shape
        target_s = random.choice(self.scales)
        ref_inputs = meta['ref_img']
        csm_mask = meta['gt_mask']
        msc_imgs = meta['region_imgs']
        num_msc_imgs = len(msc_imgs)

        base_inputs = torch.cat([base_imgs, meta['aug_img']])

        ph, pw = ref_inputs.size(2)//self.patch_size[0], ref_inputs.size(3)//self.patch_size[1]
        
        csm_inst_mask = torch.nn.functional.interpolate(csm_mask[0].unsqueeze(1), (ph, pw))
        csm_roi_mask = torch.nn.functional.interpolate(csm_mask[1].unsqueeze(1), (ph, pw))
        outputs = {}

        if is_student:

            b_cls_token, b_feats = self.backbone(base_inputs)
            
            # Image level learning
            # 1.1 learning invariant representation under cross scales
            msc_outs = [b_cls_token]
            ms_cls_token, _ = self.backbone(torch.cat(msc_imgs[:num_msc_imgs-2]))
            msc_outs.append(ms_cls_token)
            for i, s_img in enumerate(msc_imgs):
                if i < num_msc_imgs-2:
                    continue
                target_s = random.choice(self.scales)
                # target_h, target_w = int(bh * target_s), int(bw * target_s)
                # csi_inputs = torch.nn.functional.interpolate(s_img, (target_h, target_w))
                csi_inputs = torch.nn.functional.interpolate(s_img, scale_factor=target_s)
                
                csi_cls_token, csi_feats = self.backbone(csi_inputs)
                msc_outs.append(csi_cls_token)

                if i == len(msc_imgs)-2:
                    csm_region_feats = csi_feats[-2]

            img_level_tokens = torch.cat(msc_outs)
            dino_outs = self.dino_decoder(img_level_tokens)
            outputs['dino_preds'] = dino_outs

            # Pixel level learning:
            
            # learning to match instance to reference region under cross scale setting
            # features of reference region come from 'teacher' model 
            r_feats = meta['ref_feats']
            inter_b_feats = b_feats[-2][:bs]
            _r_feats = r_feats.view(bs, ph, pw, -1).permute(0, 3, 1, 2).contiguous()

            csm_inst_preds, _ = self.csm_decoder(inter_b_feats, _r_feats)
            csm_inst_outs = torch.sigmoid(csm_inst_preds.flatten(1))
            csm_inst_mask = csm_inst_mask.flatten(1)
            
            csm_inst_loss = torch.nn.functional.binary_cross_entropy_with_logits(csm_inst_preds.reshape(-1), csm_inst_mask.reshape(-1), reduction='mean')

            csm_roi_preds, _ = self.csm_decoder(csm_region_feats, _r_feats)
            csm_roi_outs = torch.sigmoid(csm_roi_preds.flatten(1))
            csm_roi_mask = csm_roi_mask.flatten(1)
            csm_roi_loss = torch.nn.functional.binary_cross_entropy_with_logits(csm_roi_preds.reshape(-1), csm_roi_mask.reshape(-1), reduction='mean')
            
            outputs['css_inst_loss'] = csm_inst_loss
            outputs['css_roi_loss'] = csm_roi_loss
            outputs['css_inst_preds'] = csm_inst_outs.reshape(-1, ph, pw)
            outputs['css_roi_preds'] = csm_roi_outs.reshape(-1, ph, pw)

        else:
            
            _, r_feats = self.backbone(ref_inputs)
            
            b_cls_token, b_feats = self.backbone(base_inputs)

            dino_outs = self.dino_decoder(b_cls_token)

            outputs['ref_feats'] = r_feats[-2]
            outputs['dino_preds'] = dino_outs
        
        return outputs


    
def csl_vit_tiny_patch16(**kwargs):
    model = CSLViTWrapper(
        patch_size=(16, 16), embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def csrs_vit_tiny_patch16(**kwargs):
    model = CSRSViTWrapper(
        patch_size=(16, 16), embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def csl_tinyvit_5m_patch16(**kwargs):
    model = CSLTinyViTWrapper(
        patch_size=(16, 16),
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



