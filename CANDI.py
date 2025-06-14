from model import *
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import tracemalloc, sys, argparse

# tracemalloc.start()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CANDI_Encoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3):
        super(CANDI_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model =  self.latent_dim = self.f2

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1,
                pool_size=pool_size) for i in range(n_cnn_layers)])

        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=True)
        self.xmd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            nn.ReLU())

        if self.pos_enc == "relative":
            self.transformer_encoder = nn.ModuleList([
                RelativeEncoderLayer(d_model=self.d_model, heads=nhead, feed_forward_hidden=expansion_factor*self.d_model, dropout=dropout) for _ in range(n_sab_layers)])
            
        else:
            self.posEnc = PositionalEncoding(self.d_model, dropout, self.l2)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=nhead, dim_feedforward=expansion_factor*self.d_model, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_sab_layers)

    def forward(self, src, x_metadata):
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)

        src = src.permute(0, 2, 1)  # to N, L', F2
        xmd_embedding = self.xmd_emb(x_metadata)
        src = torch.cat([src, xmd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.xmd_fusion(src)

        ### TRANSFORMER ENCODER ###
        if self.pos_enc != "relative":
            src = self.posEnc(src)
            src = self.transformer_encoder(src)
        else:
            for enc in self.transformer_encoder:
                src = enc(src)
        
        return src

class CANDI_Decoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size=2, expansion_factor=3):
        super(CANDI_Decoder, self).__init__()

        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model =  self.latent_dim = self.f2

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.ymd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=False)
        self.ymd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            )

        self.deconv = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])
    
    def forward(self, src, y_metadata):
        ymd_embedding = self.ymd_emb(y_metadata)
        src = torch.cat([src, ymd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.ymd_fusion(src)
        
        src = src.permute(0, 2, 1) # to N, F2, L'
        for dconv in self.deconv:
            src = dconv(src)

        src = src.permute(0, 2, 1) # to N, L, F1

        return src    

class CANDI(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", 
        expansion_factor=3, separate_decoders=True):
        super(CANDI, self).__init__()

        self.pos_enc = pos_enc
        self.separate_decoders = separate_decoders
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model = self.latent_dim = self.f2
        print("d_model: ", self.d_model)

        self.encoder = CANDI_Encoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size, dropout, context_length, pos_enc, expansion_factor)
        
        if self.separate_decoders:
            self.count_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)
            self.pval_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)
        else:
            self.decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)

        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1)
    
    def encode(self, src, x_metadata):
        """Encode input data into latent representation."""
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        
        z = self.encoder(src, x_metadata)
        return z
    
    def decode(self, z, y_metadata):
        """Decode latent representation into predictions."""
        y_metadata = torch.where(y_metadata == -2, torch.tensor(-1, device=y_metadata.device), y_metadata)
        
        if self.separate_decoders:
            count_decoded = self.count_decoder(z, y_metadata)
            pval_decoded = self.pval_decoder(z, y_metadata)

            p, n = self.neg_binom_layer(count_decoded)
            mu, var = self.gaussian_layer(pval_decoded)
        else:
            decoded = self.decoder(z, y_metadata)
            p, n = self.neg_binom_layer(decoded)
            mu, var = self.gaussian_layer(decoded)
            
        return p, n, mu, var

    def forward(self, src, x_metadata, y_metadata, availability=None, return_z=False):
        z = self.encode(src, x_metadata)
        p, n, mu, var = self.decode(z, y_metadata)
        
        if return_z:
            return p, n, mu, var, z
        else:
            return p, n, mu, var

class CANDI_DNA_Encoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3):
        super(CANDI_DNA_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2
        self.latent_dim = self.f2

        DNA_conv_channels = exponential_linspace_int(4, self.f2, n_cnn_layers+3)
        DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers+2)]

        self.convEncDNA = nn.ModuleList(
            [ConvTower(
                DNA_conv_channels[i], DNA_conv_channels[i + 1],
                DNA_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True, SE=False,
                groups=1, pool_size=5 if i >= n_cnn_layers else pool_size) for i in range(n_cnn_layers + 2)])

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size_list = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size_list[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1, SE=False,
                pool_size=pool_size) for i in range(n_cnn_layers)])
        
        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=False)

        self.fusion = nn.Sequential(
            # nn.Linear((2*self.f2), self.f2), 
            nn.Linear((2*self.f2)+metadata_embedding_dim, self.f2), 
            # nn.Linear((self.f2)+metadata_embedding_dim, self.f2), 
            nn.LayerNorm(self.f2), 

            )

        self.transformer_encoder = nn.ModuleList([
            DualAttentionEncoderBlock(self.f2, nhead, self.l2, dropout=dropout, 
                max_distance=self.l2, pos_encoding_type="relative", max_len=self.l2
                ) for _ in range(n_sab_layers)])

    def forward(self, src, seq, x_metadata):
        if len(seq.shape) != len(src.shape):
            seq = seq.unsqueeze(0).expand(src.shape[0], -1, -1)

        seq = seq.permute(0, 2, 1)  # to N, 4, 25*L
        seq = seq.float()

        ### DNA CONV ENCODER ###
        for seq_conv in self.convEncDNA:
            seq = seq_conv(seq)
        seq = seq.permute(0, 2, 1)  # to N, L', F2

        ### SIGNAL CONV ENCODER ###
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)
        src = src.permute(0, 2, 1)  # to N, L', F2

        ### SIGNAL METADATA EMBEDDING ###
        xmd_embedding = self.xmd_emb(x_metadata).unsqueeze(1).expand(-1, self.l2, -1)

        ### FUSION ###
        src = torch.cat([src, xmd_embedding, seq], dim=-1)
        # src = torch.cat([src, seq], dim=-1)
        # src = torch.cat([seq, xmd_embedding], dim=-1)
        src = self.fusion(src)

        ### TRANSFORMER ENCODER ###
        for enc in self.transformer_encoder:
            src = enc(src)

        return src

class CANDI_DNA(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", 
        expansion_factor=3, separate_decoders=True):
        super(CANDI_DNA, self).__init__()

        self.pos_enc = pos_enc
        self.separate_decoders = separate_decoders
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model = self.latent_dim = self.f2

        self.encoder = CANDI_DNA_Encoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size, dropout, context_length, pos_enc, expansion_factor)
        
        if self.separate_decoders:
            self.count_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)
            self.pval_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)
        else:
            self.decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)

        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1)
    
    def encode(self, src, seq, x_metadata):
        """Encode input data into latent representation."""
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        
        z = self.encoder(src, seq, x_metadata)
        return z
    
    def decode(self, z, y_metadata):
        """Decode latent representation into predictions."""
        y_metadata = torch.where(y_metadata == -2, torch.tensor(-1, device=y_metadata.device), y_metadata)
        
        if self.separate_decoders:
            count_decoded = self.count_decoder(z, y_metadata)
            pval_decoded = self.pval_decoder(z, y_metadata)

            p, n = self.neg_binom_layer(count_decoded)
            mu, var = self.gaussian_layer(pval_decoded)
        else:
            decoded = self.decoder(z, y_metadata)
            p, n = self.neg_binom_layer(decoded)
            mu, var = self.gaussian_layer(decoded)
            
        return p, n, mu, var

    def forward(self, src, seq, x_metadata, y_metadata, availability=None, return_z=False):
        z = self.encode(src, seq, x_metadata)
        p, n, mu, var = self.decode(z, y_metadata)
        
        if return_z:
            return p, n, mu, var, z
        else:
            return p, n, mu, var

class CANDI_UNET(CANDI_DNA):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers,
                 nhead, n_sab_layers, pool_size=2, dropout=0.1, context_length=1600,
                 pos_enc="relative", expansion_factor=3, separate_decoders=True):
        super(CANDI_UNET, self).__init__(signal_dim, metadata_embedding_dim,
                                          conv_kernel_size, n_cnn_layers,
                                          nhead, n_sab_layers,
                                          pool_size, dropout,
                                          context_length, pos_enc,
                                          expansion_factor,
                                          separate_decoders)

    def _compute_skips(self, src):
        # mask as in encode
        src = torch.where(src == -2,
                          torch.tensor(-1, device=src.device), src)
        x = src.permute(0, 2, 1)  # (N, F1, L)
        skips = []
        for conv in self.encoder.convEnc:
            x = conv(x)
            skips.append(x)
        return skips

    def _unet_decode(self, z, y_metadata, skips, decoder):
        # mask metadata
        y_metadata = torch.where(y_metadata == -2,
                                 torch.tensor(-1, device=y_metadata.device),
                                 y_metadata)
        # embed and fuse metadata
        ymd_emb = decoder.ymd_emb(y_metadata)
        x = torch.cat([z, ymd_emb.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        x = decoder.ymd_fusion(x)
        x = x.permute(0, 2, 1)  # (N, C, L)

        # apply deconvs with UNet additions
        for i, dconv in enumerate(decoder.deconv):
            skip = skips[-(i + 1)]  # matching resolution
            x = x + skip
            x = dconv(x)

        x = x.permute(0, 2, 1)  # (N, L, F1)
        return x

    def forward(self, src, seq, x_metadata, y_metadata, availability=None, return_z=False):
        # compute skip features from signal branch
        skips = self._compute_skips(src)
        # standard encode (fuses seq + signal + metadata)
        z = self.encode(src, seq, x_metadata)

        # UNet-style decode for counts
        if self.separate_decoders:
            count_decoded = self._unet_decode(z, y_metadata, skips, self.count_decoder)
        else:
            count_decoded = self._unet_decode(z, y_metadata, skips, self.decoder)
        # Negative binomial parameters
        p, n = self.neg_binom_layer(count_decoded)

        # UNet-style decode for p-values
        if self.separate_decoders:
            pval_decoded = self._unet_decode(z, y_metadata, skips, self.pval_decoder)  
        else:
            pval_decoded = self._unet_decode(z, y_metadata, skips, self.decoder)  
        # Gaussian parameters
        mu, var = self.gaussian_layer(pval_decoded)

        if return_z:
            return p, n, mu, var, z
            
        return p, n, mu, var

class CANDI_LOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(CANDI_LOSS, self).__init__()
        self.reduction = reduction
        self.gaus_nll = nn.GaussianNLLLoss(reduction=self.reduction, full=True)
        self.nbin_nll = negative_binomial_loss

    def forward(self, p_pred, n_pred, mu_pred, var_pred, true_count, true_pval, obs_map, masked_map):
        ups_true_count, ups_true_pval = true_count[obs_map], true_pval[obs_map]
        ups_n_pred, ups_p_pred = n_pred[obs_map], p_pred[obs_map]
        ups_mu_pred, ups_var_pred = mu_pred[obs_map], var_pred[obs_map]

        imp_true_count, imp_true_pval = true_count[masked_map], true_pval[masked_map]
        imp_n_pred, imp_p_pred = n_pred[masked_map], p_pred[masked_map]
        imp_mu_pred, imp_var_pred = mu_pred[masked_map], var_pred[masked_map]

        observed_count_loss = self.nbin_nll(ups_true_count, ups_n_pred, ups_p_pred) 
        imputed_count_loss = self.nbin_nll(imp_true_count, imp_n_pred, imp_p_pred)

        if self.reduction == "mean":
            observed_count_loss = observed_count_loss.mean()
            imputed_count_loss = imputed_count_loss.mean()
        elif self.reduction == "sum":
            observed_count_loss = observed_count_loss.sum()
            imputed_count_loss = imputed_count_loss.sum()

        observed_pval_loss = self.gaus_nll(ups_mu_pred, ups_true_pval, ups_var_pred)
        imputed_pval_loss = self.gaus_nll(imp_mu_pred, imp_true_pval, imp_var_pred)

        observed_pval_loss = observed_pval_loss.float()
        imputed_pval_loss = imputed_pval_loss.float()
        
        return observed_count_loss, imputed_count_loss, observed_pval_loss, imputed_pval_loss

class PRETRAIN(object):
    def __init__(
        self, model, dataset, criterion, optimizer, 
        scheduler, device=None, HPO=False, cosine_sched=False):
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Training on device: {self.device}.")

        self.model = model.to(self.device)
        self.dataset = dataset
        self.HPO = HPO
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cosine_sched = cosine_sched

    def pretrain_CANDI(
        self, num_epochs, context_length, batch_size, inner_epochs, 
        arch="", mask_percentage=0.15, hook=False, DNA=False, 
        early_stop=True, early_stop_metric="imp_pval_r2", 
        early_stop_delta=0.01, patience=2, prog_monitor_patience=150, 
        prog_monitor_delta=1e-5):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"CANDI{arch} # model_parameters: {count_parameters(self.model)}")
        logfile = open(f"models/CANDI{arch}_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        images = []
        gif_filename = f"models/CANDI{arch}_TrainProg.gif"

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        num_assays = self.dataset.signal_dim
        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        if hook:
            register_hooks(self.model)
        
        val_eval = MONITOR_VALIDATION(
            self.dataset.base_path, context_length, 4*batch_size, 
            must_have_chr_access=self.dataset.must_have_chr_access,
            token_dict=token_dict, eic=bool("eic" in arch), 
            DNA=DNA, device=self.device)

        try:
            validation_set_eval, val_metrics = val_eval.get_validation(self.model)
            torch.cuda.empty_cache()
            log_strs.append(validation_set_eval)
            print(validation_set_eval)
            log_resource_usage()
        except:
            pass

        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)
        best_metric = None

        progress_monitor = {
            "ups_count_r2":[], "imp_count_r2":[],
            "ups_pval_r2":[], "imp_pval_r2":[],
            "ups_count_spearman":[], "imp_count_spearman":[],
            "ups_pval_spearman":[], "imp_pval_spearman":[],
            "ups_count_pearson":[], "imp_count_pearson":[],
            "ups_pval_pearson":[], "imp_pval_pearson":[]}
        
        prog_mon_ema = {}
        prog_mon_best_so_far = {}
        no_prog_mon_improvement = 0
        lr_sch_steps_taken = 0

        for epoch in range(num_epochs):
            if early_stop:
                epoch_rec = {
                    "ups_count_r2":[], "imp_count_r2":[],
                    "ups_pval_r2":[], "imp_pval_r2":[],
                    "ups_count_spearman":[], "imp_count_spearman":[],
                    "ups_pval_spearman":[], "imp_pval_spearman":[],
                    "ups_count_pearson":[], "imp_count_pearson":[],
                    "ups_pval_pearson":[], "imp_pval_pearson":[],

                    "val_count_median_ups_r2":[], "val_count_median_imp_r2":[], 
                    "val_count_median_ups_pcc":[], "val_count_median_imp_pcc":[], 
                    "val_count_median_ups_srcc":[], "val_count_median_imp_srcc":[], 
                    "val_count_median_ups_loss":[], "val_count_median_imp_loss":[], 
                    
                    "val_pval_median_ups_pcc":[],  "val_pval_median_imp_pcc":[], 
                    "val_pval_median_ups_r2":[],   "val_pval_median_imp_r2":[], 
                    "val_pval_median_ups_srcc":[], "val_pval_median_imp_srcc":[],
                    "val_pval_median_ups_loss":[], "val_pval_median_imp_loss":[]

                    }

            self.dataset.new_epoch()
            next_epoch = False

            last_lopr = -1
            while (next_epoch==False):
                t0 = datetime.now()

                if DNA:
                    _X_batch, _mX_batch, _avX_batch, _dnaseq_batch= self.dataset.get_batch(side="x", dna_seq=True)
                else:
                    _X_batch, _mX_batch, _avX_batch = self.dataset.get_batch(side="x")

                _Y_batch, _mY_batch, _avY_batch, _pval_batch = self.dataset.get_batch(side="y", pval=True)

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    self.dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                
                batch_rec = {
                    "imp_count_loss":[], "ups_count_loss":[],
                    "imp_pval_loss":[], "ups_pval_loss":[],
                    "ups_count_r2":[], "imp_count_r2":[],
                    "ups_pval_r2":[], "imp_pval_r2":[],
                    "ups_count_pp":[], "imp_count_pp":[],
                    "ups_pval_pp":[], "imp_pval_pp":[],
                    "ups_count_conf":[], "imp_count_conf":[],
                    "ups_pval_conf":[], "imp_pval_conf":[],
                    "ups_count_mse":[], "imp_count_mse":[],
                    "ups_pval_mse":[], "imp_pval_mse":[],
                    "ups_count_spearman":[], "imp_count_spearman":[],
                    "ups_pval_spearman":[], "imp_pval_spearman":[],
                    "ups_count_pearson":[], "imp_count_pearson":[],
                    "ups_pval_pearson":[], "imp_pval_pearson":[], 
                    "grad_norm":[]
                    }

                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                for _ in range(inner_epochs):
                    if DNA:
                        X_batch, mX_batch, avX_batch, dnaseq_batch= _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone(), _dnaseq_batch.clone()
                    else:
                        X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                    Y_batch, mY_batch, avY_batch, pval_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone(), _pval_batch.clone()

                    if "_prog_unmask" in arch or "_prog_mask" in arch:
                        X_batch, mX_batch, avX_batch = self.masker.progressive(X_batch, mX_batch, avX_batch, num_mask)

                    if "random_mask" in arch:
                        num_mask = random.randint(1, self.dataset.signal_dim - 1)
                        X_batch, mX_batch, avX_batch = self.masker.progressive(X_batch, mX_batch, avX_batch, num_mask)

                    else:
                        X_batch, mX_batch, avX_batch = self.masker.mask_feature30(X_batch, mX_batch, avX_batch)

                    masked_map = (X_batch == token_dict["cloze_mask"])
                    observed_map = (X_batch != token_dict["missing_mask"]) & (X_batch != token_dict["cloze_mask"])
                    missing_map = (X_batch == token_dict["missing_mask"])
                    masked_map = masked_map.to(self.device) # imputation targets
                    observed_map = observed_map.to(self.device) # upsampling targets
                    
                    X_batch = X_batch.float().to(self.device).requires_grad_(True)
                    mX_batch = mX_batch.to(self.device)
                    avX_batch = avX_batch.to(self.device)
                    mY_batch = mY_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                    pval_batch = pval_batch.to(self.device)                        

                    if DNA:
                        dnaseq_batch = dnaseq_batch.to(self.device)
                        output_p, output_n, output_mu, output_var = self.model(X_batch, dnaseq_batch, mX_batch, mY_batch, avX_batch)
                    else:
                        output_p, output_n, output_mu, output_var = self.model(X_batch, mX_batch, mY_batch, avX_batch)

                    obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss = self.criterion(
                        output_p, output_n, output_mu, output_var, Y_batch, pval_batch, observed_map, masked_map) 
                    
                    # if "_prog_unmask" in arch or "_prog_mask" in arch or "random_mask" in arch:
                    if "random_mask" in arch:
                        msk_p = float(num_mask/num_assays)
                        
                        if "imponly" in arch:
                            loss = (msk_p*(imp_count_loss + imp_pval_loss))
                        
                        elif "pvalonly" in arch:
                            loss = (msk_p*imp_pval_loss) + ((1-msk_p)*obs_pval_loss)

                        else:
                            imp_pval_loss *= 4
                            obs_pval_loss *= 3
                            imp_count_loss *= 2
                            obs_count_loss *= 1
                            loss = (msk_p*(imp_count_loss + imp_pval_loss)) + ((1-msk_p)*(obs_pval_loss + obs_count_loss))

                    else:
                        loss = obs_count_loss + obs_pval_loss + imp_pval_loss + imp_count_loss

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        del X_batch, mX_batch, mY_batch, avX_batch, output_p, output_n, Y_batch, observed_map, loss
                        print(skipmessage)
                        torch.cuda.empty_cache() 
                        continue
                    
                    loss = loss.float()
                    loss.backward()  
                    
                    total_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)

                    #################################################################################

                    # IMP Count Predictions
                    neg_bin_imp = NegativeBinomial(output_p[masked_map].cpu().detach(), output_n[masked_map].cpu().detach())
                    imp_count_pred = neg_bin_imp.expect().numpy()
                    imp_count_std = neg_bin_imp.std().numpy()

                    imp_count_true = Y_batch[masked_map].cpu().detach().numpy()
                    imp_count_abs_error = torch.abs(torch.Tensor(imp_count_true) - torch.Tensor(imp_count_pred)).numpy()

                    imp_count_r2 = r2_score(imp_count_true, imp_count_pred)
                    imp_count_errstd = spearmanr(imp_count_std, imp_count_abs_error)
                    imp_count_pp = compute_perplexity(neg_bin_imp.pmf(imp_count_true))
                    imp_count_mse = ((imp_count_true - imp_count_pred)**2).mean()

                    imp_count_spearman = spearmanr(imp_count_true, imp_count_pred).correlation
                    imp_count_pearson = pearsonr(imp_count_true, imp_count_pred)[0]

                    # IMP P-value Predictions
                    imp_pval_pred = output_mu[masked_map].cpu().detach().numpy()
                    imp_pval_std = output_var[masked_map].cpu().detach().numpy() ** 0.5

                    imp_pval_true = pval_batch[masked_map].cpu().detach().numpy()
                    imp_pval_abs_error = torch.abs(torch.Tensor(imp_pval_true) - torch.Tensor(imp_pval_pred)).numpy()

                    imp_pval_r2 = r2_score(imp_pval_true, imp_pval_pred)
                    imp_pval_errstd = spearmanr(imp_pval_std, imp_pval_abs_error)
                    gaussian_imp = Gaussian(output_mu[masked_map].cpu().detach(), output_var[masked_map].cpu().detach())
                    imp_pval_pp = compute_perplexity(gaussian_imp.pdf(imp_pval_true))
                    imp_pval_mse = ((imp_pval_true - imp_pval_pred)**2).mean()

                    imp_pval_spearman = spearmanr(imp_pval_true, imp_pval_pred).correlation
                    imp_pval_pearson = pearsonr(imp_pval_true, imp_pval_pred)[0]

                    # UPS Count Predictions
                    neg_bin_ups = NegativeBinomial(output_p[observed_map].cpu().detach(), output_n[observed_map].cpu().detach())
                    ups_count_pred = neg_bin_ups.expect().numpy()
                    ups_count_std = neg_bin_ups.std().numpy()

                    ups_count_true = Y_batch[observed_map].cpu().detach().numpy()
                    ups_count_abs_error = torch.abs(torch.Tensor(ups_count_true) - torch.Tensor(ups_count_pred)).numpy()

                    ups_count_r2 = r2_score(ups_count_true, ups_count_pred)
                    ups_count_errstd = spearmanr(ups_count_std, ups_count_abs_error)
                    ups_count_pp = compute_perplexity(neg_bin_ups.pmf(ups_count_true))
                    ups_count_mse = ((ups_count_true - ups_count_pred)**2).mean()

                    ups_count_spearman = spearmanr(ups_count_true, ups_count_pred).correlation
                    ups_count_pearson = pearsonr(ups_count_true, ups_count_pred)[0]

                    # UPS P-value Predictions
                    ups_pval_pred = output_mu[observed_map].cpu().detach().numpy()
                    ups_pval_std = output_var[observed_map].cpu().detach().numpy() ** 0.5

                    ups_pval_true = pval_batch[observed_map].cpu().detach().numpy()
                    ups_pval_abs_error = torch.abs(torch.Tensor(ups_pval_true) - torch.Tensor(ups_pval_pred)).numpy()

                    ups_pval_r2 = r2_score(ups_pval_true, ups_pval_pred)
                    ups_pval_errstd = spearmanr(ups_pval_std, ups_pval_abs_error)
                    gaussian_ups = Gaussian(output_mu[observed_map].cpu().detach(), output_var[observed_map].cpu().detach())
                    ups_pval_pp = compute_perplexity(gaussian_ups.pdf(ups_pval_true))
                    ups_pval_mse = ((ups_pval_true - ups_pval_pred)**2).mean()

                    ups_pval_spearman = spearmanr(ups_pval_true, ups_pval_pred).correlation
                    ups_pval_pearson = pearsonr(ups_pval_true, ups_pval_pred)[0]

                    #################################################################################
                    batch_rec["grad_norm"].append(total_norm)

                    batch_rec["imp_count_loss"].append(imp_count_loss.item())
                    batch_rec["ups_count_loss"].append(obs_count_loss.item())
                    batch_rec["imp_pval_loss"].append(imp_pval_loss.item())
                    batch_rec["ups_pval_loss"].append(obs_pval_loss.item())

                    batch_rec["ups_count_r2"].append(ups_count_r2)
                    batch_rec["imp_count_r2"].append(imp_count_r2)

                    batch_rec["ups_pval_r2"].append(ups_pval_r2)
                    batch_rec["imp_pval_r2"].append(imp_pval_r2)

                    batch_rec["ups_count_pp"].append(ups_count_pp)
                    batch_rec["imp_count_pp"].append(imp_count_pp)

                    batch_rec["ups_pval_pp"].append(ups_pval_pp)
                    batch_rec["imp_pval_pp"].append(imp_pval_pp)

                    batch_rec["ups_count_conf"].append(ups_count_errstd)
                    batch_rec["imp_count_conf"].append(ups_count_errstd)

                    batch_rec["ups_pval_conf"].append(ups_pval_errstd)
                    batch_rec["imp_pval_conf"].append(imp_pval_errstd)

                    batch_rec["ups_count_mse"].append(ups_count_mse)
                    batch_rec["imp_count_mse"].append(imp_count_mse)

                    batch_rec["ups_pval_mse"].append(ups_pval_mse)
                    batch_rec["imp_pval_mse"].append(imp_pval_mse)

                    batch_rec["imp_count_spearman"].append(imp_count_spearman)
                    batch_rec["ups_count_spearman"].append(ups_count_spearman)
                    batch_rec["imp_pval_spearman"].append(imp_pval_spearman)
                    batch_rec["ups_pval_spearman"].append(ups_pval_spearman)

                    batch_rec["imp_count_pearson"].append(imp_count_pearson)
                    batch_rec["ups_count_pearson"].append(ups_count_pearson)
                    batch_rec["imp_pval_pearson"].append(imp_pval_pearson)
                    batch_rec["ups_pval_pearson"].append(ups_pval_pearson)

                    for k in [
                        "imp_pval_r2", "imp_pval_pearson", 
                        "imp_pval_spearman", "imp_count_r2", 
                        "imp_count_pearson", "imp_count_spearman",
                        "imp_count_loss", "ups_count_loss",
                        "imp_pval_loss", "ups_pval_loss"]:

                        mean_value = np.mean(batch_rec[k]) if not np.isnan(np.mean(batch_rec[k])) else 0
                        if "loss" in k:
                            mean_value = -1 * mean_value # since we are monitoring increasing trends

                        if k not in progress_monitor.keys():
                            progress_monitor[k] = []

                        progress_monitor[k].append(mean_value)

                        if k not in prog_mon_ema.keys():
                            prog_mon_ema[k] = mean_value
                        else:
                            alpha = 0.01 # APR4 change
                            prog_mon_ema[k] = alpha*mean_value + (1-alpha)*prog_mon_ema[k]

                        if k not in prog_mon_best_so_far.keys():
                            prog_mon_best_so_far[k] = mean_value
                        
                    if not self.cosine_sched:
                        # check if improvement in EMA
                        statement_prog_imp_pval_r2 = bool(prog_mon_ema["imp_pval_r2"] > prog_mon_best_so_far["imp_pval_r2"] + prog_monitor_delta)
                        statement_prog_imp_pval_pearson = bool(prog_mon_ema["imp_pval_pearson"] > prog_mon_best_so_far["imp_pval_pearson"] + prog_monitor_delta)
                        statement_prog_imp_pval_spearman = bool(prog_mon_ema["imp_pval_spearman"] > prog_mon_best_so_far["imp_pval_spearman"] + prog_monitor_delta)
                        statement_prog_imp_count_r2 = bool(prog_mon_ema["imp_count_r2"] > prog_mon_best_so_far["imp_count_r2"] + prog_monitor_delta)
                        statement_prog_imp_count_pearson = bool(prog_mon_ema["imp_count_pearson"] > prog_mon_best_so_far["imp_count_pearson"] + prog_monitor_delta)
                        statement_prog_imp_count_spearman = bool(prog_mon_ema["imp_count_spearman"] > prog_mon_best_so_far["imp_count_spearman"] + prog_monitor_delta)

                        for k in ["imp_pval_r2", "imp_pval_pearson", "imp_pval_spearman", "imp_count_r2", "imp_count_pearson", "imp_count_spearman"]:
                            if epoch > 0:
                                prog_mon_best_so_far[k] = max(prog_mon_best_so_far[k], prog_mon_ema[k])
                            else:
                                prog_mon_best_so_far[k] = 0.0

                        if not any([
                            statement_prog_imp_pval_r2, statement_prog_imp_pval_pearson, statement_prog_imp_pval_spearman,
                            statement_prog_imp_count_r2, statement_prog_imp_count_pearson, statement_prog_imp_count_spearman]):
                            no_prog_mon_improvement += 1
                        else:
                            no_prog_mon_improvement = 0
                        
                        if no_prog_mon_improvement >= prog_monitor_patience:
                            print(f"No improvement in EMA for {no_prog_mon_improvement} steps. Adjusting learning rate...")
                            current_lr = self.optimizer.param_groups[0]['lr']
                            self.scheduler.step()
                            lr_sch_steps_taken += 1
                            prog_monitor_patience *= 1.05
                            new_lr = self.optimizer.param_groups[0]['lr']
                            print(f"Learning rate adjusted from {current_lr:.2e} to {new_lr:.2e}")
                            no_prog_mon_improvement = 0

                    del output_p, output_n, output_mu, output_var, loss, obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss
                    del X_batch, mX_batch, mY_batch, avX_batch, Y_batch, pval_batch, observed_map, masked_map
                    if DNA:
                        del dnaseq_batch
                    gc.collect()
                
                if hook:
                    # Initialize variables to store maximum gradient norms and corresponding layer names
                    max_weight_grad_norm = 0
                    max_weight_grad_layer = None
                    max_bias_grad_norm = 0
                    max_bias_grad_layer = None

                    # Check and update maximum gradient norms
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'weight') and module.weight is not None and hasattr(module.weight, 'grad_norm'):
                            if module.weight.grad_norm > max_weight_grad_norm:
                                max_weight_grad_norm = module.weight.grad_norm
                                max_weight_grad_layer = name

                        if hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'grad_norm') and module.bias.grad_norm is not None:
                            if module.bias.grad_norm > max_bias_grad_norm:
                                max_bias_grad_norm = module.bias.grad_norm
                                max_bias_grad_layer = name

                    if max_weight_grad_layer:
                        print(f"Max Weight Grad Layer: {max_weight_grad_layer}, Weight Grad Norm: {max_weight_grad_norm:.3f}")

                self.optimizer.step()
                if self.cosine_sched:
                    self.scheduler.step()

                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                del _X_batch, _mX_batch, _avX_batch, _Y_batch, _mY_batch, _avY_batch, _pval_batch
                if DNA:
                    del _dnaseq_batch
                gc.collect()

                CurrentLR = self.optimizer.param_groups[0]['lr']
                if self.cosine_sched:
                    lr_printstatement = f"CurrentLR: {CurrentLR:.0e}" 
                    
                else:
                    lr_printstatement = f"LR_sch_steps_taken {lr_sch_steps_taken} | LR_patience {no_prog_mon_improvement} | CurrentLR: {CurrentLR:.0e}"

                logstr = [
                    f"Ep. {epoch}",
                    f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                    f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer / len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                    f"Bios Prog. {self.dataset.bios_pointer / self.dataset.num_bios:.2%}", "\n",
                    f"Imp_nbNLL {np.mean(batch_rec['imp_count_loss']):.2f}",
                    f"Ups_nbNLL {np.mean(batch_rec['ups_count_loss']):.2f}",
                    f"Imp_gNLL {np.mean(batch_rec['imp_pval_loss']):.2f}",
                    f"Ups_gNLL {np.mean(batch_rec['ups_pval_loss']):.2f}", "\n",
                    f"Imp_Count_R2 {np.mean(batch_rec['imp_count_r2']):.2f}",
                    f"Ups_Count_R2 {np.mean(batch_rec['ups_count_r2']):.2f}",
                    f"Imp_Pval_R2 {np.mean(batch_rec['imp_pval_r2']):.2f}",
                    f"Ups_Pval_R2 {np.mean(batch_rec['ups_pval_r2']):.2f}", "\n",
                    f"Imp_Count_PP {np.mean(batch_rec['imp_count_pp']):.2f}",
                    f"Ups_Count_PP {np.mean(batch_rec['ups_count_pp']):.2f}",
                    f"Imp_Pval_PP {np.mean(batch_rec['imp_pval_pp']):.2f}",
                    f"Ups_Pval_PP {np.mean(batch_rec['ups_pval_pp']):.2f}", "\n",
                    f"Imp_Count_Conf {np.mean(batch_rec['imp_count_conf']):.2f}",
                    f"Ups_Count_Conf {np.mean(batch_rec['ups_count_conf']):.2f}",
                    f"Imp_Pval_Conf {np.mean(batch_rec['imp_pval_conf']):.2f}",
                    f"Ups_Pval_Conf {np.mean(batch_rec['ups_pval_conf']):.2f}", "\n",
                    f"Imp_Count_MSE {np.mean(batch_rec['imp_count_mse']):.2f}",
                    f"Ups_Count_MSE {np.mean(batch_rec['ups_count_mse']):.2f}",
                    f"Imp_Pval_MSE {np.mean(batch_rec['imp_pval_mse']):.2f}",
                    f"Ups_Pval_MSE {np.mean(batch_rec['ups_pval_mse']):.2f}", "\n",
                    f"Imp_Count_SRCC {np.mean(batch_rec['imp_count_spearman']):.2f}",
                    f"Ups_Count_SRCC {np.mean(batch_rec['ups_count_spearman']):.2f}",
                    f"Imp_Pval_SRCC {np.mean(batch_rec['imp_pval_spearman']):.2f}",
                    f"Ups_Pval_SRCC {np.mean(batch_rec['ups_pval_spearman']):.2f}", "\n",
                    f"Imp_Count_PCC {np.mean(batch_rec['imp_count_pearson']):.2f}",
                    f"Ups_Count_PCC {np.mean(batch_rec['ups_count_pearson']):.2f}",
                    f"Imp_Pval_PCC {np.mean(batch_rec['imp_pval_pearson']):.2f}",
                    f"Ups_Pval_PCC {np.mean(batch_rec['ups_pval_pearson']):.2f}", "\n",
                    
                    
                    f"EMA_imp_pval_r2 {prog_mon_ema['imp_pval_r2']:.2f}",
                    f"EMA_imp_pval_PCC {prog_mon_ema['imp_pval_pearson']:.2f}",
                    f"EMA_imp_pval_SRCC {prog_mon_ema['imp_pval_spearman']:.2f}", 
                    f"EMA_imp_pval_loss {-1*prog_mon_ema['imp_pval_loss']:.2f}", "\n", # -1 since we multiplied it to -1 earlier :))

                    f"EMA_imp_count_r2 {prog_mon_ema['imp_count_r2']:.2f}",
                    f"EMA_imp_count_PCC {prog_mon_ema['imp_count_pearson']:.2f}",
                    f"EMA_imp_count_SRCC {prog_mon_ema['imp_count_spearman']:.2f}", 
                    f"EMA_imp_count_loss {-1*prog_mon_ema['imp_count_loss']:.2f}", "\n", # -1 since we multiplied it to -1 earlier :))
                    
                    f"took {int(minutes)}:{int(seconds):02d}", 
                    f"Gradient_Norm {np.mean(batch_rec['grad_norm']):.2f}",
                    f"num_mask {num_mask}", lr_printstatement, "\n"
                ]

                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                print(logstr)

                if lr_sch_steps_taken >= 100 and early_stop:
                    print("Early stopping due to super small learning rate...")
                    return self.model, best_metric
                
                #################################################################################
                #################################################################################
                if early_stop:
                    epoch_rec["imp_count_r2"].append(np.mean(batch_rec['imp_count_r2']))
                    epoch_rec["imp_pval_r2"].append(np.mean(batch_rec['imp_pval_r2']))
                    epoch_rec["imp_count_spearman"].append(np.mean(batch_rec['imp_count_spearman']))
                    epoch_rec["imp_pval_spearman"].append(np.mean(batch_rec['imp_pval_spearman']))
                    epoch_rec["imp_count_pearson"].append(np.mean(batch_rec['imp_count_pearson']))
                    epoch_rec["imp_pval_pearson"].append(np.mean(batch_rec['imp_pval_pearson']))
                #################################################################################
                #################################################################################

                chr0 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                dsf_pointer0 = self.dataset.dsf_pointer
                bios_pointer0 = self.dataset.bios_pointer

                next_epoch = self.dataset.update_batch_pointers()

                dsf_pointer1 = self.dataset.dsf_pointer
                chr1 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                bios_pointer1 = self.dataset.bios_pointer

                if (chr0 != chr1) or (dsf_pointer0 != dsf_pointer1) or (bios_pointer0 != bios_pointer1):
                    logfile = open(f"models/CANDI{arch}_log.txt", "w")
                    logfile.write("\n".join(log_strs))
                    logfile.close()

                    if (chr0 != chr1):
                        try:
                            validation_set_eval, val_metrics = val_eval.get_validation(self.model)
                            torch.cuda.empty_cache()
                            log_strs.append(validation_set_eval)
                            print(validation_set_eval)
                            log_resource_usage()

                            if early_stop:
                                epoch_rec["val_count_median_imp_r2"].append(val_metrics["imputed_counts"]["R2_count"]["median"])
                                epoch_rec["val_count_median_imp_pcc"].append(val_metrics["imputed_counts"]["PCC_count"]["median"])
                                epoch_rec["val_count_median_imp_srcc"].append(val_metrics["imputed_counts"]["SRCC_count"]["median"])
                                
                                epoch_rec["val_pval_median_imp_r2"].append(val_metrics["imputed_pvals"]["R2_pval"]["median"])
                                epoch_rec["val_pval_median_imp_pcc"].append(val_metrics["imputed_pvals"]["PCC_pval"]["median"])
                                epoch_rec["val_pval_median_imp_srcc"].append(val_metrics["imputed_pvals"]["SRCC_pval"]["median"])
                        except:
                            pass
                    
                    if self.HPO==False:
                        try:
                            try:
                                if os.path.exists(f'models/CANDI{arch}_model_checkpoint_epoch{epoch}_{chr0}.pth'):
                                    os.system(f"rm -rf models/CANDI{arch}_model_checkpoint_epoch{epoch}_{chr0}.pth")
                                torch.save(self.model.state_dict(), f'models/CANDI{arch}_model_checkpoint_epoch{epoch}_{chr1}.pth')
                            except:
                                pass

                            # Generate and process the plot
                            fig_title = " | ".join([
                                f"Ep. {epoch}", f"DSF{self.dataset.dsf_list[dsf_pointer0]}->{1}",
                                f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]}"])
                            
                            if "eic" in arch:
                                plot_buf = val_eval.generate_training_gif_frame_eic(self.model, fig_title)
                            else:
                                plot_buf = val_eval.generate_training_gif_frame(self.model, fig_title)

                            images.append(imageio.imread(plot_buf))
                            plot_buf.close()
                            imageio.mimsave(gif_filename, images, duration=0.5 * len(images))
                        except Exception as e:
                            pass

                if next_epoch:
                    try:
                        validation_set_eval, val_metrics = val_eval.get_validation(self.model)
                        torch.cuda.empty_cache()
                        log_strs.append(validation_set_eval)
                        print(validation_set_eval)
                        log_resource_usage()

                        if early_stop:
                            epoch_rec["val_count_mean_imp_r2"].append(val_metrics["imputed_counts"]["R2_count"]["mean"])
                            epoch_rec["val_count_mean_imp_pcc"].append(val_metrics["imputed_counts"]["PCC_count"]["mean"])
                            epoch_rec["val_count_mean_imp_srcc"].append(val_metrics["imputed_counts"]["SRCC_count"]["mean"])
                            
                            epoch_rec["val_pval_mean_imp_r2"].append(val_metrics["imputed_pvals"]["R2_pval"]["mean"])
                            epoch_rec["val_pval_mean_imp_pcc"].append(val_metrics["imputed_pvals"]["PCC_pval"]["mean"])
                            epoch_rec["val_pval_mean_imp_srcc"].append(val_metrics["imputed_pvals"]["SRCC_pval"]["mean"])
                    except:
                        pass 

            # if early_stop:
            #     # Initialize the best metrics if it's the first epoch
            #     if best_metric is None:
            #         best_metric = {key: None for key in epoch_rec.keys()}
            #         patience_counter = {key: 0 for key in epoch_rec.keys()}

            #     # Loop over all metrics
            #     for metric_name in epoch_rec.keys():
            #         current_metric = np.mean(epoch_rec[metric_name])  # Calculate the current epoch's mean for this metric

            #         if best_metric[metric_name] is None or current_metric > best_metric[metric_name] + early_stop_delta:
            #             best_metric[metric_name] = current_metric  # Update the best metric for this key
            #             patience_counter[metric_name] = 0  # Reset the patience counter
            #         else:
            #             patience_counter[metric_name] += 1  # Increment the patience counter if no improvement

            #     # Check if all patience counters have exceeded the limit (e.g., 3 epochs of no improvement)
            #     if all(patience_counter[metric] >= patience for metric in patience_counter.keys()):
            #         print(f"Early stopping at epoch {epoch}. No significant improvement across metrics.")
            #         logfile = open(f"models/CANDI{arch}_log.txt", "w")
            #         logfile.write("\n".join(log_strs))
            #         logfile.write(f"\n\nFinal best metric records:\n")
            #         for metric_name, value in best_metric.items():
            #             logfile.write(f"{metric_name}: {value}\n")
            #         logfile.close()
            #         return self.model, best_metric
            #     else:
            #         print(f"best metric records so far: \n{best_metric}")
            #         logfile = open(f"models/CANDI{arch}_log.txt", "w") 
            #         logfile.write("\n".join(log_strs))
            #         logfile.write(f"\n\nBest metric records so far:\n")
            #         for metric_name, value in best_metric.items():
            #             logfile.write(f"{metric_name}: {value}\n")
            #         logfile.close()
                
            if self.HPO==False and epoch != (num_epochs-1):
                try:
                    os.system(f"rm -rf models/CANDI{arch}_model_checkpoint_epoch{epoch-1}.pth")
                    torch.save(self.model.state_dict(), f'models/CANDI{arch}_model_checkpoint_epoch{epoch}.pth')
                except:
                    pass
        
        if early_stop:
            return self.model, best_metric
        else:
            return self.model

def Train_CANDI(hyper_parameters, eic=False, checkpoint_path=None, DNA=False, suffix="", prog_mask=False, device=None, HPO=False):
    if eic:
        arch="eic"
    else:
        arch="full"
    
    if DNA:
        arch = f"{arch}_DNA"

    if prog_mask:
        arch = f"{arch}_prog_mask"
    else:
        arch = f"{arch}_random_mask"

    arch = f"{arch}_{suffix}"
    # Defining the hyperparameters
    resolution = 25
    n_sab_layers = hyper_parameters["n_sab_layers"]
    
    epochs = hyper_parameters["epochs"]
    num_training_loci = hyper_parameters["num_loci"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]
    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    min_avail = hyper_parameters["min_avail"]
    inner_epochs = hyper_parameters["inner_epochs"]

    n_cnn_layers = hyper_parameters["n_cnn_layers"]
    conv_kernel_size = hyper_parameters["conv_kernel_size"]
    pool_size = hyper_parameters["pool_size"]
    expansion_factor = hyper_parameters["expansion_factor"]
    pos_enc = hyper_parameters["pos_enc"]
    separate_decoders = hyper_parameters["separate_decoders"]
    merge_ct = hyper_parameters["merge_ct"]
    loci_gen = hyper_parameters["loci_gen"]

    dataset = ExtendedEncodeDataHandler(hyper_parameters["data_path"])
    dataset.initialize_EED(
        m=num_training_loci, context_length=context_length*resolution, 
        bios_batchsize=batch_size, loci_batchsize=1, loci_gen=loci_gen, 
        bios_min_exp_avail_threshold=min_avail, check_completeness=True, 
        eic=eic, merge_ct=merge_ct,
        DSF_list=[1,2], 
        must_have_chr_access=hyper_parameters["must_have_chr_access"])

    signal_dim = dataset.signal_dim
    metadata_embedding_dim = dataset.signal_dim * 4

    if DNA:
        if hyper_parameters["unet"]:
            model = CANDI_UNET(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, hyper_parameters["nhead"],
                n_sab_layers, pool_size=pool_size, dropout=hyper_parameters["dropout"], context_length=context_length, 
                pos_enc=pos_enc, expansion_factor=expansion_factor, separate_decoders=separate_decoders)
        else:
            model = CANDI_DNA(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, hyper_parameters["nhead"],
                n_sab_layers, pool_size=pool_size, dropout=hyper_parameters["dropout"], context_length=context_length, 
                pos_enc=pos_enc, expansion_factor=expansion_factor, separate_decoders=separate_decoders)

    else:
        model = CANDI(
            signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, hyper_parameters["nhead"],
            n_sab_layers, pool_size=pool_size, dropout=hyper_parameters["dropout"], context_length=context_length,
            pos_enc=pos_enc, expansion_factor=expansion_factor, separate_decoders=separate_decoders)

    if hyper_parameters["optim"].lower()=="adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif hyper_parameters["optim"].lower()=="adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif hyper_parameters["optim"].lower()=="adamax":
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if hyper_parameters["LRschedule"] is None:
        cos_sch=False
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=hyper_parameters["lr_halflife"], gamma=1)
    elif hyper_parameters["LRschedule"].lower()=="cosine":
        cos_sch=True
        num_total_epochs = epochs * inner_epochs * len(dataset.m_regions) * 2
        warmup_epochs = inner_epochs * len(dataset.m_regions) * 2
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs), 
                CosineAnnealingLR(optimizer, T_max=(num_total_epochs - warmup_epochs), eta_min=0.0)],
            milestones=[warmup_epochs])
    else:
        cos_sch=False
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=hyper_parameters["lr_halflife"], gamma=0.95)

    print(f"Using optimizer: {optimizer.__class__.__name__}")

    if checkpoint_path is not None:
        print("loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path))

    print(f"CANDI_{arch} # model_parameters: {count_parameters(model)}")
    summary(model)
    
    model_name = f"CANDI{arch}_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = CANDI_LOSS()

    start_time = time.time()

    trainer = PRETRAIN(
        model, dataset, criterion, optimizer, scheduler, 
        device=device, HPO=HPO, cosine_sched=cos_sch)

    model, best_metric = trainer.pretrain_CANDI(
        num_epochs=epochs, mask_percentage=mask_percentage, context_length=context_length, 
        batch_size=batch_size, inner_epochs=inner_epochs, arch=arch, DNA=DNA)

    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    if not HPO:
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))

        # Write a description text file
        description = {
            "hyper_parameters": hyper_parameters,
            "model_architecture": str(model),
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "number_of_model_parameters": count_parameters(model),
            "training_duration": int(end_time - start_time)
        }
        with open(os.path.join(model_dir, model_name.replace(".pt", ".txt")), 'w') as f:
            f.write(json.dumps(description, indent=4))

    return model, best_metric

class CANDI_LOADER(object):
    def __init__(self, model_path, hyper_parameters, DNA=False):
        self.model_path = model_path
        self.hyper_parameters = hyper_parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DNA = DNA

    def load_CANDI(self):
        signal_dim = self.hyper_parameters["signal_dim"]
        dropout = self.hyper_parameters["dropout"]
        nhead = self.hyper_parameters["nhead"]
        n_sab_layers = self.hyper_parameters["n_sab_layers"]
        metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
        context_length = self.hyper_parameters["context_length"]

        n_cnn_layers = self.hyper_parameters["n_cnn_layers"]
        conv_kernel_size = self.hyper_parameters["conv_kernel_size"]
        pool_size = self.hyper_parameters["pool_size"]
        
        if self.DNA:
            model = CANDI_DNA(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length)
        else:
            model = CANDI(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length)

        model.load_state_dict(torch.load(self.model_path, map_location=self.device)) 

        model = model.to(self.device)
        return model
    
def main():
    parser = argparse.ArgumentParser(description="Train the model with specified hyperparameters")

    # Hyperparameters
    parser.add_argument('--data_path', type=str, default="/project/compbio-lab/encode_data/", help='Path to the data')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--n_cnn_layers', type=int, default=3, help='Number of CNN layers')
    parser.add_argument('--conv_kernel_size', type=int, default=3, help='Convolution kernel size')
    parser.add_argument('--pool_size', type=int, default=2, help='Pooling size')
    parser.add_argument('--expansion_factor', type=int, default=3, help='Expansion factor for the model')

    parser.add_argument('--nhead', type=int, default=9, help='Number of attention heads')
    parser.add_argument('--n_sab_layers', type=int, default=4, help='Number of SAB layers')
    parser.add_argument('--pos_enc', type=str, default="relative", help='Transformer Positional Encodings')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--inner_epochs', type=int, default=1, help='Number of inner epochs')
    parser.add_argument('--mask_percentage', type=float, default=0.2, help='Masking percentage (if used)')
    parser.add_argument('--context_length', type=int, default=1200, help='Context length')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_loci', type=int, default=5000, help='Number of loci')
    parser.add_argument('--lr_halflife', type=int, default=1, help='Learning rate halflife')
    parser.add_argument('--min_avail', type=int, default=3, help='Minimum available')
    parser.add_argument('--hpo', action='store_true', help='Flag to enable hyperparameter optimization')
    parser.add_argument('--shared_decoders', action='store_true', help='Flag to enable shared decoders for pval and count')
    parser.add_argument('--suffix', type=str, default='', help='Optional suffix for model name')
    parser.add_argument('--merge_ct', action='store_true', help='Flag to enable merging celltypes')
    parser.add_argument('--loci_gen', type=str, default="ccre", help='Loci generation method')

    parser.add_argument('--optim', type=str, default="sgd", help='optimizer')
    parser.add_argument('--unet', action='store_true', help='whether to use unet skip connections')
    parser.add_argument('--LRschedule', type=str, default=None, help='optimizer lr scheduler')
    
    # Flags for DNA and EIC
    parser.add_argument('--eic', action='store_true', help='Flag to enable EIC')
    parser.add_argument('--dna', action='store_true', help='Flag to enable DNA')
    parser.add_argument('--prog_mask', action='store_true', help='Flag to enable progressive masking')

    # Add checkpoint argument
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to checkpoint model for continued training')

    # Parse the arguments
    args = parser.parse_args()
    separate_decoders = not args.shared_decoders
    merge_ct = True
    must_have_chr_access = True

    # Convert parsed arguments into a dictionary for hyperparameters
    hyper_parameters = {
        "data_path": args.data_path,
        "dropout": args.dropout,
        "n_cnn_layers": args.n_cnn_layers,
        "conv_kernel_size": args.conv_kernel_size,
        "pool_size": args.pool_size,
        "expansion_factor": args.expansion_factor,
        "nhead": args.nhead,
        "n_sab_layers": args.n_sab_layers,
        "pos_enc": args.pos_enc,
        "epochs": args.epochs,
        "inner_epochs": args.inner_epochs,
        "mask_percentage": args.mask_percentage,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_loci": args.num_loci,
        "lr_halflife": args.lr_halflife,
        "min_avail": args.min_avail,
        "hpo": args.hpo,
        "separate_decoders": separate_decoders,
        "merge_ct": merge_ct,
        "loci_gen": args.loci_gen,
        "must_have_chr_access": must_have_chr_access,

        "optim": args.optim,
        "unet": args.unet,
        "LRschedule": args.LRschedule
    }

    # Call your training function with parsed arguments, including checkpoint
    Train_CANDI(hyper_parameters, eic=args.eic, checkpoint_path=args.checkpoint, 
                DNA=True, suffix=args.suffix, prog_mask=args.prog_mask, HPO=args.hpo)

if __name__ == "__main__":
    main()

#  watch -n 20 "squeue -u mfa76 && echo  && tail -n 15 models/*sab*txt && echo  && tail -n 15 models/*def*txt && echo  && tail -n 15 models/*XL*txt"

# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun9_CosSched --LRschedule cosine 
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun9_adamw --optim adamw 
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun9_adam --optim adam 
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun9_onedec --shared_decoders

# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun12_unet --unet
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun12_unet_CosSched --unet --LRschedule cosine 
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun12_unet_adamax --unet --optim adamax
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun12_unet_onedec --unet --shared_decoders

# python CANDI.py --dna --eic --hpo --suffix def_jun13_unet_onedec_admx --unet --shared_decoders --optim adamax
# python CANDI.py --dna --eic --hpo --context_length 2400 --expansion_factor 2 --n_cnn_layers 5 --suffix XLcntx_jun13_unet_onedec --unet --shared_decoders --optim adamax


"""
#!/bin/bash
#SBATCH -J candi_abl9_def
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07-00:00
#SBATCH --mem=50G
#SBATCH --output=candi_abl9_def.out
#SBATCH --partition=compbio-lab-long
#SBATCH --nodelist=cs-venus-03

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sslgpu

srun python CANDI.py --dna --eic --hpo --suffix def_imponly
"""