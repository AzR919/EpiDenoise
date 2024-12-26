
class PROCESS_EIC_DATA(object):
    def __init__(self, path, max_len=8000, resolution=25, stratified=False):
        self.path = path
        self.stratified = stratified
        self.resolution = resolution
        self.max_len = max_len * self.resolution #converts max_len from #bins to #bp
        self.util = COORD(resolution=self.resolution, Meuleman_file="_", outdir=self.path)
        self.genomesize = sum(list(self.util.chr_sizes.values()))
        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]

        self.biosamples = {}
        for f in os.listdir(self.path):
            if ".bigwig" in f: 
                if f[:3] not in self.biosamples.keys():
                    self.biosamples[f[:3]] = {}
                    
                self.biosamples[f[:3]][f[3:6]] = pyBigWig.open(self.path + "/" + f)

    def pkl_generate_m_samples(self, m, multi_p=True, n_p=20): # m per biosample           
        if self.stratified:
            self.util.get_foreground()
            df = self.util.foreground
            df = df[df["chrom"].isin(self.util.chr_sizes.keys())]
            m_regions = []
            used_regions = {chr: [] for chr in df['chrom'].unique()}

            # Sort the DataFrame by chromosome and start position
            df = df.sort_values(['chrom', 'start'])

            # Select m/2 regions from the DataFrame
            for _ in range(m // 2):
                while True:
                    # Select a random row from the DataFrame
                    row = df.sample(1).iloc[0]

                    # Generate a start position that is divisible by self.resolution and within the region
                    rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len


                    # Check if the region overlaps with any existing region in the same chromosome
                    if rand_start >= 0 and rand_end <= self.util.chr_sizes[row['chrom']]:
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                            m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
            # Select m/2 regions that are not necessarily in the DataFrame 
            for chr, size in self.util.chr_sizes.items():
                m_c = int((m // 2) * (size / self.genomesize))  # Calculate the number of instances from each chromosome proportional to its size
                for _ in range(m_c):
                    while True:
                        # Generate a random start position that is divisible by self.resolution
                        rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                        rand_end = rand_start + self.max_len

                        # Check if the region overlaps with any existing region
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                            m_regions.append([chr, rand_start, rand_end])
                            used_regions[chr].append((rand_start, rand_end))
                            break

        else:
            m_regions = []
            used_regions = {chr: [] for chr in self.util.chr_sizes.keys()}

            for chr, size in self.util.chr_sizes.items():
                m_c = int(m * (size / self.genomesize))

                for _ in range(m_c):
                    while True:
                        # Generate a random start position that is divisible by self.resolution
                        rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                        rand_end = rand_start + self.max_len

                        # Check if the region overlaps with any existing region in the same chromosome
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                            m_regions.append([chr, rand_start, rand_end])
                            used_regions[chr].append((rand_start, rand_end))
                            break

        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in self.biosamples[bios].keys():
                bios_data[assay] = []

                bw = self.biosamples[bios][assay]
                bw_query_dicts = []
                for i in range(len(m_regions)):
                    r = m_regions[i]
                    bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})

                if multi_p:
                    with mp.Pool(n_p) as p:
                        m_signals = p.map(get_bin_value, bw_query_dicts)
                    
                    for i in range(len(m_signals)):
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], m_signals[i]))
                else:
                    for i in range(len(bw_query_dicts)):
                        signals = get_bin_value(bw_query_dicts[i])
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], signals))
                    
            file_path = f"{self.path}/{bios}_m{m}_{self.resolution}bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")

    def generate_m_samples(self, m, n_datasets=50, multi_p=True, n_p=10):
        if self.stratified:
            self.util.get_foreground()
            df = self.util.foreground
            df = df[df["chrom"].isin(self.util.chr_sizes.keys())]
            m_regions = []
            used_regions = {chr: [] for chr in df['chrom'].unique()}

            # Sort the DataFrame by chromosome and start position
            df = df.sort_values(['chrom', 'start'])

            # Select m/2 regions from the DataFrame
            while len(m_regions) < (m // 2):
                while True:
                    # Select a random row from the DataFrame
                    row = df.sample(1).iloc[0]

                    # Generate a start position that is divisible by self.resolution and within the region
                    rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region in the same chromosome
                    if rand_start >= 0 and rand_end <= self.util.chr_sizes[row['chrom']]:
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                            m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
            # Select m/2 regions that are not necessarily in the DataFrame 
            for chr, size in self.util.chr_sizes.items():
                m_c = int((m // 2) * (size / self.genomesize))  # Calculate the number of instances from each chromosome proportional to its size
                mii = 0
                while mii < m_c:
                    # Generate a random start position that is divisible by self.resolution
                    rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                        m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1 
                        break

        else:
            m_regions = []
            used_regions = {chr: [] for chr in self.util.chr_sizes.keys()}

            for chr, size in self.util.chr_sizes.items():
                m_c = int(m * (size / self.genomesize))
                mii = 0

                while mii < m_c:
                    # Generate a random start position that is divisible by self.resolution
                    rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region in the same chromosome
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                        m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1 

        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        ds_number = 0  
        print("m2:   ", len(m_regions))
        samples_per_ds = len(m_regions) // n_datasets
        for ds_i in range(0, len(m_regions), samples_per_ds):
            ds_number += 1

            ds_i_regions = m_regions[ds_i : (ds_i + samples_per_ds)]
            ds_i_regions.sort(key=lambda x: x[1]) # sorted based on start coord
            
            all_samples_tensor = []

            for bios in self.biosamples.keys():
                print("     ct:   ", bios)
                bios_data = {}

                for assay in self.all_assays:
                    bios_data[assay] = []

                    if assay in self.biosamples[bios].keys(): # if available
                        print("         assay:   ", assay)
                        bw = self.biosamples[bios][assay]
                        bw_query_dicts = []

                        for r in ds_i_regions:
                            bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})
                        
                        if multi_p:
                            with mp.Pool(n_p) as p:
                                outs = p.map(get_bin_value_dict, bw_query_dicts)
                        else:
                            outs = []
                            for ii in range(len(bw_query_dicts)):
                                outs.append(get_bin_value_dict(bw_query_dicts[ii]))

                        outs.sort(key=lambda x: x['start']) # assert is sorted based on start coord
                        m_signals = [o["signals"] for o in outs]
                        
                        for sample in m_signals:
                            bios_data[assay].append(sample)

                    else: # if missing
                        for r in ds_i_regions:
                            bios_data[assay].append([-1 for _ in range(self.max_len // self.resolution)])
                
                # Convert bios_data to a numpy array
                bios_data_array = np.array([bios_data[assay] for assay in self.all_assays], dtype=np.float32)

                # Add bios_data_array to all_samples
                all_samples_tensor.append(bios_data_array)

            # Convert all_samples to a numpy array
            all_samples_tensor = np.array(all_samples_tensor)

            # Convert all_samples_array to a PyTorch tensor
            all_samples_tensor = torch.from_numpy(all_samples_tensor)

            # Ensure the tensor is of type float
            all_samples_tensor = all_samples_tensor.float()

            all_samples_tensor = torch.permute(all_samples_tensor, (2, 0, 3, 1))
            # Get the shape of the current tensor
            shape = all_samples_tensor.shape

            # Calculate the new dimensions
            new_shape = [shape[0]*shape[1]] + list(shape[2:])

            # Reshape the tensor
            all_samples_tensor = all_samples_tensor.reshape(new_shape)
            
            file_path = f"{self.path}/mixed_dataset{ds_number}_{m//n_datasets}samples_{self.resolution}bp.pt"
            torch.save(all_samples_tensor, file_path)
            print(f"saved DS # {ds_number}, with shape {all_samples_tensor.shape}")

    def load_m_regions(self, file_path):
        # Open the gzip file
        with gzip.open(file_path, 'rb') as f:
            # Load the data using pickle
            bios_data = pickle.load(f)

        # Initialize an empty list to store the m_regions
        m_regions = []

        # Iterate over each biosample and assay
        for sample in bios_data[list(bios_data.keys())[0]]:

            # Append the regions to the m_regions list
            if sample[0] not in m_regions:
                m_regions.append(sample[0])
            
        return m_regions
    
    def generate_m_samples_from_predefined_regions(self, m_regions, multi_p=True, n_p=100):
        m = len(m_regions)
        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in self.biosamples[bios].keys():
                bios_data[assay] = []

                bw = self.biosamples[bios][assay]
                bw_query_dicts = []
                for i in range(len(m_regions)):
                    r = m_regions[i]
                    bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})

                if multi_p:
                    with mp.Pool(n_p) as p:
                        m_signals = p.map(get_bin_value, bw_query_dicts)
                    
                    for i in range(len(m_signals)):
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], m_signals[i]))
                else:
                    for i in range(len(bw_query_dicts)):
                        signals = get_bin_value(bw_query_dicts[i])
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], signals))
                    
            file_path = f"{self.path}/{bios}_m{m}_{self.resolution}bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")
                
    def generate_wg_samples(self):
        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in biosamples[bios].keys():
                bios_data[assay] = {}

                bw = biosamples[bios][assay]
                for chr, size in self.util.chr_sizes.items():
                    signals = get_bin_value(bw, chr, 0, size, self.resolution)
                    bios_data[assay][chr] = signals
            
            file_path = f"{self.path}/{bios}_WG_25bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")



class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000):
        super(RelativePositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.rel_pos_emb = nn.Embedding(self.max_len*2, self.d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.rel_pos_emb(pos + self.max_len)
        return x + pos_emb

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_weights = self.softmax(self.attention(x))
        return (attention_weights * x).sum(dim=1)

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000):
        super(_PositionalEncoding, self).__init__()

        self.d_model = d_model
        d_model_pad = d_model if d_model % 2 == 0 else d_model + 1  # Ensure d_model is even

        # Create a long enough `pe` matrix
        pe = torch.zeros(max_len, d_model_pad)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_pad, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model_pad))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register `pe` as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe.squeeze(1)
        pe = pe[:,:self.d_model]

        return x + pe.unsqueeze(0)

class WeightedMSELoss(nn.Module): 
    # gives more weight to predicting larger signal values rather than depletions
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target):
        weights = target.clone().detach()  # Create a copy of target for weights
        max_val = weights.max()
        if max_val != 0:
            weights = weights / max_val  # Normalize weights to be between 0 and 1
            return torch.sum(weights * ((input - target) ** 2))
        else:
            return torch.sum((input - target) ** 2)

class DoubleMaskMultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(DoubleMaskMultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = MaskedLinear(d_model, d_model)
        self.key = MaskedLinear(d_model, d_model)
        self.value = MaskedLinear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, pmask, fmask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """

        # fmask should be of size d_model * d_model 
        # for each feature index i, if i-th feature is missing fmask[i,:]=0 ; otherwise, fmask[i,:]=1

        # Element-wise multiplication with the weight matrices
        # print("1", torch.sum(self.query.weight.data == 0).item(), self.query.weight.data.sum().item())

        # self.query.weight.data *= fmask
        # self.key.weight.data *= fmask
        # self.value.weight.data *= fmask

        # print("2", torch.sum(self.query.weight.data == 0).item(), self.query.weight.data.sum().item())

        # Element-wise multiplication of mask with the bias terms
        # bias_fmask = fmask.diag()
        # self.query.bias.data *= bias_fmask
        # self.key.bias.data *= bias_fmask
        # self.value.bias.data *= bias_fmask

        # (batch_size, max_len, d_model)
        query = self.query(query, fmask)
        key = self.key(key, fmask)        
        value = self.value(value, fmask)   
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(pmask == 0, -1e9)    

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class DoubleMaskEncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        d_model=35,
        heads=5, 
        feed_forward_hidden=35 * 4, 
        dropout=0.1
        ):
        super(DoubleMaskEncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = DoubleMaskMultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, pmask, fmask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, pmask, fmask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class MaskedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(MaskedConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x, mask):
        not_mask = mask.clone()
        not_mask = ~mask
        x = x.permute(0,2,1)
        not_mask = not_mask.permute(0,2,1)
        x = x * not_mask
        x = self.conv(x)
        x = x.permute(0,2,1)
        return x

class MaskPostConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(MaskPostConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x, mask):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        not_mask = mask.clone()
        not_mask = ~mask
        x = x * not_mask
        return x

class DualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DualConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same
        self.data_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, mask):
        mask = mask.clone()

        x = x.permute(0,2,1)
        mask = mask.permute(0,2,1)

        x = self.data_conv(x)
        mask = self.mask_conv(mask.float())

        x = x * mask
        x = x.permute(0,2,1)
        return x

class TripleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, seq_len, stride=1):
        super(TripleConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same

        # in_channel: num_features, out_channel: num_filters
        # output shape: (batch_size, num_filters, seq_len)
        self.data_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding) 

        # in_channel: num_features, out_channel: num_filters
        # output shape: (batch_size, num_filters, seq_len)
        self.position_mask_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # in_channel: seq_len, out_channel: num_features
        # output shape: (batch_size, 1, num_features)
        self.feature_mask_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        

    def forward(self, x, mask):
        mask = mask.clone()
        # x = x.permute(0,2,1)
        mask = mask.permute(0,2,1)

        # x = self.data_conv(x)
        # mask = self.mask_conv(mask.float()).permute(0,2,1)  # transpose the mask convolutions back to original shape

        x = x * mask
        x = x.permute(0,2,1)
        return x


#________________________________________________________________________________________________________________________#
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim):
        super(TransformerEncoder, self).__init__()

        self.pos_encoder = PositionalEncoding(input_dim, max_len=500)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, src, src_mask):
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class MaskedConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, kernel_size=5):
        super(MaskedConvEncoder, self).__init__()
        self.masked_conv = MaskedConv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, src, src_mask):
        src = self.masked_conv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class MaskPostConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, kernel_size=5):
        super(MaskPostConvEncoder, self).__init__()
        self.masked_conv = MaskPostConv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, src, src_mask):
        src = self.masked_conv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class DualConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, kernel_size=5):
        super(DualConvEncoder, self).__init__()
        
        self.dualconv = DualConv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim, max_len=500)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src, src_mask):
        src = self.dualconv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class TripleConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, seq_len, kernel_size=5):
        super(DualConvEncoder_T, self).__init__()
        
        self.dualconv = DualConv1d_T(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, seq_len=seq_len, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src, src_mask):
        src = self.dualconv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

def mask_missing(data, missing_features_ind, mask_value=-1):
    mask = torch.zeros_like(data, dtype=torch.bool)

    # Loop over the missing feature ids
    for id in missing_features_ind:
        # Set the mask values for the current chunk to True
        mask[:, :, id] = True

    # Create a copy of the data tensor
    masked_data = data.clone()
    # Set the masked data values to the mask_value
    masked_data[mask] = mask_value
    # Return the masked data and the mask
    return masked_data, mask


def __train_model(model, dataset, criterion, optimizer, num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, context_length=2000, batch_size=100, start_epoch=0):
    log_strs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)

    # model.to(device)
    log_strs.append(str(device))
    logfile = open("models/log.txt", "w")
    logfile.write("\n".join(log_strs))
    logfile.close()

    # Define your batch size
    for epoch in range(start_epoch, num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch+1}/{num_epochs}')

        bb=0
        for bios, f in dataset.biosamples.items():
            bb+=1
            print('-' * 10)
            x, missing_mask, missing_f_i = dataset.get_biosample(f)

            # fmask is used to mask QKV of transformer
            num_features = x.shape[2]
            fmask = torch.ones(num_features, num_features)

            for i in missing_f_i:
                fmask[i,:] = 0
            
            fmask = fmask.to(device)
            # Break down x into smaller batches
            for i in range(0, len(x), batch_size):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                
                x_batch = x[i:i+batch_size]
                missing_mask_batch = missing_mask[i:i+batch_size]

                if context_length < 8000:
                    rand_start = random.randint(0, 8000 - (context_length+1))
                    rand_end = rand_start + context_length

                    x_batch, missing_mask_batch = x_batch[:, rand_start:rand_end, :], missing_mask_batch[:, rand_start:rand_end, :]

                # print("missing_mask_batch   ", missing_mask_batch.shape, missing_mask_batch.sum(), len(missing_f_i))

                # Masking a subset of the input data
                masked_x_batch, cloze_mask = mask_data(x_batch, mask_value=-1, chunk=chunk, n_chunks=n_chunks, mask_percentage=mask_percentage)
                pmask = cloze_mask[:,:,0].unsqueeze(1).unsqueeze(1)
                # print("pmask1    ", pmask.shape, pmask.sum())

                # print("cloze_mask1    ", cloze_mask.shape, cloze_mask.sum())
                cloze_mask = cloze_mask & ~missing_mask_batch
                # print("cloze_mask2    ", cloze_mask.shape, cloze_mask.sum())

                # Convert the boolean values to float and switch the masked and non-masked values
                pmask = 1 - pmask.float()
                # print("pmask2    ", pmask.shape, pmask.sum())
                

                # print("x_batch  ", x_batch[cloze_mask].shape, x_batch[cloze_mask].mean().item(), x_batch[cloze_mask].min().item(), x_batch[cloze_mask].max().item())
                # print("masked_x_batch   ", masked_x_batch[cloze_mask].shape, masked_x_batch[cloze_mask].mean().item(), masked_x_batch[cloze_mask].min().item(), masked_x_batch[cloze_mask].max().item())

                x_batch = x_batch.to(device)
                masked_x_batch = masked_x_batch.to(device)
                pmask = pmask.to(device)
                cloze_mask = cloze_mask.to(device)

                outputs = model(masked_x_batch, pmask, fmask)
                loss = criterion(outputs[cloze_mask], x_batch[cloze_mask])


                sum_pred, sum_target = outputs[cloze_mask].sum().item(), x_batch[cloze_mask].sum().item()

                if torch.isnan(loss).sum() > 0:
                    skipmessage = "Encountered nan loss! Skipping batch..."
                    log_strs.append(skipmessage)
                    print(skipmessage)
                    continue

                del x_batch
                del pmask
                del masked_x_batch
                del outputs

                # Clear GPU memory again
                torch.cuda.empty_cache()

                if (((i//batch_size))+1) % 10 == 0 or i==0:
                    logfile = open("models/log.txt", "w")

                    logstr = f'Epoch {epoch+1}/{num_epochs} | Bios {bb}/{len(dataset.biosamples)}| Batch {((i//batch_size))+1}/{(len(x)//batch_size)+1}\
                        | Loss: {loss.item():.4f} | S_P: {sum_pred:.1f} | S_T: {sum_target:.1f}'

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                loss.backward()                    
                optimizer.step()
        
        # Save the model after each epoch
        torch.save(model.state_dict(), f'models/model_checkpoint_epoch_{epoch+1}.pth')

    return model

def _train_model(model, dataset, criterion, optimizer, d_model, num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, context_length=2000, batch_size=100, start_bios=0):
    log_strs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)

    # model.to(device)
    log_strs.append(str(device))
    logfile = open("models/log.txt", "w")
    logfile.write("\n".join(log_strs))
    logfile.close()

    bb=0
    # Define your batch size
    for bios, f in dataset.biosamples.items():
        bb+=1
        if bb < start_bios:
            continue

        print('-' * 10)
        x, missing_mask, missing_f_i = dataset.get_biosample(f)

        # fmask is used to mask QKV of transformer
        num_features = x.shape[2]
        fmask = torch.ones(num_features, d_model)

        for i in missing_f_i:
            fmask[i,:] = 0
        
        fmask = fmask.to(device)
        for epoch in range(0, num_epochs):
            print('-' * 10)
            print(f'Epoch {epoch+1}/{num_epochs}')
            optimizer.zero_grad()
            # Break down x into smaller batches
            for i in range(0, len(x), batch_size):
                torch.cuda.empty_cache()
                
                x_batch = x[i:i+batch_size]
                missing_mask_batch = missing_mask[i:i+batch_size]

                if context_length < 8000:
                    rand_start = random.randint(0, 8000 - (context_length+1))
                    rand_end = rand_start + context_length

                    x_batch, missing_mask_batch = x_batch[:, rand_start:rand_end, :], missing_mask_batch[:, rand_start:rand_end, :]

                # print("missing_mask_batch   ", missing_mask_batch.shape, missing_mask_batch.sum(), len(missing_f_i))

                x_batch = torch.arcsinh_(x_batch)

                # Masking a subset of the input data
                masked_x_batch, cloze_mask = mask_data(x_batch, mask_value=-1, chunk=chunk, n_chunks=n_chunks, mask_percentage=mask_percentage)
                pmask = cloze_mask[:,:,0].squeeze()

                cloze_mask = cloze_mask & ~missing_mask_batch
                x_batch = x_batch.to(device)
                masked_x_batch = masked_x_batch.to(device)
                pmask = pmask.to(device)
                cloze_mask = cloze_mask.to(device)

                outputs = model(masked_x_batch, pmask, fmask)
                loss = criterion(outputs[cloze_mask], x_batch[cloze_mask])

                mean_pred, std_pred = outputs[cloze_mask].mean().item(), outputs[cloze_mask].std().item()
                mean_target, std_target = x_batch[cloze_mask].mean().item(), x_batch[cloze_mask].std().item()

                if torch.isnan(loss).sum() > 0:
                    skipmessage = "Encountered nan loss! Skipping batch..."
                    log_strs.append(skipmessage)
                    print(skipmessage)
                    del x_batch
                    del pmask
                    del masked_x_batch
                    del outputs
                    torch.cuda.empty_cache()
                    continue

                del x_batch
                del pmask
                del masked_x_batch
                del outputs

                # Clear GPU memory again
                torch.cuda.empty_cache()

                if (((i//batch_size))+1) % 10 == 0 or i==0:
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f'Epoch {epoch+1}/{num_epochs}', f"Bios {bb}/{len(dataset.biosamples)}", 
                        f"Batch {((i//batch_size))+1}/{(len(x)//batch_size)+1}",
                        f"Loss: {loss.item():.4f}", 
                        f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                        f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                loss.backward()     

            optimizer.step()

        # Save the model after each epoch
        try:
            torch.save(model.state_dict(), f'models/model_checkpoint_bios_{bb}.pth')
        except:
            pass

    return model


class old__BAM_TO_SIGNAL(object):
    def __init__(self, resolution):
        """
        Initialize the object
        """
        self.resolution = resolution

    def read_chr_sizes(self):
        """
        Read a file with chromosome sizes and return a dictionary where keys are 
        chromosome names and values are chromosome sizes.
        
        Parameters:
        file_path (str): The path to the file with chromosome sizes.

        Returns:
        dict: A dictionary where keys are chromosome names and values are chromosome sizes.
        """

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        
        self.chr_sizes = {}
        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

    def load_bam(self):
        """
        Load the BAM file using pysam.
        """
        self.bam = pysam.AlignmentFile(self.bam_file, 'rb')

    def initialize_empty_bins(self):
        """
        Initialize empty bins for each chromosome based on the resolution.
        """
        self.bins = {chr: [0] * (size // self.resolution + 1) for chr, size in self.chr_sizes.items()}

    def calculate_coverage(self):
        """
        Calculate the coverage for each bin.
        """
        self.bins = {chr: [0] * (self.chr_sizes[chr] // self.resolution + 1) for chr in self.chr_sizes}
        self.coverage = {}

        for chr in self.chr_sizes:
            self.coverage[chr] = {
                'chr': [],
                'start': [],
                'end': [],
                'read_count': []}

            # print(f"getting {chr} coverage...")
            for read in self.bam.fetch(chr):
                start_bin = read.reference_start // self.resolution
                end_bin = read.reference_end // self.resolution
                for i in range(start_bin, end_bin+1):
                    self.bins[chr][i] += 1

            for i, count in enumerate(self.bins[chr]):
                start = i * self.resolution
                end = start + self.resolution
                self.coverage[chr]["chr"].append(str(chr))
                self.coverage[chr]["start"].append(int(start))
                self.coverage[chr]["end"].append(int(end))
                self.coverage[chr]["read_count"].append(float(count))

    def calculate_signal_pvalues(self):
        """
        Calculate the per position signal p-value according to the MACS2 pipeline.
        """
        self.pvalues = {}

        # Calculate the mean coverage across all bins
        mean_coverage = np.mean([np.mean(self.coverage[chr]["read_count"]) for chr in self.coverage.keys()])

        for chr in self.coverage.keys():
            self.pvalues[chr] = {
                'chr': [],
                'start': [],
                'end': [],
                'pvalue': []}

            for i, count in enumerate(self.coverage[chr]["read_count"]):
                # Calculate the p-value of the Poisson distribution
                pvalue = 1 - poisson.cdf(count, mean_coverage)

                # Convert the p-value to -log10(p-value)
                pvalue = -np.log10(pvalue + 1e-19)

                self.pvalues[chr]["chr"].append(str(chr))
                self.pvalues[chr]["start"].append(self.coverage[chr]["start"][i])
                self.pvalues[chr]["end"].append(self.coverage[chr]["end"][i])
                self.pvalues[chr]["pvalue"].append(pvalue)

    def save_coverage_pkl(self):
        """
        Save the coverage data to a pickle file.

        Parameters:
        file_path (str): The path to the pickle file.
        """

        for chr in self.coverage.keys():
            file_path = self.bam_file.replace(".bam", f"_{chr}_cvrg{self.resolution}bp.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(self.coverage[chr], f)
        
            os.system(f"gzip {file_path}")
    
    def save_coverage_bigwig(self):
        """
        Save the coverage data to a BigWig file.

        Parameters:
        file_path (str): The path to the BigWig file.
        """
        file_path = self.bam_file.replace(".bam", f"_cvrg{self.resolution}bp.bw")
        bw = pyBigWig.open(file_path, 'w')
        bw.addHeader([(k, v) for k, v in self.chr_sizes.items()])

        for chr in self.coverage.keys():
            bw.addEntries(
                self.coverage[chr]["chr"], 
                self.coverage[chr]["start"], 
                ends=self.coverage[chr]["end"], 
                values=self.coverage[chr]["read_count"])
        bw.close()

    def save_signal_pkl(self):
        """
        Save the signal pval data to a pickle file.

        Parameters:
        file_path (str): The path to the pickle file.
        """

        for chr in self.pvalues.keys():
            file_path = self.bam_file.replace(".bam", f"_{chr}_signal{self.resolution}bp.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(self.pvalues[chr], f)
        
            os.system(f"gzip {file_path}")
    
    def save_signal_bigwig(self):
        """
        Save the signal pval data to a BigWig file.

        Parameters:
        file_path (str): The path to the BigWig file.
        """
        file_path = self.bam_file.replace(".bam", f"_signal{self.resolution}bp.bw")
        bw = pyBigWig.open(file_path, 'w')
        bw.addHeader([(k, v) for k, v in self.chr_sizes.items()])

        for chr in self.pvalues.keys():
            bw.addEntries(
                self.pvalues[chr]["chr"], 
                self.pvalues[chr]["start"], 
                ends=self.pvalues[chr]["end"], 
                values=self.pvalues[chr]["read_count"])
                
        bw.close()

    def full_preprocess(self, bam_file, chr_sizes_file, resolution=25):
        t0 = datetime.datetime.now()
        self.bam_file = bam_file
        self.chr_sizes_file = chr_sizes_file
        self.resolution = resolution

        self.read_chr_sizes()
        self.load_bam()
        self.initialize_empty_bins()
        self.calculate_coverage()
        self.calculate_signal_pvalues()

        self.save_coverage_pkl()
        self.save_signal_pkl()

        t1 = datetime.datetime.now()
        print(f"took {t1-t0} to get coverage for {bam_file} at resolution: {resolution}bp")



def reshape_tensor(tensor, context_length_factor):
    # Get the original size of the tensor
    samples, seq_length, features = tensor.size()

    # Calculate the new sequence length and number of samples
    new_seq_length = int(seq_length * context_length_factor)
    new_samples = int(samples / context_length_factor)

    # Check if the new sequence length is valid
    if seq_length % new_seq_length != 0:
        raise ValueError("The context_length_factor does not evenly divide the sequence length")

    # Reshape the tensor
    reshaped_tensor = tensor.view(new_samples, new_seq_length, features)

    return reshaped_tensor

def mask_data(data, mask_value=-1, chunk=False, n_chunks=1, mask_percentage=0.15): # used for epidenoise 1.0
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    seq_len = data.size(1)

    if chunk:
        # Calculate the size of each chunk
        chunk_size = int(mask_percentage * seq_len / n_chunks)
    else: 
        chunk_size = 1
        n_chunks =  int(mask_percentage * seq_len)

    # Initialize an empty list to store the start indices
    start_indices = []
    while len(start_indices) < n_chunks:
        # Generate a random start index
        start = torch.randint(0, seq_len - chunk_size, (1,))
        # Check if the chunk overlaps with any existing chunks
        if not any(start <= idx + chunk_size and start + chunk_size >= idx for idx in start_indices):
            # If not, add the start index to the list
            start_indices.append(start.item())

    # Loop over the start indices
    for start in start_indices:
        # Calculate the end index for the current chunk
        end = start + chunk_size
        # Set the mask values for the current chunk to True
        mask[:, start:end, :] = True

    # Create a copy of the data tensor
    masked_data = data.clone()
    # Set the masked data values to the mask_value
    masked_data[mask] = mask_value
    # Return the masked data and the mask

    return masked_data, mask

def mask_data15(data, mask_value=-1, chunk=False, n_chunks=1, mask_percentage=0.15): # used for epidenoise 1.5
    """
    in this version, we added special tokens and made sure not to mask them
    similar to BERT, using 3 different maskings:
        1. mask
        2. replace with random data
        3. do nothing
    """
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    seq_len = data.size(1)
    seglength = (seq_len - 3)/2

    cls_sep_indices = [0, seglength+1, 2*seglength + 2]
    
    if chunk:
        # Calculate the size of each chunk
        chunk_size = int(mask_percentage * seq_len / n_chunks)
    else: 
        chunk_size = 1
        n_chunks =  int(mask_percentage * seq_len)

    # Initialize an empty list to store the start indices
    start_indices = []
    while len(start_indices) < n_chunks:
        # Generate a random start index
        start = torch.randint(0, seq_len - chunk_size, (1,))
        # Check if the chunk overlaps with any existing chunks
        if not any(start <= idx + chunk_size and start + chunk_size >= idx for idx in start_indices + cls_sep_indices):
            # If not, add the start index to the list
            start_indices.append(start.item())

    # Create a copy of the data tensor
    masked_data = data.clone()

    # Loop over the start indices
    for start in start_indices:
        # Calculate the end index for the current chunk
        end = start + chunk_size
        # Set the mask values for the current chunk to True
        mask[:, start:end, :] = True

        # For each position in the chunk, decide how to mask it
        for pos in range(start, end):
            rand_num = random.random()
            if rand_num < 0.8:
                # 80% of the time, replace with mask_value
                masked_data[:, pos, :] = mask_value
            elif rand_num < 0.9:
                # 10% of the time, replace with a random value in the range of the data
                data_min = 0
                data_max = torch.max(data)
                random_value = data_min + torch.rand(1) * (data_max - data_min)
                masked_data[:, pos, :] = random_value

    # Return the masked data and the mask
    return masked_data, mask

def mask_data16(data, available_features, mask_value=-1, chunk_size=6, mask_percentage=0.15): # used for epidenoise 1.6 and 1.7
    """
    dimensions of the data: (batch_size, context_length, features)
    in this version, we make the following changes
    find available features -> for unavailable features, are corresponding values are -1. 
    num_all_signals = context * num_available_features
    num_mask_start = (num_all_signals * mask_percentage) / chunk_size
    randomly select mask_start coordinates 
        length: axis2 (start + chunk_size -> no overlap with special tokens)
        feature: random.choice(available_features)
    """
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    if mask_percentage == 0:
        return data, mask

    seq_len = data.size(1)
    seglength = (seq_len - 3)/2

    special_tokens = [0, seglength+1, (2*seglength)+2]

    # Calculate total number of signals and number of chunks to be masked
    num_all_signals = data.size(1) * len(available_features)
    num_mask_start = int((num_all_signals * mask_percentage) / chunk_size)

    # Loop over the number of chunks to be masked
    for _ in range(num_mask_start):
        while True:
            # Randomly select start coordinates for the chunk
            length_start = torch.randint(0, seq_len - chunk_size, (1,))
            feature_start = available_features[torch.randint(0, len(available_features), (1,))]

            # Check if the chunk overlaps with any special tokens
            if not any(length_start <= idx < length_start+chunk_size for idx in special_tokens):
                break

        # Apply the masking to the selected chunk
        mask[:, length_start:length_start+chunk_size, feature_start] = True
        data[mask] = mask_value

    return data, mask

def mask_data18(data, available_features, mask_value=-1, mask_percentage=0.15):
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)

    if len(available_features) == 1:
        mask_percentage = 0

    if mask_percentage == 0:
        return data, mask

    seq_len = data.size(1)
    num_mask_features = int(len(available_features) * mask_percentage)
    
    if num_mask_features == 0:
        num_mask_features += 1

    selected_indices = []
    while len(selected_indices) < num_mask_features:
        randomF = random.choice(available_features)
        if randomF not in selected_indices:
            selected_indices.append(randomF)

    # Loop over the selected indices
    for mask_f in selected_indices:

        # Apply the masking to the selected chunk
        mask[:, :, mask_f] = True
        
    data[mask] = mask_value

    return data, mask
 

def eDICE_eval():
    e = Evaluation(
        model_path= "models/EpiDenoise_20231210014829_params154531.pt", 
        hyper_parameters_path= "models/hyper_parameters_EpiDenoise_20231210014829_params154531.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/", 
        is_arcsin=True
    )

    preds_dir = "/project/compbio-lab/EIC/mehdi_preds/scratch/"
    obs_dir1 = "/project/compbio-lab/EIC/validation_data/"
    obs_dir2 = "/project/compbio-lab/EIC/blind_data/"

    results = []

    for pf in os.listdir(preds_dir):
        name = pf.replace(".pkl","")
        assay = name[3:]
        ct = name[:3]
        print(ct, assay)

        with open(preds_dir + pf, 'rb') as pf_file:
            pred = pickle.load(pf_file)
            pred = np.sinh(pred)
        
        if pf.replace(".pkl", ".bigwig") in os.listdir(obs_dir1):
            target = torch.load(obs_dir1 + f"/{ct}_chr21_25.pt")
            target = target[:, int(assay.replace("M", "")) - 1].numpy()

        elif pf.replace(".pkl", ".bigwig") in os.listdir(obs_dir2):
            target = torch.load(obs_dir2 + f"/{ct}_chr21_25.pt")
            target = target[:, int(assay.replace("M", "")) - 1].numpy()

        print(pf, target.sum(), pred.sum())
        metrics = {
                'celltype': ct,
                'feature': assay,

                'MSE-GW': e.mse(target, pred),
                'Pearson-GW': e.pearson(target, pred),
                'Spearman-GW': e.spearman(target, pred),

                'MSE-1obs': e.mse1obs(target, pred),
                'Pearson_1obs': e.pearson1_obs(target, pred),
                'Spearman_1obs': e.spearman1_obs(target, pred),

                'MSE-1imp': e.mse1imp(target, pred),
                'Pearson_1imp': e.pearson1_imp(target, pred),
                'Spearman_1imp': e.spearman1_imp(target, pred),

                'MSE-gene': e.mse_gene(target, pred),
                'Pearson_gene': e.pearson_gene(target, pred),
                'Spearman_gene': e.spearman_gene(target, pred),

                'MSE-prom': e.mse_prom(target, pred),
                'Pearson_prom': e.pearson_prom(target, pred),
                'Spearman_prom': e.spearman_prom(target, pred),
            }
        

        results.append(metrics)

    results = pd.DataFrame(results)
    results.to_csv("eDICE_results.csv", index=False)

class Evaluation: # on chr21
    def __init__(
        self, model_path, hyper_parameters_path, 
        traindata_path, evaldata_path, version="16",
        resolution=25, chr_sizes_file="data/hg38.chrom.sizes", is_arcsin=True):

        self.traindata_path = traindata_path
        self.evaldata_path = evaldata_path
        self.is_arcsin = is_arcsin
        self.version = version

        with open(hyper_parameters_path, 'rb') as f:
            self.hyper_parameters = pickle.load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        loader = MODEL_LOADER(model_path, self.hyper_parameters)

        self.model = loader.load_epidenoise(version=self.version)

        print(f"# model_parameters: {count_parameters(self.model)}")

        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]
        self.model.eval()  # set the model to evaluation mode

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        self.chr_sizes = {}
        self.resolution = resolution

        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

        self.results = []

        self.train_data = {}
        self.eval_data = {}

        # load and bin chr21 of all bigwig files 
        for t in os.listdir(traindata_path):
            if ".bigwig" in t:

                for e in os.listdir(evaldata_path):
                    if ".bigwig" in e:
                        
                        if t[:3] == e[:3]:

                            if t[:3] not in self.train_data:
                                self.train_data[t[:3]] = {}

                            if e[:3] not in self.eval_data:
                                self.eval_data[e[:3]] = {}

                            self.train_data[t[:3]][t[3:6]] = traindata_path + "/" + t
                            self.eval_data[e[:3]][e[3:6]] = evaldata_path + "/" + e

        print(self.eval_data.keys())
        print(self.train_data.keys())

    def load_biosample(self, bios_name, mode="train"):
        chr, start, end = "chr21", 0, self.chr_sizes["chr21"]
        all_samples = []
        missing_ind = []

        if mode  == "train": 
            source = self.train_data
            savepath = self.traindata_path + f"/{bios_name}_chr21_{self.resolution}.pt"
        elif mode == "eval":
            source = self.eval_data
            savepath = self.evaldata_path + f"/{bios_name}_chr21_{self.resolution}.pt"
        
        if os.path.exists(savepath):
            all_samples = torch.load(savepath)
            # fill-in missing_ind
            for i in range(all_samples.shape[1]):
                if (all_samples[:, i] == -1).all():
                    missing_ind.append(i)
                    
            return all_samples, missing_ind

        else:
            for i in range(len(self.all_assays)):
                assay = self.all_assays[i]
                if assay in source[bios_name].keys():
                    print("loading ", assay)
                    bw = pyBigWig.open(source[bios_name][assay])
                    signals = bw.stats(chr, start, end, type="mean", nBins=(end - start) // self.resolution)
                
                else:
                    print(assay, "missing")
                    signals = [-1 for _ in range((end - start) // self.resolution)]
                    missing_ind.append(i)

            
                all_samples.append(signals)

            all_samples = torch.from_numpy(np.array(all_samples, dtype=np.float32)).transpose_(0, 1)

            # replace NaN with zero
            all_samples = torch.where(torch.isnan(all_samples), torch.zeros_like(all_samples), all_samples)

            nan_count = torch.isnan(all_samples).sum().item()
            minus_one_count = (all_samples == -1).sum().item()

            torch.save(all_samples, savepath)
            
            return all_samples, missing_ind
      
    def evaluate_biosample(self, bios_name):
        X, missing_x_i = self.load_biosample(bios_name, mode="train")
        Y, missing_y_i = self.load_biosample(bios_name, mode="eval")

        context_length, batch_size = self.hyper_parameters["context_length"], self.hyper_parameters["batch_size"]
        num_rows = (X.shape[0] // context_length) * context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]
        
        if self.is_arcsin:
            arcmask = (X != -1)
            X[arcmask] = torch.arcsinh_(X[arcmask])

        X = X.view(-1, context_length, X.shape[-1])
        Y = Y.view(-1, context_length, Y.shape[-1])

        d_model = X.shape[-1]

        if self.version == "10":
            fmask = torch.ones(d_model, self.hyper_parameters["d_model"])
            for i in missing_x_i: # input fmask
                fmask[i,:] = 0
            fmask = fmask.to(self.device)

        elif self.version == "16" or self.version == "17":
            CLS_x = torch.full((X.shape[0], 1, X.shape[2]), -2)
            SEP_x = torch.full((X.shape[0], 1, X.shape[2]), -3)
            CLS_y = torch.full((Y.shape[0], 1, Y.shape[2]), -2)
            SEP_y = torch.full((Y.shape[0], 1, Y.shape[2]), -3)

            X = torch.cat([CLS_x, X[:, :context_length//2, :], SEP_x, X[:, context_length//2:, :], SEP_x], dim=1)
            Y = torch.cat([CLS_y, Y[:, :context_length//2, :], SEP_y, Y[:, context_length//2:, :], SEP_y], dim=1)

            segment_label = [0] + [1 for i in range(context_length//2)] + [0] + [2 for i in range(context_length//2)] + [0]
            segment_label = torch.from_numpy(np.array(segment_label))
            segment_label = segment_label.to(self.device)

        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)
                if self.version == "10":
                    # (no position is masked)
                    pmask = torch.zeros((x_batch.shape[0], x_batch.shape[1]), dtype=torch.bool,  device=self.device)
                    outputs = self.model(x_batch, pmask, fmask)

                elif self.version == "16":
                    outputs, pred_mask, SAP = self.model(x_batch, segment_label)

                elif self.version == "17":
                    mask = torch.zeros_like(x_batch, dtype=torch.bool)
                    for i in missing_x_i: 
                        mask[:,:,i] = True

                    outputs, SAP = self.model(x_batch, ~mask, segment_label)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) # eval data
        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1]) # train data

        if self.is_arcsin:
            arcmask = (X != -1)
            P = torch.sinh_(P)
            X[arcmask] = torch.sinh_(X[arcmask])

        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()
                comparison = 'imputed'
            
            elif j not in missing_x_i:
                target = X[:, j].numpy()
                comparison = 'denoised'

            else:
                continue
            
            metrics = {
                'celltype': bios_name,
                'feature': self.all_assays[j],
                'comparison': comparison,
                'available train assays': len(self.all_assays) - len(missing_x_i),
                'available eval assays': len(self.all_assays) - len(missing_y_i),

                'MSE-GW': self.mse(target, pred),
                'Pearson-GW': self.pearson(target, pred),
                'Spearman-GW': self.spearman(target, pred),

                'MSE-1obs': self.mse1obs(target, pred),
                'Pearson_1obs': self.pearson1_obs(target, pred),
                'Spearman_1obs': self.spearman1_obs(target, pred),

                'MSE-1imp': self.mse1imp(target, pred),
                'Pearson_1imp': self.pearson1_imp(target, pred),
                'Spearman_1imp': self.spearman1_imp(target, pred),

                'MSE-gene': self.mse_gene(target, pred),
                'Pearson_gene': self.pearson_gene(target, pred),
                'Spearman_gene': self.spearman_gene(target, pred),

                'MSE-prom': self.mse_prom(target, pred),
                'Pearson_prom': self.pearson_prom(target, pred),
                'Spearman_prom': self.spearman_prom(target, pred),

                "peak_overlap_01thr": self.peak_overlap(target, pred, threshold=0.01),
                "peak_overlap_05thr": self.peak_overlap(target, pred, threshold=0.05),
                "peak_overlap_10thr": self.peak_overlap(target, pred, threshold=0.10)
            }
            self.results.append(metrics)
    
    def biosample_generate_imputations(self, bios_name, savedir="data/imputations/"):
        if os.path.exists(savedir) == False:
            os.mkdir(savedir)

        X, missing_x_i = self.load_biosample(bios_name, mode="train")
        Y, missing_y_i = self.load_biosample(bios_name, mode="eval")

        context_length, batch_size = self.hyper_parameters["context_length"], self.hyper_parameters["batch_size"]
        num_rows = (X.shape[0] // context_length) * context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        if self.is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])

        X = X.view(-1, context_length, X.shape[-1])
        Y = Y.view(-1, context_length, Y.shape[-1])

        d_model = X.shape[-1]

        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)
                mask = torch.zeros_like(x_batch, dtype=torch.bool, device=self.device)
                for ii in missing_x_i: 
                    mask[:,:,ii] = True
                mask = mask.to(self.device)

                if self.version == "10":
                    # (no position is masked)
                    pmask = torch.zeros((x_batch.shape[0], x_batch.shape[1]), dtype=torch.bool,  device=self.device)
                    outputs = self.model(x_batch, pmask, fmask)

                elif self.version == "16":
                    outputs, pred_mask, SAP = self.model(x_batch, segment_label)

                elif self.version == "17":
                    outputs, SAP = self.model(x_batch, ~mask, segment_label)
                
                elif self.version == "18":
                    outputs, pred_mask = self.model(x_batch)

                elif self.version == "20":
                    outputs, pred_mask = self.model(x_batch, mask)
                
                elif self.version == "21":
                    outputs, pred_mask = self.model(x_batch, mask)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        torch.save(P, savedir+ bios_name + "_imp.pt")

    def evaluate_model(self, outdir):
        for bios in self.eval_data.keys():
            print("evaluating ", bios)
            self.evaluate_biosample(bios)

        self.results = pd.DataFrame(self.results)
        self.results.to_csv(outdir, index=False)

    ################################################################################

    def get_gene_positions(self, chrom, bin_size):
        gene_df = pd.read_csv(PROC_GENE_BED_FPATH, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'gene_id', 'gene_name'])
        chrom_subset = gene_df[gene_df['chrom'] == chrom].copy()

        chrom_subset['start'] = (chrom_subset['start'] / bin_size).apply(lambda s: math.floor(s))
        chrom_subset['end'] = (chrom_subset['end'] / bin_size).apply(lambda s: math.floor(s))
        return chrom_subset

    def get_prom_positions(self, chrom, bin_size):
        prom_df = pd.read_csv(PROC_PROM_BED_PATH, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'gene_id', 'gene_name', "strand"])
        chrom_subset = prom_df[prom_df['chrom'] == chrom].copy()

        chrom_subset['start'] = (chrom_subset['start'] / bin_size).apply(lambda s: math.floor(s))
        chrom_subset['end'] = (chrom_subset['end'] / bin_size).apply(lambda s: math.floor(s))

        return chrom_subset

    def get_signals(self, array, df):
        signals = []
        for idx, row in df.iterrows():
            gene_bins = slice(row['start'], row['end'])
            signals += array[gene_bins].tolist()

        return signals

    ################################################################################

    def mse(self, y_true, y_pred):
        """
        Calculate the genome-wide Mean Squared Error (MSE). This is a measure of the average squared difference 
        between the true and predicted values across the entire genome at a resolution of 25bp.
        """
        return np.mean((np.array(y_true) - np.array(y_pred))**2)

    def mse_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=gene_df)
        pred_vals = self.get_signals(array=y_pred, df=gene_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=gene_df)
        pred_vals = self.get_signals(array=y_pred, df=gene_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=gene_df)
        pred_vals = self.get_signals(array=y_pred, df=gene_df)

        return self.spearman(y_true=gt_vals, y_pred=pred_vals)

    def pearson(self, y_true, y_pred):
        """
        Calculate the genome-wide Pearson Correlation. This measures the linear relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return pearsonr(y_pred, y_true)[0]

    def spearman(self, y_true, y_pred):
        """
        Calculate the genome-wide Spearman Correlation. This measures the monotonic relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return spearmanr(y_pred, y_true)[0]

    def mse_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=prom_df)
        pred_vals = self.get_signals(array=y_pred, df=prom_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=prom_df)
        pred_vals = self.get_signals(array=y_pred, df=prom_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=prom_df)
        pred_vals = self.get_signals(array=y_pred, df=prom_df)

        return self.spearman(y_true=gt_vals, y_pred=pred_vals)

    def mse1obs(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by experimental signal (mse1obs). 
        This is a measure of how well predictions match observations at positions with high experimental signal. 
        It's similar to recall.
        """
        top_1_percent = int(0.01 * len(y_true))
        top_1_percent_indices = np.argsort(y_true)[-top_1_percent:]
        return mean_squared_error(y_true[top_1_percent_indices], y_pred[top_1_percent_indices])

    def mse1imp(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by predicted signal (mse1imp). 
        This is a measure of how well predictions match observations at positions with high predicted signal. 
        It's similar to precision.
        """
        top_1_percent = int(0.01 * len(y_pred))
        top_1_percent_indices = np.argsort(y_pred)[-top_1_percent:]
        return mean_squared_error(y_true[top_1_percent_indices], y_pred[top_1_percent_indices])

    def pearson1_obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def spearman1_obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.spearman(y_true[perc_99_pos], y_pred[perc_99_pos])

    def pearson1_imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def spearman1_imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.spearman(y_true[perc_99_pos], y_pred[perc_99_pos])

    def peak_overlap(self, y_true, y_pred, p=0.01):
        top_p_percent = int(p * len(y_true))
        
        # Get the indices of the top p percent of the observed (true) values
        top_p_percent_obs_i = np.argsort(y_true)[-top_p_percent:]
        
        # Get the indices of the top p percent of the predicted values
        top_p_percent_pred_i = np.argsort(y_pred)[-top_p_percent:]

        # Calculate the overlap
        overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))

        # Calculate the percentage of overlap
        self.overlap_percent = overlap / top_p_percent 

        return self.overlap_percent

def check_poisson_vs_nbinom(data, assay_name):
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    # Fit Negative Binomial Distribution
    def negbinom_loglik(params, data):
        r, p = params
        return -np.sum(stats.nbinom.logpmf(data, r, p))

    initial_params = [1, 0.5]  # Initial guess for r and p
    result_nbinom = minimize(negbinom_loglik, initial_params, args=(data), bounds=[(1e-5, None), (1e-5, 1-1e-5)])
    r, p = result_nbinom.x

    # Fit Poisson Distribution
    lambda_poisson = np.mean(data)

    # Calculate Log-Likelihoods
    log_likelihood_nbinom = -negbinom_loglik([r, p], data)
    log_likelihood_poisson = np.sum(stats.poisson.logpmf(data, lambda_poisson))

    # Calculate AIC and BIC
    def aic_bic(log_likelihood, num_params, num_samples):
        aic = 2 * num_params - 2 * log_likelihood
        bic = num_params * np.log(num_samples) - 2 * log_likelihood
        return aic, bic

    aic_nbinom, bic_nbinom = aic_bic(log_likelihood_nbinom, 2, len(data))
    aic_poisson, bic_poisson = aic_bic(log_likelihood_poisson, 1, len(data))

    print(f"Negative Binomial - AIC: {aic_nbinom}, BIC: {bic_nbinom}")
    print(f"Poisson - AIC: {aic_poisson}, BIC: {bic_poisson}")

    # Plot the fit
    x = np.arange(0, max(data)+1)
    plt.hist(data, bins=x-0.5, density=True, alpha=0.6, color='g', label='Data')

    plt.plot(x, stats.nbinom.pmf(x, r, p), 'o-', label=f'Negative Binomial (r={r:.2f}, p={p:.2f})')
    plt.plot(x, stats.poisson.pmf(x, lambda_poisson), 'o-', label=f'Poisson ($\lambda$={lambda_poisson:.2f})')

    plt.legend()
    plt.xlabel('Data')
    plt.ylabel('Frequency')
    plt.title('Fit Comparison')
    plt.savefig(f"models/evals/examples/{assay_name}", dpi=150)

class EpiDenoise30b_old(nn.Module):
    def __init__(self, 
        input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, n_decoder_layers,
        dropout=0.1, context_length=2000, pos_enc="relative"):
        super(EpiDenoise30b_old, self).__init__()
        self.pos_enc = "abs"#pos_enc

        conv_out_channels = exponential_linspace_int(
            d_model//n_cnn_layers, d_model, n_cnn_layers, divisible_by=2)

        stride = 1
        dilation=1
        self.context_length = context_length
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.metadata_embedder = MetadataEmbeddingModule(input_dim, embedding_dim=metadata_embedding_dim, non_linearity=True)
        self.lin = nn.Linear(input_dim + metadata_embedding_dim, d_model)

        self.signal_layer_norm = nn.LayerNorm(input_dim)
        self.embedd_layer_norm = nn.LayerNorm(d_model)

        self.conv0 = ConvTower(
                input_dim + metadata_embedding_dim, conv_out_channels[0],
                conv_kernel_size[0], stride, dilation, 
                pool_type="max", residuals=True)

        self.convtower = nn.ModuleList([ConvTower(
                conv_out_channels[i], conv_out_channels[i + 1],
                conv_kernel_size[i + 1], stride, dilation, 
                pool_type="max", residuals=True
            ) for i in range(n_cnn_layers - 1)])

        if self.pos_enc == "relative":
            self.encoder_layer = RelativeEncoderLayer(
                d_model=d_model, heads=nhead, feed_forward_hidden=4*d_model, dropout=dropout)

            self.decoder_layer = RelativeDecoderLayer(
                hid_dim=d_model, n_heads=nhead, pf_dim=4*d_model, dropout=dropout)
        else:
            self.position = PositionalEncoding(d_model, dropout, context_length)
            # self.enc_position = AbsPositionalEmbedding15(d_model=d_model, max_len=self.context_length//(2**n_cnn_layers))
            # self.dec_position = AbsPositionalEmbedding15(d_model=d_model, max_len=self.context_length)

            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.ModuleList(
            [self.encoder_layer for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList(
            [self.decoder_layer for _ in range(n_decoder_layers)])
        
        self.neg_binom_layer = NegativeBinomialLayer(d_model, output_dim)
        self.mask_pred_layer = nn.Linear(d_model, output_dim)
        self.mask_obs_layer = nn.Linear(d_model, output_dim)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)
        md_embedding = md_embedding.unsqueeze(1).expand(-1, self.context_length, -1)

        md_embedding = F.relu(md_embedding)
        src = self.signal_layer_norm(src)

        src = F.relu(torch.cat([src, md_embedding], dim=-1)) # N, L, F

        ### CONV ENCODER ###

        e_src = src.permute(0, 2, 1) # to N, F, L
        e_src = self.conv0(e_src)
        for conv in self.convtower:
            e_src = conv(e_src)
        e_src = e_src.permute(0, 2, 1)  # to N, L, F

        ### TRANSFORMER ENCODER ###
        if self.pos_enc != "relative":
            # encpos = torch.permute(self.enc_position(src), (1, 0, 2)) # to N, L, F
            e_src = self.position(e_src)
            # e_src = e_src + encpos

        for enc in self.transformer_encoder:
            e_src = enc(e_src)
        
        src = F.relu(self.embedd_layer_norm(self.lin(src)))

        ### TRANSFORMER DECODER ###
        if self.pos_enc != "relative":
            # trgpos = torch.permute(self.dec_position(src), (1, 0, 2))
            src = self.position(src)
            # src = src + trgpos

        for dec in self.transformer_decoder:
            src = dec(src, e_src)

        p, n = self.neg_binom_layer(src)
        mp = torch.sigmoid(self.mask_pred_layer(src))
        mo = torch.sigmoid(self.mask_obs_layer(src))

        return p, n, mp, mo

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

def fit_negative_binomial(data):
    # Estimate the parameters of the negative binomial distribution
    def negative_binomial_log_likelihood(params):
        n, p = params
        return -np.sum(stats.nbinom.logpmf(data, n, p))
    # Initial guess for n and p
    mean = np.mean(data)
    var = np.var(data)
    p_initial = mean / var
    n_initial = mean * p_initial / (1 - p_initial)
    # Bounds and constraints for parameters
    bounds = [(1e-5, None), (1e-5, 1-1e-5)]  # n > 0, 0 < p < 1
    # Minimize the negative log-likelihood
    result = minimize(negative_binomial_log_likelihood, [n_initial, p_initial], bounds=bounds, method='L-BFGS-B')
    n, p = result.x
    return n, p

def load_and_fit_nb(file_path):
    # Load the array from the npz file
    data = np.load(file_path)['arr_0']
    # Ensure data is integer
    data = np.round(data).astype(int)
    # Fit negative binomial distribution
    n, p = fit_negative_binomial(data)
    return n, p

class DeconvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D):
        super(DeconvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_C)
        padding = W // 2
        output_padding = 1 
        self.deconv = nn.ConvTranspose1d(
            in_C, out_C, kernel_size=W, dilation=D, stride=S, 
            padding=padding, output_padding=output_padding)
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.deconv(x)
        x = F.gelu(x)
        return x

class MONITOR_VALIDATION(object):
    def __init__(
        self, data_path, context_length, batch_size,
        chr_sizes_file="data/hg38.chrom.sizes", 
        resolution=25, split="val", arch="a", 
        token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}, eic=False):

        self.data_path = data_path
        self.context_length = context_length
        self.batch_size = batch_size
        self.resolution = resolution
        self.arch = arch
        self.eic = eic

        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=10, eic=eic)

        self.mark_dict = {v: k for k, v in self.dataset.aliases["experiment_aliases"].items()}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.example_coords = [
            (33481539//self.resolution, 33588914//self.resolution), # GART
            (25800151//self.resolution, 26235914//self.resolution), # APP
            (31589009//self.resolution, 31745788//self.resolution), # SOD1
            (39526359//self.resolution, 39802081//self.resolution), # B3GALT5
            (33577551//self.resolution, 33919338//self.resolution), # ITSN1
            (36260000//self.resolution, 36450000//self.resolution), # RUNX1
            (45000000//self.resolution, 45250000//self.resolution), # COL18A1
            (36600000//self.resolution, 36850000//self.resolution), # MX1
            (39500000//self.resolution, 40000000//self.resolution) # Highly Conserved Non-Coding Sequences (HCNS)
            ]

        self.token_dict = token_dict
        
        if self.arch in ["c"]:
            self.token_dict["cloze_mask"] = self.token_dict["missing_mask"]

        self.chr_sizes = {}
        self.metrics = METRICS()
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)
    
    def pred(self, X, mX, mY, avail, imp_target=[]):
        # print("making preds")
        # Initialize a tensor to store all predictions
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.clone()
                avail_batch = avail_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()

                x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])
                # mY_batch_missing_vals = (mY_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]
                # mY_batch[mY_batch_missing_vals] = self.token_dict["cloze_mask"]
                if self.arch in ["a", "b"]:
                    avail_batch[avail_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    # mY_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    if self.arch in ["a", "b"]:
                        avail_batch[:, imp_target] = self.token_dict["cloze_mask"]
                    elif self.arch in ["c", "d"]:
                        avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.arch in ["a", "b", "d"]:
                    outputs_p, outputs_n, _, _ = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)
                elif self.arch in ["c"]:
                    outputs_p, outputs_n = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)

            # Store the predictions in the large tensor
            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n  # Free up memory
            torch.cuda.empty_cache()  # Free up GPU memory

        return n, p

    def get_bios(self, bios_name, x_dsf=1, y_dsf=1):
        print(f"getting bios vals for {bios_name}")
        temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx
        
        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        subsets_X = []
        subsets_Y = []

        for start, end in self.example_coords:
            segment_length = end - start
            adjusted_length = (segment_length // self.context_length) * self.context_length
            adjusted_end = start + adjusted_length

            subsets_X.append(X[start:adjusted_end, :])
            subsets_Y.append(Y[start:adjusted_end, :])

        # Concatenate the subsets along the sequence length dimension (second dimension)
        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        for leave_one_out in available_indices:
            n, p = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])
            
            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]
            # print(f"got imputations for feature #{leave_one_out+1}")
            del n, p  # Free up memory
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        del X, mX, mY, avX, avY  # Free up memoryrm m
        # print("got upsampled")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        imp_dist = NegativeBinomial(p_imp, n_imp)
        ups_dist = NegativeBinomial(p_ups, n_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])

        return imp_dist, ups_dist, Y, bios_name, available_indices
    
    def get_bios_eic(self, bios_name, x_dsf=1, y_dsf=1):
        print(f"getting bios vals for {bios_name}")

        temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf, eic=True)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx
        
        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        # mY = self.dataset.fill_in_y_prompt(mY)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        subsets_X = []
        subsets_Y = []

        for start, end in self.example_coords:
            segment_length = end - start
            adjusted_length = (segment_length // self.context_length) * self.context_length
            adjusted_end = start + adjusted_length

            subsets_X.append(X[start:adjusted_end, :])
            subsets_Y.append(Y[start:adjusted_end, :])

        # Concatenate the subsets along the sequence length dimension (second dimension)
        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        available_X_indices = torch.where(avX[0, :] == 1)[0]
        available_Y_indices = torch.where(avY[0, :] == 1)[0]
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        # del X, mX, mY, avX, avY  # Free up memoryrm m

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        ups_dist = NegativeBinomial(p_ups, n_ups)

        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1])
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])

        return ups_dist, Y, X, bios_name, available_X_indices, available_Y_indices

    def get_frame(self, bios_name, x_dsf=1, y_dsf=1):
        temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx

        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        subsets_X = []
        subsets_Y = []

        start, end = 33481539//self.resolution, 33588914//self.resolution
        segment_length = end - start
        adjusted_length = (segment_length // self.context_length) * self.context_length
        adjusted_end = start + adjusted_length

        subsets_X.append(X[start:adjusted_end, :])
        subsets_Y.append(Y[start:adjusted_end, :])

        # Concatenate the subsets along the sequence length dimension (second dimension)
        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        for leave_one_out in available_indices:
            n, p = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])
            
            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]
            # print(f"got imputations for feature #{leave_one_out+1}")
            del n, p  # Free up memory
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        del X, mX, mY, avX, avY  # Free up memoryrm m
        # print("got upsampled")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        imp_dist = NegativeBinomial(p_imp, n_imp)
        ups_dist = NegativeBinomial(p_ups, n_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])

        return imp_dist, ups_dist, Y, bios_name, available_indices

    def get_metric(self, imp_dist, ups_dist, Y, bios_name, availability):
        # print(f"getting metrics")
        imp_mean = imp_dist.expect()
        ups_mean = ups_dist.expect()

        # print(f"got nbinom stuff")
        # imp_lower_95, imp_upper_95 = imp_dist.interval(confidence=0.95)
        # ups_lower_95, ups_upper_95 = ups_dist.interval(confidence=0.95)
        
        results = []
        # for j in availability:  # for each feature i.e. assay
        for j in range(Y.shape[1]):

            if j in list(availability):
                # j = j.item()
                for comparison in ['imputed', 'upsampled']:
                    if comparison == "imputed":
                        pred = imp_mean[:, j].numpy()
                        # lower_95 = imp_lower_95[:, j].numpy()
                        # upper_95 = imp_upper_95[:, j].numpy()
                        
                    elif comparison == "upsampled":
                        pred = ups_mean[:, j].numpy()
                        # lower_95 = ups_lower_95[:, j].numpy()
                        # upper_95 = ups_upper_95[:, j].numpy()

                    target = Y[:, j].numpy()

                    # Check if the target values fall within the intervals
                    # within_interval = (target >= lower_95) & (target <= upper_95)
                    
                    # Calculate the fraction
                    # print(
                    #     f"adding {bios_name} | {self.mark_dict[f'M{str(j+1).zfill(len(str(len(self.mark_dict))))}']} | {comparison}")
                    metrics = {
                        'bios':bios_name,
                        'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                        'comparison': comparison,
                        'available assays': len(availability),

                        'MSE': self.metrics.mse(target, pred),
                        'Pearson': self.metrics.pearson(target, pred),
                        'Spearman': self.metrics.spearman(target, pred),
                        'r2': self.metrics.r2(target, pred)
                    }
                    results.append(metrics)

        return results
    
    def get_metric_eic(self, ups_dist, Y, X, bios_name, availability_X, availability_Y):
        ups_mean = ups_dist.expect()
        
        results = []
        for j in range(Y.shape[1]):
            pred = ups_mean[:, j].numpy()
            if j in list(availability_X):
                comparison = "upsampled"
                target = X[:, j].numpy()

                metrics = {
                    'bios':bios_name,
                    'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                    'comparison': comparison,
                    'available assays': len(availability_X),

                    'MSE': self.metrics.mse(target, pred),
                    'Pearson': self.metrics.pearson(target, pred),
                    'Spearman': self.metrics.spearman(target, pred),
                    'r2': self.metrics.r2(target, pred)
                }
                results.append(metrics)
                
            elif j in list(availability_Y):
                comparison = "imputed"
                target = Y[:, j].numpy()

                metrics = {
                    'bios':bios_name,
                    'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                    'comparison': comparison,
                    'available assays': len(availability_X),

                    'MSE': self.metrics.mse(target, pred),
                    'Pearson': self.metrics.pearson(target, pred),
                    'Spearman': self.metrics.spearman(target, pred),
                    'r2': self.metrics.r2(target, pred)
                }
                results.append(metrics)

        return results

    def get_validation(self, model, x_dsf=1, y_dsf=1):
        t0 = datetime.datetime.now()
        self.model = model

        full_res = []
        bioses = list(self.dataset.navigation.keys())
        if not self.eic:
            for bios_name in bioses:
                try:
                    imp_dist, ups_dist, Y, _, available_indices = self.get_bios(bios_name, x_dsf=x_dsf, y_dsf=y_dsf)
                    full_res += self.get_metric(imp_dist, ups_dist, Y, bios_name, available_indices)
                    del imp_dist, ups_dist, Y
                except:
                    pass
        else:
            for bios_name in bioses:
                # try:
                ups_dist, Y, X, bios_name, available_X_indices, available_Y_indices = self.get_bios_eic(bios_name, x_dsf=x_dsf, y_dsf=y_dsf)
                full_res += self.get_metric_eic(ups_dist, Y, X, bios_name, available_X_indices, available_Y_indices)
                del ups_dist, Y, X
                # except:
                #     pass

        del self.model
        df = pd.DataFrame(full_res)

        # Separate the data based on comparison type
        imputed_df = df[df['comparison'] == 'imputed']
        upsampled_df = df[df['comparison'] == 'upsampled']

        # Function to calculate mean, min, and max for a given metric
        def calculate_stats(df, metric):
            return df[metric].mean(), df[metric].min(), df[metric].max()

        # Imputed statistics
        imp_mse_stats = calculate_stats(imputed_df, 'MSE')
        imp_pearson_stats = calculate_stats(imputed_df, 'Pearson')
        imp_spearman_stats = calculate_stats(imputed_df, 'Spearman')
        imp_r2_stats = calculate_stats(imputed_df, 'r2')
        # imp_frac95conf_stats = calculate_stats(imputed_df, 'frac_95_confidence')

        # Upsampled statistics
        ups_mse_stats = calculate_stats(upsampled_df, 'MSE')
        ups_pearson_stats = calculate_stats(upsampled_df, 'Pearson')
        ups_spearman_stats = calculate_stats(upsampled_df, 'Spearman')
        ups_r2_stats = calculate_stats(upsampled_df, 'r2')
        # ups_frac95conf_stats = calculate_stats(upsampled_df, 'frac_95_confidence')

        elapsed_time = datetime.datetime.now() - t0
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Create the compact print statement
        print_statement = f"""
        Took {int(minutes)}:{int(seconds)}
        For Imputed:
        - MSE: mean={imp_mse_stats[0]:.2f}, min={imp_mse_stats[1]:.2f}, max={imp_mse_stats[2]:.2f}
        - PCC: mean={imp_pearson_stats[0]:.2f}, min={imp_pearson_stats[1]:.2f}, max={imp_pearson_stats[2]:.2f}
        - SRCC: mean={imp_spearman_stats[0]:.2f}, min={imp_spearman_stats[1]:.2f}, max={imp_spearman_stats[2]:.2f}
        - R2: mean={imp_r2_stats[0]:.2f}, min={imp_r2_stats[1]:.2f}, max={imp_r2_stats[2]:.2f}

        For Upsampled:
        - MSE: mean={ups_mse_stats[0]:.2f}, min={ups_mse_stats[1]:.2f}, max={ups_mse_stats[2]:.2f}
        - PCC: mean={ups_pearson_stats[0]:.2f}, min={ups_pearson_stats[1]:.2f}, max={ups_pearson_stats[2]:.2f}
        - SRCC: mean={ups_spearman_stats[0]:.2f}, min={ups_spearman_stats[1]:.2f}, max={ups_spearman_stats[2]:.2f}
        - R2: mean={ups_r2_stats[0]:.2f}, min={ups_r2_stats[1]:.2f}, max={ups_r2_stats[2]:.2f}
        """

        return print_statement

    def generate_training_gif_frame(self, model, fig_title):
        def gen_subplt(
            ax, x_values, observed_values, 
            ups11, ups21, ups41, 
            imp11, imp21, imp41, 
            col, assname, ytick_fontsize=6, title_fontsize=6):

            # Define the data and labels
            data = [
                (observed_values, "Observed", "royalblue", f"{assname}_Observed"),
                (ups11, "Upsampled 1->1", "darkcyan", f"{assname}_Ups1->1"),
                (imp11, "Imputed 1->1", "salmon", f"{assname}_Imp1->1"),
                (ups21, "Upsampled 2->1", "darkcyan", f"{assname}_Ups2->1"),
                (imp21, "Imputed 2->1", "salmon", f"{assname}_Imp2->1"),
                (ups41, "Upsampled 4->1", "darkcyan", f"{assname}_Ups4->1"),
                (imp41, "Imputed 4->1", "salmon", f"{assname}_Imp4->1"),
            ]
            
            for i, (values, label, color, title) in enumerate(data):
                ax[i, col].plot(x_values, values, "--" if i != 0 else "-", color=color, alpha=0.7, label=label, linewidth=0.01)
                ax[i, col].fill_between(x_values, 0, values, color=color, alpha=0.7)
                
                if i != len(data)-1:
                    ax[i, col].tick_params(axis='x', labelbottom=False)
                
                ax[i, col].tick_params(axis='y', labelsize=ytick_fontsize)
                ax[i, col].set_xticklabels([])
                ax[i, col].set_title(title, fontsize=title_fontsize)

        self.model = model

        bios = list(self.dataset.navigation.keys())[0]
        # print(bios)

        # dsf4-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=4, y_dsf=1)
        imp_mean41, ups_mean41 = imp_dist.expect(), ups_dist.expect()

        # dsf2-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=2, y_dsf=1)
        imp_mean21, ups_mean21 = imp_dist.expect(), ups_dist.expect()

        # dsf1-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=1, y_dsf=1)
        imp_mean11, ups_mean11 = imp_dist.expect(), ups_dist.expect()

        del self.model

        selected_assays = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K9me3", "CTCF", "DNase-seq", "ATAC-seq"]
        available_selected = []
        for col, jj in enumerate(available_indices):
            assay = self.mark_dict[f"M{str(jj.item()+1).zfill(len(str(len(self.mark_dict))))}"]
            if assay in selected_assays:
                available_selected.append(jj)

        fig, axes = plt.subplots(7, len(available_selected), figsize=(len(available_selected) * 3, 6), sharex=True, sharey=False)
        
        for col, jj in enumerate(available_selected):
            j = jj.item()
            assay = self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"]
            x_values = list(range(len(Y[:, j])))

            obs = Y[:, j].numpy()

            gen_subplt(axes, x_values, 
                    obs, 
                    ups_mean11[:, j].numpy(), ups_mean21[:, j].numpy(), ups_mean41[:, j].numpy(), 
                    imp_mean11[:, j].numpy(), imp_mean21[:, j].numpy(), imp_mean41[:, j].numpy(), 
                    col, assay)

        fig.suptitle(fig_title, fontsize=10)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        return buf


    # e = EVAL_EED(
    #     model="/project/compbio-lab/EPD/pretrained/EPD30d_model_checkpoint_Jul8th.pth", 
    #     data_path="/project/compbio-lab/encode_data/", 
    #     context_length=3200, batch_size=50, 
    #     hyper_parameters_path="/project/compbio-lab/EPD/pretrained/hyper_parameters30d_EpiDenoise30d_20240710133714_params237654660.pkl",
    #     train_log={}, chr_sizes_file="data/hg38.chrom.sizes", 
    #     version="30d", resolution=25, savedir="/project/compbio-lab/EPD/eval_30d/", mode="eval")
    
    # print(e.bios_pipeline("ENCBS343AKO", x_dsf=1))


class SyntheticData:
    def __init__(self, n, p, num_features, sequence_length):
        self.n = n
        self.p = p
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.transformations = [
            (self.transform_scale, {'scale': 2}),
            (self.transform_exponential, {'base': 1.03, 'scale': 3}),
            (self.transform_log_scale, {'scale': 40}),
            (self.transform_sqrt_scale, {'scale': 15}),
            (self.transform_piecewise_linear, {'scale_factors': [1.5, 50, 0.5, 100, 0.3]}),
            (self.transform_scaled_sin, {'scale': 30}),
            (self.transform_scaled_cos, {'scale': 30}),
            (self.transform_hyperbolic_sinh, {'scale': 10}),
            (self.transform_polynomial, {'scale': 0.02, 'power': 2}),
            (self.transform_exponential, {'base': 1.05, 'scale': 2}),
            (self.transform_log_scale, {'scale': 60}),
            (self.transform_sqrt_scale, {'scale': 25}),
            (self.transform_piecewise_linear, {'scale_factors': [2, 60, 0.4, 110, 0.2]}),
            (self.transform_scaled_sin, {'scale': 60}),
            (self.transform_scaled_cos, {'scale': 60}),
            (self.transform_hyperbolic_sinh, {'scale': 15}),
            (self.transform_polynomial, {'scale': 0.005, 'power': 3}),
            (self.transform_exponential, {'base': 1.05, 'scale': 1}),
            (self.transform_log_scale, {'scale': 20}),
            (self.transform_sqrt_scale, {'scale': 15}),
            (self.transform_piecewise_linear, {'scale_factors': [2.5, 40, 0.3, 120, 0.25]}),
            (self.transform_scaled_sin, {'scale': 20}),
            (self.transform_scaled_cos, {'scale': 20}),
            (self.transform_hyperbolic_sinh, {'scale': 20}),
            (self.transform_polynomial, {'scale': 0.01, 'power': 3})
        ]

    def generate_base_sequence(self):
        self.base_sequence = np.random.negative_binomial(self.n, self.p, self.sequence_length)
        return self.base_sequence

    def transform_scale(self, sequence, scale):
        return np.clip(sequence * scale, 0, 200)

    def transform_exponential(self, sequence, base, scale):
        return np.clip(np.round(scale * (base ** (sequence))), 0, 1000)

    def transform_log_scale(self, sequence, scale):
        return np.clip(np.round(scale * np.log1p(sequence)), 0, 1000)

    def transform_sqrt_scale(self, sequence, scale):
        return np.clip(np.round(scale * np.sqrt(sequence)), 0, 1000)

    def transform_piecewise_linear(self, sequence, scale_factors):
        transformed = np.zeros_like(sequence)
        for i, value in enumerate(sequence):
            if value < 50:
                transformed[i] = value * scale_factors[0]
            elif value < 150:
                transformed[i] = scale_factors[1] + scale_factors[2] * value
            else:
                transformed[i] = scale_factors[3] + scale_factors[4] * value
        return np.clip(np.round(transformed), 0, 1000)

    def transform_scaled_sin(self, sequence, scale):
        return np.clip(np.round(scale * np.abs(np.sin(sequence))), 0, 1000)

    def transform_scaled_cos(self, sequence, scale):
        return np.clip(np.round(scale * np.abs(np.cos(sequence))), 0, 1000)

    def transform_hyperbolic_sinh(self, sequence, scale):
        return np.clip(np.round(scale * np.sinh(sequence / 50)), 0, 1000)

    def transform_polynomial(self, sequence, scale, power):
        return np.clip(np.round(scale * (sequence ** power)), 0, 1000)

    def apply_transformations(self):
        transformed_sequences = []
        for i in range(self.num_features):
            transform, params = self.transformations[i % len(self.transformations)]
            transformed_seq = transform(self.base_sequence, **params)
            transformed_sequences.append((transformed_seq, transform.__name__))

        return transformed_sequences

    def smooth_sequence(self, sequence, sigma=0.001):
        return gaussian_filter1d(sequence, sigma=sigma)

    def apply_smoothing(self, sequences):
        return [self.smooth_sequence(seq).astype(int) for seq, name in sequences]
    
    def synth_metadata(self, sequences):
        def depth(seq):
            return np.log2((3e9*np.sum(seq))/len(seq))
        def coverage(seq):
            return 100 * np.abs(np.sin(np.sum(seq)))
        def read_length(seq):
            return np.log10(np.sum(seq)+1)
        def run_type(seq):
            if np.mean(seq) <= np.median(seq):
                return 1
            else:
                return 0

        return [np.array([depth(seq), coverage(seq), read_length(seq), run_type(seq)]) for seq, name in sequences]

    def miss(self, sequences, metadata, missing_percentage):
        to_miss = random.choices(range(self.num_features), k=int(self.num_features*missing_percentage))
        avail = [1 for i in range(self.num_features)]

        for miss in to_miss:
            sequences[miss, :] = -1
            metadata[miss, :] = -1
            avail[miss] = 0
        
        return sequences, metadata, avail

    def mask(self, sequences, metadata, avail, mask_percentage):
        to_mask = random.choices([x for x in range(self.num_features) if avail[x]==1], k=int(self.num_features*mask_percentage))

        for mask in to_mask:
            sequences[mask, :] = -2
            metadata[mask, :] = -2
            avail[mask] = -2

        return sequences, metadata, avail

    def get_batch(self, batch_size, miss_perc_range=(0.3, 0.9), mask_perc_range=(0.1, 0.2)):
        batch_X, batch_Y = [], []
        md_batch_X, md_batch_Y = [], []
        av_batch_X, av_batch_Y = [], []
        
        for b in range(batch_size):
            self.generate_base_sequence()
            transformed_sequences = self.apply_transformations()

            smoothed_sequences = self.apply_smoothing(transformed_sequences)
            smoothed_sequences = np.array(smoothed_sequences)

            syn_metadata = self.synth_metadata(transformed_sequences)
            syn_metadata = np.array(syn_metadata)

            miss_p_b = random.uniform(miss_perc_range[0], miss_perc_range[1])
            mask_p_b = random.uniform(mask_perc_range[0], mask_perc_range[1])
            
            y_b, ymd_b, yav_b = self.miss(smoothed_sequences, syn_metadata, miss_p_b)
            x_b, xmd_b, xav_b = self.mask(y_b.copy(), ymd_b.copy(), yav_b.copy(), mask_p_b)

            batch_X.append(x_b)
            batch_Y.append(y_b)

            md_batch_X.append(xmd_b)
            md_batch_Y.append(ymd_b)

            av_batch_X.append(xav_b)
            av_batch_Y.append(yav_b)
        
        batch_X, batch_Y = torch.Tensor(np.array(batch_X)).permute(0, 2, 1), torch.Tensor(np.array(batch_Y)).permute(0, 2, 1)
        md_batch_X, md_batch_Y = torch.Tensor(np.array(md_batch_X)).permute(0, 2, 1), torch.Tensor(np.array(md_batch_Y)).permute(0, 2, 1)
        av_batch_X, av_batch_Y = torch.Tensor(np.array(av_batch_X)), torch.Tensor(np.array(av_batch_Y))

        return batch_X, batch_Y, md_batch_X, md_batch_Y, av_batch_X, av_batch_Y
    

    def __new_epoch(self):
        self.current_bios_batch_pointer = 0
        self.current_loci_batch_pointer = 0
    
    def __update_batch_pointers(self, cycle_biosamples_first=True):
        if cycle_biosamples_first:
            # Cycle through all biosamples for each loci before moving to the next loci
            if self.current_bios_batch_pointer + self.bios_batchsize >= self.num_bios:
                self.current_bios_batch_pointer = 0
                if self.current_loci_batch_pointer + self.loci_batchsize < self.num_regions:
                    self.current_loci_batch_pointer += self.loci_batchsize
                else:
                    self.current_loci_batch_pointer = 0  # Reset loci pointer after the last one
                    return True
            else:
                self.current_bios_batch_pointer += self.bios_batchsize
        else:
            # Cycle through all loci for each batch of biosamples before moving to the next batch of biosamples
            if self.current_loci_batch_pointer + self.loci_batchsize >= self.num_regions:
                self.current_loci_batch_pointer = 0
                if self.current_bios_batch_pointer + self.bios_batchsize < self.num_bios:
                    self.current_bios_batch_pointer += self.bios_batchsize
                else:
                    self.current_bios_batch_pointer = 0  # Reset biosample pointer after the last one
                    return True
            else:
                self.current_loci_batch_pointer += self.loci_batchsize

        return False

    def __get_batch(self, dsf):
        batch_loci_list = self.m_regions[self.current_loci_batch_pointer : self.current_loci_batch_pointer+self.loci_batchsize]
        batch_bios_list = list(self.navigation.keys())[self.current_bios_batch_pointer : self.current_bios_batch_pointer+self.bios_batchsize]
        
        batch_data = []
        batch_metadata = []
        batch_availability = []

        for locus in batch_loci_list:
            self.make_region_tensor
            d, md, avl = self.__make_region_tensor(batch_bios_list, locus, DSF=dsf)
            batch_data.append(d)
            batch_metadata.append(md)
            batch_availability.append(avl)
        
        batch_data, batch_metadata, batch_availability = torch.concat(batch_data), torch.concat(batch_metadata), torch.concat(batch_availability)
        return batch_data, batch_metadata, batch_availability


    
    def __make_region_tensor(self, list_bios, locus, DSF, max_workers=-1):
        """Load and process data for multiple biosamples in parallel."""
        def load_and_process(bios):
            try:
                loaded_data, loaded_metadata = self.load_bios(bios, locus, DSF)
                return self.make_bios_tensor(loaded_data, loaded_metadata)
            except Exception as e:
                print(f"Failed to process {bios}: {e}")
                return None

        if max_workers == -1:
            max_workers = self.bios_batchsize//2

        # Use ThreadPoolExecutor to handle biosamples in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(load_and_process, list_bios))

        # Aggregate results
        data, metadata, availability = [], [], []
        for result in results:
            if result is not None:
                d, md, avl = result
                data.append(d)
                metadata.append(md)
                availability.append(avl)

        data, metadata, availability = torch.stack(data), torch.stack(metadata), torch.stack(availability)
        return data, metadata, availability



class EVAL_EED(object):
    """
    for imputating missing tracks, we should replace mY with 'prompt' metadata.
    prompt = [24, ~max_assay_genome_coverage, ~max_assay_read_length, pair-end]
    """
    def __init__(
        self, model, data_path, context_length, batch_size, hyper_parameters_path="",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", version="30a", resolution=25, 
        savedir="models/evals/", mode="eval", split="test"):

        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.mkdir(self.savedir)

        self.data_path = data_path
        self.version = version
        self.context_length = context_length
        self.batch_size = batch_size
        self.resolution = resolution

        self.model = model
        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=5)

        self.mark_dict = {v: k for k, v in self.dataset.aliases["experiment_aliases"].items()}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.token_dict = {
                    "missing_mask": -1, 
                    "cloze_mask": -2,
                    "pad": -3
                }

        self.chr_sizes = {}
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

        self.train_data = {}
        self.eval_data = {}
        self.metrics = METRICS()
        self.viz = VISUALS(resolution=self.resolution, savedir=self.savedir)

        self.gene_coords = load_gene_coords("data/parsed_genecode_data_hg38_release42.csv")
        self.gene_coords = self.gene_coords[self.gene_coords["chr"] == "chr21"].reset_index(drop=True)

        if mode == "dev":
            return

        if type(self.model) == str:
            with open(hyper_parameters_path, 'rb') as f:
                self.hyper_parameters = pickle.load(f)
            loader = MODEL_LOADER(model, self.hyper_parameters)
            self.model = loader.load_epidenoise(version=self.version)

        self.model = self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode
        print(f"# model_parameters: {count_parameters(self.model)}")

    def eval_rnaseq(self, bios_name, y_pred, y_true, availability, k_fold=10, plot_REC=True):
        # columns=  chr, start, end, geneID, length, TPM, FPKM
        rna_seq_data = self.dataset.load_rna_seq_data(bios_name, self.gene_coords) 
        print(rna_seq_data)
        
        pred_features = []
        true_features = []
        available_assays = [self.mark_dict[f"M{str(a+1).zfill(len(str(len(self.mark_dict))))}"] for a in range(y_pred.shape[1]) if a in list(availability)]
        print(available_assays)
        
        for i in range(len(rna_seq_data)):
            for a in range(y_pred.shape[1]):
                assay_name = self.mark_dict[f"M{str(a+1).zfill(len(str(len(self.mark_dict))))}"]

                if a in list(availability):
                    true_signal_a = y_true[:, a].numpy()
                    f = signal_feature_extraction(
                        rna_seq_data["start"][i], rna_seq_data["end"][i], 
                        rna_seq_data["strand"][i], true_signal_a
                        )

                    f = [assay_name, rna_seq_data["geneID"][i], f["mean_sig_promoter"], f["mean_sig_gene_body"], 
                        f["mean_sig_around_TES"], rna_seq_data["TPM"][i], rna_seq_data["FPKM"][i]]

                    true_features.append(f)
                
                pred_signal_a = y_pred[:, a].numpy()
                f = signal_feature_extraction(
                        rna_seq_data["start"][i], rna_seq_data["end"][i], 
                        rna_seq_data["strand"][i], pred_signal_a
                        )
                    
                f = [assay_name, rna_seq_data["geneID"][i], f["mean_sig_promoter"], f["mean_sig_gene_body"], 
                    f["mean_sig_around_TES"], rna_seq_data["TPM"][i], rna_seq_data["FPKM"][i]]

                pred_features.append(f)
        
        true_features = pd.DataFrame(true_features, columns=["assay", "geneID", "promoter_signal", "gene_body_signal", "TES_signal", "TPM", "FPKM"])
        pred_features_all = pd.DataFrame(pred_features, columns=["assay", "geneID", "promoter_signal", "gene_body_signal", "TES_signal", "TPM", "FPKM"])
        pred_features_avail = pred_features_all[pred_features_all["assay"].isin(available_assays)]

        report = {}
        # Perform K-Fold Cross Validation for both true and predicted data
        # print("Evaluating Experimental Data")
        report['true_linear'] = k_fold_cross_validation(true_features, k=k_fold, target='TPM', logscale=True, model_type='linear')
        
        # print("Evaluating Denoised + Imputed Data")
        report['denoised_imputed_linear'] = k_fold_cross_validation(pred_features_all, k=k_fold, target='TPM', logscale=True, model_type='linear')

        # print("Evaluating Denoised Data")
        report['denoised_linear'] = k_fold_cross_validation(pred_features_avail, k=k_fold, target='TPM', logscale=True, model_type='linear')

        # Perform K-Fold Cross Validation for both true and predicted data
        # print("Evaluating Experimental Data")
        report['true_svr'] = k_fold_cross_validation(true_features, k=k_fold, target='TPM', logscale=True, model_type='svr')
        
        # print("Evaluating Denoised + Imputed Data")
        report['denoised_imputed_svr'] = k_fold_cross_validation(pred_features_all, k=k_fold, target='TPM', logscale=True, model_type='svr')

        # print("Evaluating Denoised Data")
        report['denoised_svr'] = k_fold_cross_validation(pred_features_avail, k=k_fold, target='TPM', logscale=True, model_type='svr')
        
        # Plotting REC curves for comparison
        if plot_REC:
            plt.figure(figsize=(14, 7))
            
            # Plot REC for SVR models
            plt.subplot(1, 2, 1)
            true_errors_svr = report['true_svr']['errors']
            denoised_errors_svr = report['denoised_svr']['errors']
            denoised_imputed_errors_svr = report['denoised_imputed_svr']['errors']
            
            sorted_true_errors_svr = np.sort(true_errors_svr)
            cumulative_true_svr = np.arange(1, len(sorted_true_errors_svr) + 1) / len(sorted_true_errors_svr)
            
            sorted_denoised_errors_svr = np.sort(denoised_errors_svr)
            cumulative_denoised_svr = np.arange(1, len(sorted_denoised_errors_svr) + 1) / len(sorted_denoised_errors_svr)

            sorted_denoised_imputed_errors_svr = np.sort(denoised_imputed_errors_svr)
            cumulative_denoised_imputed_svr = np.arange(1, len(sorted_denoised_imputed_errors_svr) + 1) / len(sorted_denoised_imputed_errors_svr)
            
            plt.plot(sorted_true_errors_svr, cumulative_true_svr, label='Observed', color='blue', alpha=0.7)
            plt.plot(sorted_denoised_errors_svr, cumulative_denoised_svr, label='Denoised', color='orange', alpha=0.7)
            plt.plot(sorted_denoised_imputed_errors_svr, cumulative_denoised_imputed_svr, label='Denoised+Imputed', color='green', alpha=0.7)
            plt.xlabel('Error Tolerance')
            plt.ylabel('Proportion of Points within Tolerance')
            plt.title('REC Curve - SVR')
            plt.legend()
            plt.grid(True)
            
            # Plot REC for Linear models
            plt.subplot(1, 2, 2)
            true_errors_linear = report['true_linear']['errors']
            denoised_errors_linear = report['denoised_linear']['errors']
            denoised_imputed_errors_linear = report['denoised_imputed_linear']['errors']
            
            sorted_true_errors_linear = np.sort(true_errors_linear)
            cumulative_true_linear = np.arange(1, len(sorted_true_errors_linear) + 1) / len(sorted_true_errors_linear)
            
            sorted_denoised_errors_linear = np.sort(denoised_errors_linear)
            cumulative_denoised_linear = np.arange(1, len(sorted_denoised_errors_linear) + 1) / len(sorted_denoised_errors_linear)

            sorted_denoised_imputed_errors_linear = np.sort(denoised_imputed_errors_linear)
            cumulative_denoised_imputed_linear = np.arange(1, len(sorted_denoised_imputed_errors_linear) + 1) / len(sorted_denoised_imputed_errors_linear)
            
            plt.plot(sorted_true_errors_linear, cumulative_true_linear, label='Observed', color='blue', alpha=0.7)
            plt.plot(sorted_denoised_errors_linear, cumulative_denoised_linear, label='Denoised', color='orange', alpha=0.7)
            plt.plot(sorted_denoised_imputed_errors_linear, cumulative_denoised_imputed_linear, label='Denoised+Imputed', color='green', alpha=0.7)
            plt.xlabel('Error Tolerance')
            plt.ylabel('Proportion of Points within Tolerance')
            plt.title('REC Curve - Linear Regression')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            savepath = os.path.join(self.savedir, bios_name+f"_{len(available_assays)}")
            if os.path.exists(savepath) ==False:
                os.mkdir(savepath)

            plt.savefig(savepath+"/RNAseq_REC.svg", format="svg")

        return report
            
    def get_metrics(self, imp_dist, ups_dist, Y, bios_name, availability):
        """
        reportoir of metrics -- per_bios:

            peak_ovr: 01thr, 05thr, 10thr

            GeWi: MSE, Pearson, Spearman
            1imp: MSE, Pearson, Spearman
            1obs: MSE, Pearson, Spearman
            gene: MSE, Pearson, Spearman
            prom: MSE, Pearson, Spearman
        """

        imp_mean = imp_dist.expect()
        ups_mean = ups_dist.expect()

        imp_std = imp_dist.std()
        ups_std = ups_dist.std()

        if self.dataset.has_rnaseq(bios_name):
            print("got rna-seq data")
            rnaseq_res = self.eval_rnaseq(bios_name, ups_mean, Y, availability, k_fold=10, plot_REC=True)

        # imp_lower_60, imp_upper_60 = imp_dist.interval(confidence=0.6)
        # ups_lower_60, ups_upper_60 = ups_dist.interval(confidence=0.6)

        # imp_lower_80, imp_upper_80 = imp_dist.interval(confidence=0.8)
        # ups_lower_80, ups_upper_80 = ups_dist.interval(confidence=0.8)
        print("getting 0.95 interval conf")

        imp_lower_95, imp_upper_95 = imp_dist.interval(confidence=0.95)
        ups_lower_95, ups_upper_95 = ups_dist.interval(confidence=0.95)

        results = []
        # for j in availability:  # for each feature i.e. assay
        for j in range(Y.shape[1]):

            if j in list(availability):
                target = Y[:, j].numpy()

                for comparison in ['imputed', 'upsampled']:
                    
                    if comparison == "imputed":
                        pred = imp_mean[:, j].numpy()
                        pred_std = imp_std[:, j].numpy()
                        # lower_60 = imp_lower_60[:, j].numpy()
                        # lower_80 = imp_lower_80[:, j].numpy()
                        lower_95 = imp_lower_95[:, j].numpy()

                        # upper_60 = imp_upper_60[:, j].numpy()
                        # upper_80 = imp_upper_80[:, j].numpy()
                        upper_95 = imp_upper_95[:, j].numpy()

                        quantile = self.metrics.confidence_quantile(imp_dist.p[:,j], imp_dist.n[:,j], target)
                        p0bgdf = self.metrics.foreground_vs_background(imp_dist.p[:,j], imp_dist.n[:,j], target)
                        
                    elif comparison == "upsampled":
                        pred = ups_mean[:, j].numpy()
                        pred_std = ups_std[:, j].numpy()
                        # lower_60 = ups_lower_60[:, j].numpy()
                        # lower_80 = ups_lower_80[:, j].numpy()
                        lower_95 = ups_lower_95[:, j].numpy()

                        # upper_60 = ups_upper_60[:, j].numpy()
                        # upper_80 = ups_upper_80[:, j].numpy()
                        upper_95 = ups_upper_95[:, j].numpy()

                        quantile = self.metrics.confidence_quantile(ups_dist.p[:,j], ups_dist.n[:,j], target)
                        p0bgdf = self.metrics.foreground_vs_background(ups_dist.p[:,j], ups_dist.n[:,j], target)


                    # corresp, corresp_deriv = self.metrics.correspondence_curve(target, pred)
                    metrics = {
                        'bios':bios_name,
                        'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                        'comparison': comparison,
                        'available assays': len(availability),

                        "obs":target,
                        "imp":pred,
                        "pred_quantile":quantile,
                        "pred_std":pred_std,

                        # "lower_60" : lower_60,
                        # "lower_80" : lower_80,
                        "lower_95" : lower_95,

                        # "upper_60": upper_60,
                        # "upper_80": upper_80,
                        "upper_95": upper_95,

                        "p0_bg":p0bgdf["p0_bg"],
                        "p0_fg":p0bgdf["p0_fg"],

                        'MSE-GW': self.metrics.mse(target, pred),
                        'Pearson-GW': self.metrics.pearson(target, pred),
                        'Spearman-GW': self.metrics.spearman(target, pred),
                        'r2_GW': self.metrics.r2(target, pred),

                        'MSE-1obs': self.metrics.mse1obs(target, pred),
                        'Pearson_1obs': self.metrics.pearson1_obs(target, pred),
                        'Spearman_1obs': self.metrics.spearman1_obs(target, pred),
                        'r2_1obs': self.metrics.r2_1obs(target, pred),

                        'MSE-1imp': self.metrics.mse1imp(target, pred),
                        'Pearson_1imp': self.metrics.pearson1_imp(target, pred),
                        'Spearman_1imp': self.metrics.spearman1_imp(target, pred),
                        'r2_1imp': self.metrics.r2_1imp(target, pred),

                        'MSE-gene': self.metrics.mse_gene(target, pred),
                        'Pearson_gene': self.metrics.pearson_gene(target, pred),
                        'Spearman_gene': self.metrics.spearman_gene(target, pred),
                        'r2_gene': self.metrics.r2_gene(target, pred),

                        'MSE-prom': self.metrics.mse_prom(target, pred),
                        'Pearson_prom': self.metrics.pearson_prom(target, pred),
                        'Spearman_prom': self.metrics.spearman_prom(target, pred),
                        'r2_prom': self.metrics.r2_prom(target, pred),

                        "peak_overlap_01thr": self.metrics.peak_overlap(target, pred, p=0.01),
                        "peak_overlap_05thr": self.metrics.peak_overlap(target, pred, p=0.05),
                        "peak_overlap_10thr": self.metrics.peak_overlap(target, pred, p=0.10),

                    #     "corresp_curve": corresp,
                    #     "corresp_curve_deriv": corresp_deriv
                    }
                    
                    if self.dataset.has_rnaseq(bios_name):
                        metrics["rnaseq-true-pcc-linear"] = rnaseq_res["true_linear"]["avg_pcc"]
                        metrics["rnaseq-true-pcc-svr"] = rnaseq_res["true_svr"]["avg_pcc"]

                        metrics["rnaseq-denoised-pcc-linear"] = rnaseq_res["denoised_linear"]["avg_pcc"]
                        metrics["rnaseq-denoised-pcc-svr"] = rnaseq_res["denoised_svr"]["avg_pcc"]

                        metrics["rnaseq-true-mse-linear"] = rnaseq_res["true_linear"]["avg_mse"]
                        metrics["rnaseq-true-mse-svr"] = rnaseq_res["true_svr"]["avg_mse"]
                        
                        metrics["rnaseq-denoised-mse-linear"] = rnaseq_res["denoised_linear"]["avg_mse"]
                        metrics["rnaseq-denoised-mse-svr"] = rnaseq_res["denoised_svr"]["avg_mse"]

                    results.append(metrics)

            else:
                # continue
                pred = ups_mean[:, j].numpy()
                # lower_60 = ups_lower_60[:, j].numpy()
                # lower_80 = ups_lower_80[:, j].numpy()
                lower_95 = ups_lower_95[:, j].numpy()

                # upper_60 = ups_upper_60[:, j].numpy()
                # upper_80 = ups_upper_80[:, j].numpy()
                upper_95 = ups_upper_95[:, j].numpy()

                metrics = {
                    'bios':bios_name,
                    'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                    'comparison': "None",
                    'available assays': len(availability),

                    "imp":pred,

                    # "lower_60" : lower_60,
                    # "lower_80" : lower_80,
                    "lower_95" : lower_95,

                    # "upper_60": upper_60,
                    # "upper_80": upper_80,
                    "upper_95": upper_95
                    }
                results.append(metrics)
            
        return results
    
    def load_bios(self, bios_name, x_dsf, y_dsf=1):
        """
        Load biosample data for a specified biosample at given downsampling factors for X and Y.

        Parameters:
        bios_name (str): The name of the biosample.
        x_dsf (int): Downsampling factor for the X dataset.
        y_dsf (int): Downsampling factor for the Y dataset, defaults to 1.

        Returns:
        tuple: A tuple containing the tensors for X, mX, avX, Y, mY, and avY.
        """
        temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        # temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", self.chr_sizes["chr21"]//4, self.chr_sizes["chr21"]//2], x_dsf)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx

        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        # temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", self.chr_sizes["chr21"]//4, self.chr_sizes["chr21"]//2], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length

        X, Y = X[:num_rows, :], Y[:num_rows, :]


        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        
        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        return X, mX, avX, Y, mY, avY

    def pred(self, X, mX, mY, avail, imp_target=[]):
        # Initialize a tensor to store all predictions
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.clone()
                avail_batch = avail_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()

                x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])
                # mY_batch_missing_vals = (mY_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]
                # mY_batch[mY_batch_missing_vals] = self.token_dict["cloze_mask"]
                if self.version in ["a", "b"]:
                    avail_batch[avail_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    # mY_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    if self.version in ["a", "b"]:
                        avail_batch[:, imp_target] = self.token_dict["cloze_mask"]
                    elif self.version in ["c", "d"]:
                        avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.version in ["30a", "30b", "30d"]:
                    outputs_p, outputs_n, _, _ = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)
                elif self.version in ["30c"]:
                    outputs_p, outputs_n = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)

            # Store the predictions in the large tensor
            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()

        return n, p

    def bios_pipeline(self, bios_name, x_dsf):
        X, mX, avX, Y, mY, avY = self.load_bios(bios_name, x_dsf)  

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        for leave_one_out in available_indices:
            n, p = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])
            
            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]
            print(f"got imputations for feature #{leave_one_out+1}")
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        print("got upsampled")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        imp_dist = NegativeBinomial(p_imp, n_imp)
        ups_dist = NegativeBinomial(p_ups, n_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) 

        eval_res = self.get_metrics(imp_dist, ups_dist, Y, bios_name, available_indices)
        return eval_res

    def viz_bios(self, eval_res):
        print("plotting signal tracks")
        try:
            self.viz.BIOS_signal_track(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal tracks: {e}")

        print("plotting signal confidence")
        try:
            self.viz.BIOS_signal_confidence(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal confidence: {e}")

        # Filter out results without 'obs'
        eval_res = [res for res in eval_res if "obs" in res]

        print("plotting mean vs. std hexbin")
        try:
            self.viz.BIOS_mean_std_hexbin(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot mean vs. std hexbin: {e}")

        # print("plotting quantile heatmap")
        # try:
        #     self.viz.BIOS_quantile_heatmap(eval_res)
        #     self.viz.clear_pallete()
        # except Exception as e:
        #     print(f"Failed to plot quantile heatmap: {e}")

        print("plotting error vs. std hexbin")
        try:
            self.viz.BIOS_error_std_hexbin(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot error vs. std hexbin: {e}")

        print("plotting quantile histogram")
        try:
            self.viz.BIOS_quantile_hist(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot quantile histogram: {e}")

        print("plotting context-specific performance")
        try:
            self.viz.BIOS_context_length_specific_performance(eval_res, self.context_length, bins=10)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot context-specific performance: {e}")

        print("plotting signal scatter with marginals")
        try:
            self.viz.BIOS_signal_scatter_with_marginals(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal scatter with marginals: {e}")

        print("plotting signal heatmap")
        try:
            self.viz.BIOS_signal_heatmap(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal heatmap: {e}")

        print("plotting signal rank heatmap")
        try:
            self.viz.BIOS_signal_rank_heatmap(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal rank heatmap: {e}")

        # Uncomment the following blocks if you want to include these plots as well:
        # print("plotting mean vs. std scatter")
        # try:
        #     self.viz.BIOS_mean_std_scatter(eval_res)
        #     self.viz.clear_pallete()
        # except Exception as e:
        #     print(f"Failed to plot mean vs. std scatter: {e}")
        # print("plotting correspondence curve")
        # try:
        #     self.viz.BIOS_corresp_curve(eval_res)
        #     self.viz.clear_pallete()
        # except Exception as e:
        #     print(f"Failed to plot correspondence curve: {e}")

        # print("plotting correspondence curve derivative")
        # try:
        #     self.viz.BIOS_corresp_curve_deriv(eval_res)
        #     self.viz.clear_pallete()
        # except Exception as e:
        #     print(f"Failed to plot correspondence curve derivative: {e}")

    def viz_all(self, dsf=1):
        """
        visualizations -- all_bios:
        
            denoised vs imputed
                boxplots for metric per assay
                    peak_ovr: 01thr, 05thr, 10thr
                    GeWi: MSE, Pearson, Spearman
                    1imp: MSE, Pearson, Spearman
                    1obs: MSE, Pearson, Spearman
                    gene: MSE, Pearson, Spearman
                    prom: MSE, Pearson, Spearman
        """
        
        self.model_res = []
        print(f"Evaluating {len(list(self.dataset.navigation.keys()))} biosamples...")
        for bios in list(self.dataset.navigation.keys()):
            try:
                print("evaluating ", bios)
                eval_res_bios = self.bios_pipeline(bios, dsf)
                print("got results for ", bios)
                self.viz_bios(eval_res_bios)
                
                to_del = [
                    "obs", "imp", "pred_quantile", "pred_std", 
                    "lower_60", "lower_80", "lower_95", 
                    "upper_60", "upper_80", "upper_95"]

                for f in eval_res_bios:
                    fkeys = list(f.keys())
                    for d in to_del:
                        if d in fkeys:
                            del f[d]
                    
                    if f["comparison"] != "None":
                        self.model_res.append(f)
            except:
                pass

        self.model_res = pd.DataFrame(self.model_res)
        self.model_res.to_csv(f"{self.savedir}/model_eval_DSF{dsf}.csv", index=False)

        # boxplot_metrics = [
        #     'MSE-GW', 'Pearson-GW', 'Spearman-GW',
        #     'MSE-1obs', 'Pearson_1obs', 'Spearman_1obs',
        #     'MSE-1imp', 'Pearson_1imp', 'Spearman_1imp',
        #     'MSE-gene', 'Pearson_gene', 'Spearman_gene',
        #     'MSE-prom', 'Pearson_prom', 'Spearman_prom',
        #     'peak_overlap_01thr', 'peak_overlap_05thr', 
        #     'peak_overlap_10thr']
        
        # for m in boxplot_metrics:
        #     self.viz.MODEL_boxplot(self.model_res, metric=m)
        #     self.viz.MODEL_regplot_overall(self.model_res, metric=m)
        #     self.viz.MODEL_regplot_perassay(self.model_res, metric=m)



class VISUALS(object):
    def __init__(self, resolution=25, savedir="models/evals/"):
        self.metrics = METRICS()
        self.resolution = resolution
        self.savedir = savedir

    def clear_pallete(self):
        sns.reset_orig
        plt.close("all")
        plt.style.use('default')
        plt.clf()

    def BIOS_signal_track(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        example_gene_coord = (33481539//self.resolution, 33588914//self.resolution) # GART
        example_gene_coord2 = (25800151//self.resolution, 26235914//self.resolution) # APP
        example_gene_coord3 = (31589009//self.resolution, 31745788//self.resolution) # SOD1
        example_gene_coord4 = (39526359//self.resolution, 39802081//self.resolution) # B3GALT5
        example_gene_coord5 = (33577551//self.resolution, 33919338//self.resolution) # ITSN1

        # Create a list of example gene coordinates for iteration
        example_gene_coords = [
            example_gene_coord, example_gene_coord2, example_gene_coord3,
            example_gene_coord4, example_gene_coord5]

        # Define the size of the figure
        plt.figure(figsize=(6 * len(example_gene_coords), len(eval_res) * 2))

        # Loop over each result
        for j in range(len(eval_res)):
            # Loop over each gene
            for i, gene_coord in enumerate(example_gene_coords):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(example_gene_coords), j * len(example_gene_coords) + i + 1)
                
                # Calculate x_values based on the current gene's coordinates
                x_values = range(gene_coord[0], gene_coord[1])
                imputed_values = eval_res[j]["imp"][gene_coord[0]:gene_coord[1]]

                # Plot the lines
                if "obs" in eval_res[j].keys():
                    observed_values = eval_res[j]["obs"][gene_coord[0]:gene_coord[1]]
                    ax.plot(x_values, observed_values, color="blue", alpha=0.7, label="Observed", linewidth=0.1)
                    ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")

                ax.plot(x_values, imputed_values, "--", color="red", alpha=0.5, label="Imputed", linewidth=0.1)
                ax.fill_between(x_values, 0, imputed_values, color="red", alpha=0.5)

                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("Signal")

                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                                mlines.Line2D([], [], color='red',  label='Imputed')]
                ax.legend(handles=custom_lines)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_tracks.png", dpi=300)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_tracks.svg", format="svg")

    def BIOS_signal_confidence(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # example_gene_coord =  (33481539//self.resolution, 33588914//self.resolution) # GART
        # example_gene_coord2 = (25800151//self.resolution, 26235914//self.resolution) # APP
        # example_gene_coord3 = (31589009//self.resolution, 31745788//self.resolution) # SOD1
        # example_gene_coord4 = (39526359//self.resolution, 39802081//self.resolution) # B3GALT5
        # example_gene_coord5 = (33577551//self.resolution, 33919338//self.resolution) # ITSN1

        # Create a list of example gene coordinates for iteration
        example_gene_coords = [
            (33481539//self.resolution, 33588914//self.resolution), # GART
            (25800151//self.resolution, 26235914//self.resolution), # APP
            (31589009//self.resolution, 31745788//self.resolution), # SOD1
            (39526359//self.resolution, 39802081//self.resolution), # B3GALT5
            (33577551//self.resolution, 33919338//self.resolution) # ITSN1
            ]
            # example_gene_coord, example_gene_coord2, example_gene_coord3,
            # example_gene_coord4, example_gene_coord5]

        # Define the size of the figure
        plt.figure(figsize=(8 * len(example_gene_coords), len(eval_res) * 2))
        # plt.subplots_adjust(hspace=0.4, wspace=0.3)

        for j, result in enumerate(eval_res):
            for i, gene_coord in enumerate(example_gene_coords):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(example_gene_coords), j * len(example_gene_coords) + i + 1)
                
                # Calculate x_values based on the current gene's coordinates
                x_values = range(gene_coord[0], gene_coord[1])

                # Fill between for confidence intervals
                ax.fill_between(
                    x_values, result['lower_95'][gene_coord[0]:gene_coord[1]], result['upper_95'][gene_coord[0]:gene_coord[1]], 
                    color='coral', alpha=0.4, label='95% Confidence')

                # ax.fill_between(
                #     x_values, result['lower_80'][gene_coord[0]:gene_coord[1]], result['upper_80'][gene_coord[0]:gene_coord[1]], 
                #     color='coral', alpha=0.2, label='80% Confidence')

                # ax.fill_between(
                #     x_values, result['lower_60'][gene_coord[0]:gene_coord[1]], result['upper_60'][gene_coord[0]:gene_coord[1]], 
                #     color='coral', alpha=0.4, label='60% Confidence')

                # Plot the median predictions
                ax.plot(x_values, result['imp'][gene_coord[0]:gene_coord[1]], label='Mean', color='red', linewidth=0.5)

                if "obs" in result.keys():
                    # Plot the actual observations
                    ax.plot(
                        x_values, result['obs'][gene_coord[0]:gene_coord[1]], 
                        label='Observed', color='royalblue', linewidth=0.4, alpha=0.8)


                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution

                # Set plot titles and labels
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("Signal")
                ax.set_yscale('log') 
                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                # Only show legend in the first subplot to avoid redundancy
                if i == 0 and j ==0:
                    ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/confidence_intervals.pdf", dpi=300)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/confidence_intervals.svg", format="svg")

    def BIOS_quantile_scatter(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["pred_quantile"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                ax.scatter(xs, ys, color="black", s=5, alpha=0.7)
                # ax.grid(True, linestyle='-', color='gray', alpha=0.5)
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"Obs. vs. Pred. Quantile {eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed Values")
                ax.set_ylabel("Predicted Quantile")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_scatter.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_scatter.svg", format="svg")

    def BIOS_quantile_density(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["pred_quantile"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                sns.kdeplot(x=xs, y=ys, cmap="Blues", fill=True, ax=ax)
                # ax.scatter(xs, ys, color='red', alpha=0.3)
                # ax.grid(True, linestyle='-', color='gray', alpha=0.5)

                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"Obs. vs. Pred. Quantile w/ Density {eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed Values")
                ax.set_ylabel("Predicted Quantile")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_density_scatter.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_density_scatter.svg", format="svg")
    
    def BIOS_quantile_hist(self, eval_res, b=20):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["pred_quantile"]

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])

                ax.hist(ys, bins=b, color='blue', alpha=0.7, density=True)
                # ax.grid(True, linestyle='-', color='gray', alpha=0.5)
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"Obs. vs. Pred. Quantile {eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}")
                ax.set_xlabel("Predicted CDF Quantile")
                ax.set_ylabel("Density")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_hist.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_hist.svg", format="svg")

    def BIOS_quantile_heatmap(self, eval_res, b=20):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["pred_quantile"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                # Create the heatmap
                h, xedges, yedges = np.histogram2d(np.asarray(xs), np.asarray(ys), bins=b, density=True)
                h = h.T  # Transpose to correct the orientation
                h = h / h.sum(axis=0, keepdims=True)  # Normalize cols

                im = ax.imshow(
                    h, interpolation='nearest', origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    aspect='auto', cmap='viridis', norm=LogNorm())
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted Quantiles")
                plt.colorbar(im, ax=ax, orientation='vertical')
                

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_heatmap.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_heatmap.svg", format="svg")

    def BIOS_mean_std_scatter(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        plt.figure(figsize=(5, len(eval_res) * 5))  # one column with len(eval_res) rows

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(len(eval_res), 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs"], eval_res[j]["imp"], eval_res[j]["pred_std"]
            pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

            sc = ax.scatter(observed, pred_mean, c=pred_std, cmap='viridis', alpha=0.6, s=5)
            plt.colorbar(sc, ax=ax, label='Predicted std')
            ax.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'k--')
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted Mean')
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc}")
            # plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/mean_std_scatter.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/mean_std_scatter.svg", format="svg")
    
    def BIOS_error_std_hexbin(self, eval_res):
        save_path = f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        num_plots = len(eval_res) * 3  # Each evaluation will have 3 subplots
        plt.figure(figsize=(15, len(eval_res) * 5))  # Adjust width for 3 columns

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                # skip rows without observed signal
                continue

            observed, pred_mean, pred_std = eval_res[j]["obs"], eval_res[j]["imp"], eval_res[j]["pred_std"]
            pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"
            error = np.abs(observed - pred_mean)

            # Calculate the percentiles for x-axis limits
            x_90 = np.percentile(error, 99)
            x_99 = np.percentile(error, 99.9)

            # Define the ranges for subsetting
            ranges = [(0, x_90), (x_90, x_99), (x_99, error.max())]

            for i, (x_min, x_max) in enumerate(ranges):
                # Subset the data for the current range
                mask = (error >= x_min) & (error <= x_max)
                subset_error = error[mask]
                subset_pred_std = pred_std[mask]
                
                ax = plt.subplot(len(eval_res), 3, j * 3 + i + 1)

                # Hexbin plot for the subset data
                hb = ax.hexbin(subset_error, subset_pred_std, gridsize=50, cmap='viridis', mincnt=1, norm=LogNorm())

                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Predicted Std Dev')
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc} (Range: {x_min:.2f}-{x_max:.2f})")

                # Add color bar
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Log10(Counts)')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/error_std_hexbin.png", dpi=150)
        plt.savefig(f"{save_path}/error_std_hexbin.svg", format="svg")
    
    def BIOS_mean_std_hexbin(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        plt.figure(figsize=(5, len(eval_res) * 5))  # one column with len(eval_res) rows

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(len(eval_res), 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs"], eval_res[j]["imp"], eval_res[j]["pred_std"]
            pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

            hb = ax.hexbin(observed, pred_mean, C=pred_std, gridsize=30, cmap='viridis', reduce_C_function=np.mean)
            plt.colorbar(hb, ax=ax, label='Predicted std')
            ax.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'k--')
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted Mean')
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc}")
            # plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/mean_std_hexbin.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/mean_std_hexbin.svg", format="svg")
        
    def BIOS_signal_scatter(self, eval_res, share_axes=True):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue

            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                ax.scatter(xs, ys, color="black", s=5, alpha=0.7)
                
                if share_axes:
                    # Determine the range for x and y axes
                    common_min = min(min(xs), min(ys))
                    common_max = max(max(xs), max(ys))
                    
                    # Set the same range for x and y axes
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters.svg", format="svg")

    def BIOS_signal_scatter_with_marginals(self, eval_res, share_axes=True, percentile_cutoff=99):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]
        num_rows = len(eval_res)
        num_cols = len(cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

        for j, result in enumerate(eval_res):
            if "obs" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = axes[j, i] if num_rows > 1 else axes[i]

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"
                    
                sns.scatterplot(x=xs, y=ys, ax=ax, color="#4CB391", s=3, alpha=0.9)

                # Calculate percentile cutoffs for both axes
                x_upper = np.percentile(xs, percentile_cutoff)
                y_upper = np.percentile(ys, percentile_cutoff)
                
                # Use the same upper bound for both axes to maintain aspect ratio
                upper_bound = min(x_upper, y_upper)
                
                # Filter points within bounds
                mask = (xs <= upper_bound) & (ys <= upper_bound)
                xs_filtered = xs[mask]
                ys_filtered = ys[mask]

                # Update bin range for histograms using filtered data
                bin_range = np.linspace(0, upper_bound, 50)

                ax_histx = ax.inset_axes([0, 1.05, 1, 0.2])
                ax_histy = ax.inset_axes([1.05, 0, 0.2, 1])
                
                ax_histx.hist(xs_filtered, bins=bin_range, alpha=0.9, color="#f25a64")
                ax_histy.hist(ys_filtered, bins=bin_range, orientation='horizontal', alpha=0.9, color="#f25a64")
                
                ax_histx.set_xticklabels([])
                ax_histx.set_yticklabels([])
                ax_histy.set_xticklabels([])
                ax_histy.set_yticklabels([])

                # Set title, labels, and range if share_axes is True
                ax.set_title(f"{result['feature']}_{c}_{result['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

                if share_axes:
                    # Set axis limits
                    ax.set_xlim(0, upper_bound)
                    ax.set_ylim(0, upper_bound)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters_with_marginals.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters_with_marginals.svg", format="svg")

    def BIOS_signal_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    title_suffix = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    title_suffix = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    title_suffix = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    title_suffix = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    title_suffix = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                # Create the heatmap
                h, xedges, yedges = np.histogram2d(xs, ys, bins=bins, density=True)
                h = h.T  # Transpose to correct the orientation
                ax.imshow(h, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', norm=LogNorm())

                if share_axes:
                    common_min = min(min(xs), min(ys))
                    common_max = max(max(xs), max(ys))
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)

                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{title_suffix}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_heatmaps.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_heatmaps.svg", format="svg")
        
    def BIOS_signal_scatter_rank(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                continue
            # Loop over each gene
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    scc = f"SRCC_GW: {eval_res[j]['Spearman-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    scc = f"SRCC_Gene: {eval_res[j]['Spearman_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    scc = f"SRCC_TSS: {eval_res[j]['Spearman_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    scc = f"SRCC_1obs: {eval_res[j]['Spearman_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    scc = f"SRCC_1imp: {eval_res[j]['Spearman_1imp']:.2f}"


                # Convert values to ranks
                xs = rankdata(xs)
                ys = rankdata(ys)

                ax.scatter(xs, ys, color="black", s=5, alpha=0.7)

                # Set the formatter for both axes
                formatter = mticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-1, 1))  # This will use scientific notation for numbers outside this range

                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)

                # Update the subplot with the new formatter
                plt.draw()  # This updates the current figure and applies the formatter
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{scc}", fontsize=9)
                ax.set_xlabel("Observed | rank")
                ax.set_ylabel("Predicted | rank")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_scatters.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_scatters.svg", format="svg")
    
    def BIOS_signal_rank_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    scc = f"SRCC_GW: {eval_res[j]['Spearman-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    scc = f"SRCC_Gene: {eval_res[j]['Spearman_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    scc = f"SRCC_TSS: {eval_res[j]['Spearman_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    scc = f"SRCC_1obs: {eval_res[j]['Spearman_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    scc = f"SRCC_1imp: {eval_res[j]['Spearman_1imp']:.2f}"

                # Convert values to ranks
                xs = rankdata(xs)
                ys = rankdata(ys)

                # Create the heatmap for ranked values
                h, xedges, yedges = np.histogram2d(xs, ys, bins=bins, density=True)
                h = h.T  # Transpose to correct the orientation
                ax.imshow(h, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='viridis', norm=LogNorm())

                if share_axes:
                    common_min = min(xedges[0], yedges[0])
                    common_max = max(xedges[-1], yedges[-1])
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)

                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{scc}")
                ax.set_xlabel("Observed | rank")
                ax.set_ylabel("Predicted | rank")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_heatmaps.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_heatmaps.svg", format="svg")
        
    def BIOS_corresp_curve(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        num_assays = len(eval_res)
        n_cols = math.floor(math.sqrt(num_assays))
        n_rows = math.ceil(num_assays / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=((4*n_cols), (4*n_rows)))

        c = 0

        for i in range(n_rows):
            for j in range(n_cols):

                if "obs" not in eval_res[c]:
                    continue

                if c>=num_assays:
                    continue
                
                t = [p[0] for p in eval_res[c]['corresp_curve']]
                psi = [p[1] for p in eval_res[c]['corresp_curve']]

                axs[i,j].plot(t, psi, c="red")

                axs[i,j].plot(t, t, "--", c="black")

                axs[i,j].set_title(f"{eval_res[c]['feature']}_{eval_res[c]['comparison']}")

                axs[i,j].fill_between(t, t, psi, color="red", alpha=0.4)

                c += 1
                axs[i,j].set_xlabel("t")
                axs[i,j].set_ylabel("psi")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/corresp_curve.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/corresp_curve.svg", format="svg")
    
    def BIOS_corresp_curve_deriv(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")
            
        num_assays = len(eval_res)
        n_cols = math.floor(math.sqrt(num_assays))
        n_rows = math.ceil(num_assays / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=((4*n_cols), (4*n_rows)))

        c = 0

        for i in range(n_rows):
            for j in range(n_cols):

                if "obs" not in eval_res[c]:
                    continue

                if c>=num_assays:
                    continue
                    
                t = [p[0] for p in eval_res[c]['corresp_curve_deriv']]
                psii = [p[1] for p in eval_res[c]['corresp_curve_deriv']]

                axs[i,j].plot(t, psii, c="red")

                axs[i,j].plot(t, [1 for _ in range(len(t))], "--", c="black")

                axs[i,j].set_title(f"{eval_res[c]['feature']}_{eval_res[c]['comparison']}")

                axs[i,j].fill_between(t, [1 for _ in range(len(t))], psii, color="red", alpha=0.4)

                c += 1
                axs[i,j].set_xlabel("t")
                axs[i,j].set_ylabel("psi'")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/corresp_curve_deriv.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/corresp_curve_deriv.svg", format="svg")
    
    def BIOS_context_length_specific_performance(self, eval_res, context_length, bins=10):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        list_of_metrics = ['MSE-GW', 'Pearson-GW', 'Spearman-GW']

        # Define the size of the figure
        plt.figure(figsize=(6 * len(list_of_metrics), len(eval_res) * 2))

        # Loop over each result
        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue

            observed_values = eval_res[j]["obs"]
            imputed_values = eval_res[j]["imp"]

            bin_size = context_length // bins

            observed_values = observed_values.reshape(-1, context_length)
            imputed_values = imputed_values.reshape(-1, context_length)

            observed_values = observed_values.reshape(observed_values.shape[0]*bin_size, bins)
            imputed_values = imputed_values.reshape(imputed_values.shape[0]*bin_size, bins)

            for i, m in enumerate(list_of_metrics):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(list_of_metrics), j * len(list_of_metrics) + i + 1)
                
                xs = [float(xt)/bins for xt in range(bins)]
                # Calculate x_values based on the current gene's coordinates
                ys = []
                for b in range(bins):
                    
                    obs, imp = observed_values[:,b].flatten(), imputed_values[:,b].flatten()
                    if m == 'MSE-GW':
                        ys.append(self.metrics.mse(obs, imp))

                    elif m == 'Pearson-GW':
                        ys.append(self.metrics.pearson(obs, imp))

                    elif m == 'Spearman-GW':
                        ys.append(self.metrics.spearman(obs, imp))
                
                ax.plot(xs, ys, color="grey", linewidth=3)
                # ax.fill_between(xs, 0, ys, alpha=0.7, color="grey")
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_xlabel("position in context")
                ax.set_ylabel(m)
        
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/context.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/context.svg", format="svg")
    
    def MODEL_boxplot(self, df, metric):
        df = df.copy()
        # Sort the dataframe by 'feature'
        df.sort_values('feature', inplace=True)
        fig, axs = plt.subplots(2, figsize=(10, 6))
        fig.suptitle('Boxplots for Imputed and Denoised')

        # Boxplot for Imputed
        imputed_df = df[df['comparison'] == 'imputed']

        if "MSE" in metric:
            imputed_df[metric] = np.log(imputed_df[metric])
            
        sns.boxplot(x='feature', y=metric, data=imputed_df, ax=axs[0], color="grey")
        axs[0].set_title('Imputed')
        axs[0].set(xlabel='Assay', ylabel='log('+metric+')' if "MSE" in metric else metric)
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)  # Rotate x-axis labels

        # Boxplot for Denoised
        denoised_df = df[df['comparison'] == 'denoised']
        if "MSE" in metric:
            denoised_df[metric] = np.log(denoised_df[metric])

        sns.boxplot(x='feature', y=metric, data=denoised_df, ax=axs[1], color="grey")
        axs[1].set_title('Denoised')
        axs[1].set(xlabel='Assay', ylabel='log('+metric+')' if "MSE" in metric else metric)
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)  # Rotate x-axis labels

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{metric}_boxplot.png", dpi=150)
        plt.savefig(f"{self.savedir}/{metric}_boxplot.svg", format="svg")
    
    def MODEL_regplot_overall(self, df, metric):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Scatter plots for Imputed and Denoised')

        # Plot for Imputed
        imputed_df = df[df['comparison'] == 'imputed']
        x_imputed = imputed_df['available train assays']
        y_imputed = imputed_df[metric]

        if "MSE" in metric:
            y_imputed = np.log(y_imputed)
        sns.regplot(x=x_imputed, y=y_imputed, scatter=True, line_kws={"color": "red"}, scatter_kws={"color": "red"}, ax=ax, label='Imputed')
        
        # Plot for Denoised
        denoised_df = df[df['comparison'] == 'denoised']
        x_denoised = denoised_df['available train assays']
        y_denoised = denoised_df[metric]

        if "MSE" in metric:
            y_denoised = np.log(y_denoised)
        sns.regplot(x=x_denoised, y=y_denoised, scatter=True, line_kws={"color": "green"}, scatter_kws={"color": "green"}, ax=ax, label='Denoised')
        
        ax.set(xlabel='Number of Available Train Assays', ylabel='log('+metric+')' if "MSE" in metric else metric)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{metric}_overall_regplot.png", dpi=200)
        plt.savefig(f"{self.savedir}/{metric}_overall_regplot.svg", format="svg")
    
    def MODEL_regplot_perassay(self, df, metric):
        # Get the unique features (assays)
        features = df['feature'].unique()
        num_features = len(features)

        # Determine the layout of the subplots
        n_cols = math.ceil(math.sqrt(num_features))
        n_rows = math.ceil(num_features / n_cols)

        # Create a large figure to accommodate all subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
        
        # Flatten the array of axes for easy iteration
        axs = axs.flatten()

        # Iterate over each unique feature and create a subplot
        for i, feature in enumerate(features):
            # Data for current feature
            feature_df = df[df['feature'] == feature]
            
            # Plot for Imputed
            imputed_df = feature_df[feature_df['comparison'] == 'imputed']
            x_imputed = imputed_df['available train assays']
            y_imputed = imputed_df[metric]

            if "MSE" in metric:
                y_imputed = np.log(y_imputed)
            
            sns.regplot(x=x_imputed, y=y_imputed, scatter=True, line_kws={"color": "red"}, scatter_kws={"color": "red"}, ax=axs[i], label='Imputed')
            
            # Plot for Denoised
            denoised_df = feature_df[feature_df['comparison'] == 'denoised']
            x_denoised = denoised_df['available train assays']
            y_denoised = denoised_df[metric]

            if "MSE" in metric:
                y_denoised = np.log(y_denoised)
            
            sns.regplot(x=x_denoised, y=y_denoised, scatter=True, line_kws={"color": "green"}, scatter_kws={"color": "green"}, ax=axs[i], label='Denoised')
            
            # Set the title and labels
            axs[i].set_title(feature)
            axs[i].set_xlabel('Number of Available Train Assays')
            axs[i].set_ylabel('log('+metric+')' if "MSE" in metric else metric)
            axs[i].legend()

            # Turn off axes for any empty subplots
            if i >= num_features:
                axs[i].axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{metric}_per_assay_metric.png", dpi=200)
        plt.savefig(f"{self.savedir}/{metric}_per_assay_metric.svg", format="svg")

class EVAL_EIC(object): # on chr21
    def __init__(
        self, model, traindata_path, evaldata_path, context_length, batch_size, hyper_parameters_path="",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", version="22", resolution=25, 
        is_arcsin=True, savedir="models/evals/", mode="eval"):

        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.mkdir(self.savedir)

        self.traindata_path = traindata_path
        self.evaldata_path = evaldata_path
        self.is_arcsin = is_arcsin
        self.version = version
        self.context_length = context_length
        self.batch_size = batch_size

        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]
        self.mark_dict = {
            "M01": "ATAC-seq", "M02": "DNase-seq", "M03": "H2AFZ",
            "M04": "H2AK5ac", "M05": "H2AK9ac", "M06": "H2BK120ac",
            "M07": "H2BK12ac", "M08": "H2BK15ac", "M09": "H2BK20ac",
            "M10": "H2BK5ac", "M11": "H3F3A", "M12": "H3K14ac",
            "M13": "H3K18ac", "M14": "H3K23ac", "M15": "H3K23me2",
            "M16": "H3K27ac", "M17": "H3K27me3", "M18": "H3K36me3",
            "M19": "H3K4ac", "M20": "H3K4me1", "M21": "H3K4me2",
            "M22": "H3K4me3", "M23": "H3K56ac", "M24": "H3K79me1",
            "M25": "H3K79me2", "M26": "H3K9ac", "M27": "H3K9me1",
            "M28": "H3K9me2", "M29": "H3K9me3", "M30": "H3T11ph",
            "M31": "H4K12ac", "M32": "H4K20me1", "M33": "H4K5ac",
            "M34": "H4K8ac", "M35": "H4K91ac"
        }

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        self.chr_sizes = {}
        self.resolution = resolution

        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

        self.train_data = {}
        self.eval_data = {}
        self.metrics = METRICS()
        self.viz = VISUALS(resolution=self.resolution, savedir=self.savedir)

        if mode == "dev":
            return

        if type(self.model) == str:
            with open(hyper_parameters_path, 'rb') as f:
                self.hyper_parameters = pickle.load(f)
            loader = MODEL_LOADER(model, self.hyper_parameters)
            self.model = loader.load_epidenoise(version=self.version)

        self.model = self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode
        print(f"# model_parameters: {count_parameters(self.model)}")

        # load and bin chr21 of all bigwig files 
        for t in os.listdir(traindata_path):
            if ".bigwig" in t:

                for e in os.listdir(evaldata_path):
                    if ".bigwig" in e:
                        
                        if t[:3] == e[:3]:

                            if t[:3] not in self.train_data:
                                self.train_data[t[:3]] = {}

                            if e[:3] not in self.eval_data:
                                self.eval_data[e[:3]] = {}

                            self.train_data[t[:3]][t[3:6]] = traindata_path + "/" + t
                            self.eval_data[e[:3]][e[3:6]] = evaldata_path + "/" + e

        print(self.eval_data.keys())
        # print(self.train_data.keys())

    def load_tensor(self, bios_name, mode="train"):
        chr, start, end = "chr21", 0, self.chr_sizes["chr21"]
        all_samples = []
        missing_ind = []

        if mode  == "train": 
            source = self.train_data
            savepath = self.traindata_path + f"/{bios_name}_chr21_{self.resolution}.pt"
        elif mode == "eval":
            source = self.eval_data
            savepath = self.evaldata_path + f"/{bios_name}_chr21_{self.resolution}.pt"
        
        if os.path.exists(savepath):
            all_samples = torch.load(savepath)
            # fill-in missing_ind
            for i in range(all_samples.shape[1]):
                if (all_samples[:, i] == -1).all():
                    missing_ind.append(i)
                    
            return all_samples, missing_ind

        else:
            for i in range(len(self.all_assays)):
                assay = self.all_assays[i]
                if assay in source[bios_name].keys():
                    print("loading ", assay)
                    bw = pyBigWig.open(source[bios_name][assay])
                    signals = bw.stats(chr, start, end, type="mean", nBins=(end - start) // self.resolution)
                
                else:
                    print(assay, "missing")
                    signals = [-1 for _ in range((end - start) // self.resolution)]
                    missing_ind.append(i)

                all_samples.append(signals)

            all_samples = torch.from_numpy(np.array(all_samples, dtype=np.float32)).transpose_(0, 1)

            # replace NaN with zero
            all_samples = torch.where(torch.isnan(all_samples), torch.zeros_like(all_samples), all_samples)

            nan_count = torch.isnan(all_samples).sum().item()
            minus_one_count = (all_samples == -1).sum().item()

            torch.save(all_samples, savepath)
            
            return all_samples, missing_ind

    def load_bios(self, bios_name):
        X, missing_x_i = self.load_tensor(bios_name, mode="train")
        Y, missing_y_i = self.load_tensor(bios_name, mode="eval")

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        if self.is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])
        
        return X, Y, missing_x_i, missing_y_i

    def get_imp(self, X, missing_x_i): # X: train data
        d_model = X.shape[-1]

        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)

                if self.version == "18":
                    outputs, pred_mask = self.model(x_batch)

                elif self.version in ["20", "21"]:
                    mask = torch.zeros_like(x_batch, dtype=torch.bool, device=self.device)
                    for ii in missing_x_i: 
                        mask[:,:,ii] = True

                    mask = mask.to(self.device)
                    outputs, pred_mask = self.model(x_batch, mask)
                
                elif self.version=="22":
                    token_dict = {
                        "missing_mask": -1, 
                        "cloze_mask": -2,
                        "pad": -3
                    }
                    # change missing token to cloze token to force prediction
                    x_batch_missing_vals = (x_batch == -1)
                    x_batch[x_batch_missing_vals] = token_dict["cloze_mask"] 

                    mask = torch.zeros_like(x_batch, dtype=torch.bool, device=self.device)
                    for ii in missing_x_i: 
                        mask[:,:,ii] = True

                    mask = mask.to(self.device)


                    # outputs, aggrmean, aggrstd = self.model(x_batch, mask, None)
                    outputs = self.model(x_batch, mask, None)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        return P

    def get_metrics(self, X, Y, P, missing_x_i, missing_y_i, bios_name):
        """
        reportoir of metrics -- per_bios:

            peak_ovr: 01thr, 05thr, 10thr

            GeWi: MSE, Pearson, Spearman
            1imp: MSE, Pearson, Spearman
            1obs: MSE, Pearson, Spearman
            gene: MSE, Pearson, Spearman
            prom: MSE, Pearson, Spearman
        """

        results = []
        
        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()
                comparison = 'imputed'
            
            elif j not in missing_x_i:
                target = X[:, j].numpy()
                comparison = 'denoised'

            else:
                continue
                
            # if np.isnan(pred).any():
            #     print(f"{self.mark_dict[self.all_assays[j]]} contains nan. skipping")
            #     continue
            # else:
            #     print(f"{self.mark_dict[self.all_assays[j]]} worked")

            # corresp, corresp_deriv = self.metrics.correspondence_curve(target, pred)
            metrics = {
                'bios':bios_name,
                'feature': self.mark_dict[self.all_assays[j]],
                'comparison': comparison,
                'available train assays': len(self.all_assays) - len(missing_x_i),
                'available eval assays': len(self.all_assays) - len(missing_y_i),

                "obs":target,
                "imp":pred,

                'MSE-GW': self.metrics.mse(target, pred),
                'Pearson-GW': self.metrics.pearson(target, pred),
                'Spearman-GW': self.metrics.spearman(target, pred),

                'MSE-1obs': self.metrics.mse1obs(target, pred),
                'Pearson_1obs': self.metrics.pearson1_obs(target, pred),
                'Spearman_1obs': self.metrics.spearman1_obs(target, pred),

                'MSE-1imp': self.metrics.mse1imp(target, pred),
                'Pearson_1imp': self.metrics.pearson1_imp(target, pred),
                'Spearman_1imp': self.metrics.spearman1_imp(target, pred),

                'MSE-gene': self.metrics.mse_gene(target, pred),
                'Pearson_gene': self.metrics.pearson_gene(target, pred),
                'Spearman_gene': self.metrics.spearman_gene(target, pred),

                'MSE-prom': self.metrics.mse_prom(target, pred),
                'Pearson_prom': self.metrics.pearson_prom(target, pred),
                'Spearman_prom': self.metrics.spearman_prom(target, pred),

                # "peak_overlap_01thr": self.metrics.peak_overlap(target, pred, p=0.01),
                # "peak_overlap_05thr": self.metrics.peak_overlap(target, pred, p=0.05),
                # "peak_overlap_10thr": self.metrics.peak_overlap(target, pred, p=0.10),

            #     "corresp_curve": corresp,
            #     "corresp_curve_deriv": corresp_deriv
            }
            results.append(metrics)
        
        return results

    def bios_pipeline(self, bios_name):
        X, Y, missing_x_i, missing_y_i = self.load_bios(bios_name)
        P = self.get_imp(X, missing_x_i)

        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) # eval data
        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1]) # train data

        eval_res = self.get_metrics(X, Y, P, missing_x_i, missing_y_i, bios_name)

        return eval_res

    def bios_test(self):
        missing_x_i, missing_y_i = [], []
        
        X = torch.load("data/C23_trn.pt")
        Y = torch.load("data/C23_val.pt")
        P = torch.load("data/C23_imp.pt")

        
        # fill-in missing_ind
        for i in range(X.shape[1]):
            if (X[:, i] == -1).all():
                missing_x_i.append(i)
        
        # fill-in missing_ind
        for i in range(Y.shape[1]):
            if (Y[:, i] == -1).all():
                missing_y_i.append(i)

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        if self.is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])

        eval_res = self.get_metrics(X, Y, P, missing_x_i, missing_y_i, "test")

        self.viz.BIOS_context_length_specific_performance(eval_res, self.context_length, bins=10)
        self.viz.clear_pallete()

        # self.viz.BIOS_signal_scatter_with_marginals(eval_res)
        # self.viz.clear_pallete()

        # self.viz.BIOS_signal_heatmap(eval_res)
        # self.viz.clear_pallete()

        # self.viz.BIOS_signal_rank_heatmap(eval_res)
        # self.viz.clear_pallete()

    def viz_bios(self, eval_res):
        """
        visualizations -- per_bios:

            highlight imputed vs denoised
            corresp curve + deriv

            scatter_gewi: value, rank 
            scatter_gene: value, rank 
            scatter_prom: value, rank 
            scatter_1imp: value, rank 
            scatter_1obs: value, rank 

            selected regions' signals
        """

        try: 
            print("plotting signal tracks")
            self.viz.BIOS_signal_track(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot signal tracks")

        try:
            print("plotting context_specific performance")
            self.viz.BIOS_context_length_specific_performance(eval_res, self.context_length, bins=10)
            self.viz.clear_pallete()
        except:
            print("faild to plot context_specific performance")
            
        try:
            print("plotting signal scatter")
            self.viz.BIOS_signal_scatter(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot  signal scatter")

        try:
            print("plotting signal scatter with marginals")
            self.viz.BIOS_signal_scatter_with_marginals(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot scatter with marginals")

        try:
            print("plotting signal heatmap")
            self.viz.BIOS_signal_heatmap(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot  signal heatmap")

        try:
            print("plotting signal rank heatmap")
            self.viz.BIOS_signal_rank_heatmap(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot  signal rank heatmap")

        # try:
        #     print("plotting corresp_curve")
        #     self.viz.BIOS_corresp_curve(eval_res)
        #     self.viz.clear_pallete()
        # except:
        #     print("faild to plot corresp_curve")

        # try:
        #     print("plotting corresp_curve_deriv")
        #     self.viz.BIOS_corresp_curve_deriv(eval_res)
        #     self.viz.clear_pallete()
        # except:
        #     print("faild to plot corresp_curve_deriv")
    
    def viz_all(self):
        """
        visualizations -- all_bios:
        
            denoised vs imputed
                boxplots for metric per assay
                    peak_ovr: 01thr, 05thr, 10thr
                    GeWi: MSE, Pearson, Spearman
                    1imp: MSE, Pearson, Spearman
                    1obs: MSE, Pearson, Spearman
                    gene: MSE, Pearson, Spearman
                    prom: MSE, Pearson, Spearman
        """
        self.model_res = []
        for bios in self.eval_data.keys():
            print("evaluating ", bios)
            eval_res_bios = self.bios_pipeline(bios)
            print("got results for ", bios)
            self.viz_bios(eval_res_bios)

            for f in eval_res_bios:
                del f["obs"], f["imp"]
                self.model_res.append(f)

        self.model_res = pd.DataFrame(self.model_res)
        self.model_res.to_csv(f"{self.savedir}/model_eval.csv", index=False)

        boxplot_metrics = [
            'MSE-GW', 'Pearson-GW', 'Spearman-GW',
            'MSE-1obs', 'Pearson_1obs', 'Spearman_1obs',
            'MSE-1imp', 'Pearson_1imp', 'Spearman_1imp',
            'MSE-gene', 'Pearson_gene', 'Spearman_gene',
            'MSE-prom', 'Pearson_prom', 'Spearman_prom',
            'peak_overlap_01thr', 'peak_overlap_05thr', 
            'peak_overlap_10thr']
        
        for m in boxplot_metrics:
            self.viz.MODEL_boxplot(self.model_res, metric=m)
            self.viz.MODEL_regplot_overall(self.model_res, metric=m)
            self.viz.MODEL_regplot_perassay(self.model_res, metric=m)

