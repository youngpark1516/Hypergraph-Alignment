import torch
import torch.nn as nn
from torch_scatter import scatter_add

class FPSWE_pool(nn.Module):
    def __init__(self, d_in, num_anchors=1024, num_projections=1024, anch_freeze=True, out_type='linear'):
        '''
        The PSWE and LPSWE module that produces 
        fixed-dimensional permutation-invariant embeddings 
        for input sets of arbitrary size.
        '''

        super(FPSWE_pool, self).__init__()
        self.d_in = d_in # the dimensionality of the space that each set sample belongs to
        self.num_ref_points = num_anchors # number of points in the reference set
        self.num_projections = num_projections # number of slices
        self.anch_freeze = anch_freeze # if True the reference set and the theta are not learnable

        uniform_ref = torch.linspace(-1, 1, num_anchors).unsqueeze(1).repeat(1, num_projections) #num_anchors x num_preojections
        self.reference = nn.Parameter(uniform_ref, requires_grad=not self.anch_freeze)

        # slicer
        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_projections, bias=False), dim=0)
        if num_projections <= d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        self.theta.weight_v.requires_grad = not self.anch_freeze

        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data)
        self.theta.weight_g.requires_grad = False

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Parameter(torch.zeros(num_projections, num_anchors))
        nn.init.xavier_uniform_(self.weight)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        device = self.weight.device

        if self.anch_freeze == False:
            uniform_ref = torch.linspace(-1, 1, self.num_ref_points).unsqueeze(1).repeat(1, self.num_projections).to(device) #num_anchors x num_preojections
            self.reference.data = uniform_ref

        if self.num_projections <= self.d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        

    def double_self_loops(self, features, index):
        '''
        for the isolated nodes double them because pooling only works
            for minimum 2 elements in a set
        '''
        # Find indices where the group appears only once
        counts = torch.bincount(index)
        unique_mask = counts[index] == 1

        # Get elements to be duplicated
        unique_features = features[unique_mask]
        unique_groups = index[unique_mask]
        
        # Stack original tensor with duplicated unique elements
        new_features = torch.cat((features, unique_features), dim=0)
        new_index = torch.cat((index, unique_groups), dim=0)
        
        # Cleanup intermediate tensors
        del counts, unique_mask, unique_features, unique_groups
        
        return new_features, new_index

    def forward(self, X, hyperedge_index, data=None, name=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        Input:
            X:  N x dn tensor containing N samples in a dn-dimensional space
        Output:
            weighted_embeddings: E x num_projections tensor, containing an embedding of dimension "num_projections" (i.e., number of slices)
        '''
   
        # Step 1: project samples into the 1D slices
        N, dn = X.shape
        Xslices = self.get_slice(X) # N x num_projections

        # for the self-loops double the node to be able to apply the pooling 
        Xslices, hyperedge_index_new = self.double_self_loops(Xslices, hyperedge_index)
        Xslices_sorted, Xind = sparse_sort(Xslices, hyperedge_index_new)

        # regardless of the column sorting, all of them should have the same resorted index
        hyperedge_index_1_sorted = hyperedge_index_new[Xind[:,0]]
        M, dm = self.reference.shape

        eps = 0.00001
        #this should allow a correct interpolation when M>N
        margin_up = 0.9999
        assert (margin_up+eps < 1)

        # Compute constants for this forward pass (no caching to support varying batch structures)
        deg_helper = torch.ones_like(hyperedge_index_1_sorted)
        R = torch.arange(hyperedge_index_1_sorted.shape[0]).to(X.device).to(X.dtype)+1
        pad = torch.tensor([0.0]).to(X.device)
        edges = torch.sort(torch.unique(hyperedge_index_1_sorted))[0]
        hyperedge_index_anchors_1 = edges.repeat_interleave(M)
        num_edges = edges.shape[0]
        xnew = torch.linspace(0, 1, M).repeat(num_edges).to(X.device).to(X.dtype)
        xnew = xnew * 0.99998+eps

        ynew = torch.zeros((self.num_projections, M*num_edges)).to(X.device)

        # compute the degree
        D1 = scatter_add(deg_helper, hyperedge_index_1_sorted) #E
        D = torch.index_select(D1, 0,  hyperedge_index_1_sorted)

        # Step 2: interpolate 

        # compute the x indices to be used as positions for interpolation
        # they are computer for each hyperedge in parallel and are uniformly arranged
        ptr = torch.cat((pad,torch.cumsum(D1, dim=0)))
        P = torch.index_select(ptr, 0,  hyperedge_index_1_sorted)
        assert (D.min() >= 2)
        x = (R-P-1)/(D-1)*0.99999+eps +hyperedge_index_1_sorted
        x = x.unsqueeze(0).repeat(self.num_projections, 1)

        xnew = xnew + hyperedge_index_anchors_1
        xnew = xnew.unsqueeze(0).repeat(self.num_projections, 1)

        #this still correspond to hyperedge_index_1_sorted
        y = torch.transpose(Xslices_sorted, 0, 1).reshape(self.num_projections, -1)
        
        # interpolate y based on the x values
        Xslices_sorted_interpolated = interp1d(x, y, xnew, ynew, hyperedge_index_1_sorted).view(self.num_projections, -1)
        Xslices_sorted_interpolated = torch.transpose(Xslices_sorted_interpolated, 0, 1)

        # reshape the (projected) references. no need for projection since we sample them already projected
        Rslices = self.reference.unsqueeze(0).repeat(num_edges,1,1)#.to(X.device) # num_edges x M x num_projections
        Rslices = Rslices.reshape(num_edges*M,-1) # num_edges x  x num_projections

        # Step 3: sort references and compute proper Wasserstein distance
        _, Rind = sparse_sort(Rslices, hyperedge_index_anchors_1)
        
        # Compute distance between sorted samples (this IS the 1D Wasserstein distance)
        embeddings = Rslices - torch.gather(Xslices_sorted_interpolated, dim=0, index=Rind)
        del Rind, Rslices, Xslices_sorted_interpolated  # Free immediately
        
        embeddings = embeddings.transpose(0, 1)  # [num_projections, num_edges*M]
        embeddings = embeddings.reshape(self.num_projections, num_edges, M)  # [num_projections, num_edges, M]
        
        # Step 4: weighted sum (learned weights are important for Wasserstein!)
        w = self.weight.unsqueeze(1).repeat(1, num_edges, 1)  # [num_projections, num_edges, M]
        weighted_embeddings = (w * embeddings).mean(-1)  # [num_projections, num_edges]
        del w, embeddings  # Free
        
        out = weighted_embeddings.transpose(0, 1)  # [num_edges, num_projections]
        del weighted_embeddings
        
        # Cleanup remaining tensors
        del Xslices, Xslices_sorted, Xind, hyperedge_index_new, hyperedge_index_1_sorted
        del deg_helper, R, pad, hyperedge_index_anchors_1, xnew, ynew
        del D1, D, ptr, P, x, y
        
        # Return compact output - caller handles index mapping
        return out.contiguous(), edges
        

    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)

def interp1d(x, y, xnew, ynew, index):
    """
    Proper linear interpolation using searchsorted - process per projection to save memory
    """
    num_proj, M = xnew.shape
    num_proj, N = x.shape
    
    # Process each projection separately to keep memory manageable
    for proj_idx in range(num_proj):
        x_proj = x[proj_idx]  # [N]
        y_proj = y[proj_idx]  # [N]
        xnew_proj = xnew[proj_idx]  # [M]
        
        # Use searchsorted to find interpolation indices
        # This finds where each xnew value would be inserted in sorted x
        indices = torch.searchsorted(x_proj.contiguous(), xnew_proj.contiguous())
        indices = torch.clamp(indices, 1, N-1)  # Clamp to valid range
        
        # Get bracketing points for linear interpolation
        x0 = x_proj[indices - 1]
        x1 = x_proj[indices]
        y0 = y_proj[indices - 1]
        y1 = y_proj[indices]
        
        # Linear interpolation: y = y0 + (y1-y0) * (x-x0)/(x1-x0)
        alpha = (xnew_proj - x0) / (x1 - x0 + 1e-10)
        ynew[proj_idx] = y0 + alpha * (y1 - y0)
        
        del x_proj, y_proj, xnew_proj, indices, x0, x1, y0, y1, alpha
    
    return ynew

def sparse_sort(values, index):
    """
    Sort values within groups - GPU-native, robust gap
    """
    device = values.device
    N, num_proj = values.shape
    
    # Keep on GPU, avoid .item() sync. Make gap robust for fp16/bf16
    vmin = values.amin()  # GPU tensor
    vmax = values.amax()  # GPU tensor
    gap = (vmax - vmin) + 1.0  # Larger margin for numerical safety
    
    # Normalize values to [0, gap) then add group offset
    normalized_values = values - vmin
    # Use fp32 for sort keys if index is large to avoid precision loss
    sort_keys = index[:, None].float() * gap.float() + normalized_values.float()
    
    # Sort all projections at once
    sorted_keys, sort_idx = torch.sort(sort_keys, dim=0)
    
    # Gather sorted values
    sorted_values = torch.gather(values, 0, sort_idx)
    
    del sort_keys, normalized_values
    return sorted_values, sort_idx