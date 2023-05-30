"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch
#Some auxiliary functions

def window_views_2d(X,H_W,W_W,stride_y,stride_x):
    """Returns views of the last 2 dimensions of an array X. The views are all the 2d rectangles of height=H_W, width=W_W. Different views being obtained by translating this window by the strides.

    H_W,W_W are the height and width of each window (i.e. of the window)
    The first window has the upper left corner at (0,0) the following windows are shifted vertically and horizontally by respectively stride_y and stride_x.
    If H_X is not a multiple of H_W then only the first H_X//H_W elements are used.
    It returns an array of dimensions H_out,W_out,X.shape[:-2],H,W
    the indice of the output have this structure: y position of window, x position of window, indices up to -2, y position within window, x position within window
    the function torch.as_strided should be used with care: it does not throw an error if array bounds are exceeded, insteads it reads the adjacent memory."""
    
    jump_y,jump_x=X.stride()[-2:]  #tells me how many positions I have to traverse in memory to move along axis.
    H_X,W_X=X.shape[-2:]
    H_out=1 + torch.div(H_X-H_W, stride_y,rounding_mode='floor')
    W_out=1 + torch.div(W_X-W_W, stride_x,rounding_mode='floor')
    
    out_shape=(H_out,W_out)+X.shape[:-2]+(H_W,W_W)# plus is concatenation here
    strides=(stride_y*jump_y,stride_x*jump_x)+X.stride()[:-2]+(jump_y,jump_x)
    return torch.as_strided(X,size=out_shape,stride=strides) 


def conv2d_layer(X,W,stride_y,stride_x):
    """Implements a 2d convolution between the last two indices of these arrays, furthermore sums over the channels.

    X is a tensor with 4 indices indicating respectively: sample index mu (n), channel index beta (C_l), row index (H_X), column index (W_X) .For each index the range is reported between parentheses.
    W is a tensor with 4 indices indicating respectively: output channel index alpha (C_lp1), input channel index beta (C_l), row_index (H_W), column index (W_W).
    X input, W convolutional filter, stride_y, stride_x. The output shape will be (n, C_lp1, (H_X-H_W)//stride_y + 1, (W_X-W_W)//stride_x + 1).
    No padding is used.
    """

    C_lp1,_,H_W,W_W=W.shape
    n,C_l,H_X,W_X=X.shape
    H_out=1 + torch.div(H_X-H_W,stride_y ,rounding_mode='floor')
    W_out=1 + torch.div(W_X-W_W,stride_x ,rounding_mode='floor')
    W_lin=W.reshape([C_lp1,C_l*H_W*W_W])
    X_windows=window_views_2d(X,H_W,W_W,stride_y,stride_x)
    X_windows=X_windows.reshape([H_out,W_out,n,C_l*H_W*W_W])
    return torch.einsum('abcd,ed->ceab',X_windows,W_lin)


def average_pool2d(X,H_P,W_P):
    """Performs average pooling over the last two indices of the tensor X.

    H_P, W_P are respectively the height and width of the pooling filter.
    The filter is applied with strides (H_P,W_P) so that each pixel of X contributes to exactly one output pixel.
    Let H_X, W_X be the last two dimensions of X. The shape of the output will be X.shape[:-2] + (H_X//H_P , W_X//W_P). Here '+' stands for concatenation.
    If H_X is not a multiple of H_P then only the first H_X-(H_X % H_P) pixels contribute to the pooling (same for W_X,W_P). 
    """
    H_X,W_X=X.shape[-2:]
    H_out= torch.div(H_X,H_P,rounding_mode='floor')
    W_out=torch.div(W_X,W_P,rounding_mode='floor')
    jump_y,jump_x=X.stride()[-2:]
    out_shape=X.shape[:-2]+(H_out,W_out)+(H_P,W_P)# plus is concatenation here
    strides=X.stride()[:-2]+(H_P*jump_y,W_P*jump_x)+(jump_y,jump_x)
    return torch.mean(torch.torch.as_strided(X,size=out_shape,stride=strides),axis=(-2,-1))


def sample_trunc_norm_neg(beta, mean, std):
    """Samples a truncated normal random variable, from -infinity up to a threshold b.

    'mean' is the expected value of the normal distribution, 'std' is the standard deviation of the normal distribution, beta= (b-mean)/std is the parameter that sets the threshold.
    Notice that one should pass beta as parameter instead of the threshold b.
    beta and mean should have the same shape, which will also be the shape of the output.
    This function only samples when beta > -6 more or less (i.e. in the left tail, the truncation is within 6 standard deviations from the position of the mean). This is due to the floating point precision limit of the gaussian inverse, which gives -inf.
    When, -inf would be returned, the function instead returns b, deterministically.
    Below we summarize the performance of this function
     for type Float64, mean=0, beta=-5, std=1, less than one value in 1e8 is manually truncated (over a sample of 1e8 draws)
     for type Float64, mean=0, beta=-6, std=1, about 4 values in 1e8 are manually truncated (over a sample of 1e8 draws)
     for type Float64, mean=0, beta=-7, std=1, about 2.2 values in 1e4 are manually truncated (over a sample of 1e8 draws)
     for type Float64, mean=0, beta=-8, std=1, about 5 values in 100 are manually truncated (over a sample of 1e8 draws)
     for type Float64, mean=0, beta=-9, std=1, every value is manually truncated (over a sample of 1e8 draws)

     for type Float32, mean=0, beta= 0, std=1,  about 10 values in 1e8 are manually truncated (over a sample of 1e8 draws)
     for type Float32, mean=0, beta=-1, std=1,  about 10 values in 1e8 are manually truncated (over a sample of 1e8 draws)
     for type Float32, mean=0, beta=-2, std=1,  about 65 values in 1e8 are manually truncated (over a sample of 1e8 draws)
     for type Float32, mean=0, beta=-3, std=1,  about 1 values in 1e5 are manually truncated (over a sample of 1e8 draws)
     for type Float32, mean=0, beta=-4, std=1,  about 5 values in 1e4 are manually truncated (over a sample of 1e8 draws)
     for type Float32, mean=0, beta=-5, std=1,  about 5 values in 100 are manually truncated (over a sample of 1e8 draws)
     for type Float32, mean=0, beta=-6, std=1,  every value is manually truncated (over a sample of 1e8 draws)
    """

    U=torch.rand(size=beta.shape)
    sqrt_2=torch.sqrt(torch.tensor(2))
    norm_cdf_beta=0.5+0.5*torch.erf(beta/sqrt_2)
    z=sqrt_2*torch.erfinv(2*U*norm_cdf_beta -1)
    z[z==-torch.inf]=beta[z==-torch.inf]
    return z*std+mean


def sample_trunc_norm_neg_precise(beta, mean, std): 
    """Samples a truncated normal random variable, from -infinity up to a threshold b.

    'mean' is the expected value of the normal distribution, 'std' is the standard deviation of the normal distribution, beta= (b-mean)/std is the parameter that sets the threshold.
    Notice that one should pass beta as parameter instead of the threshold b.
    beta and mean should have the same shape, which will also be the shape of the output.
    This function is a slower but more precise version of 'sample_trunc_norm_neg'. In fact 'sample_trunc_norm_neg' is inaccurate when sampling more that ~6 standard deviations to the left of the mean. 
    The sampling algorithm works in the following way. A first sampling step by direct inversion is made. This is likely to produce a few -torch.inf and other imprecise values when the sample variable is too far from the mean.
    The direct sampling approach is reasonably precise up to 7 (resp. 4) standard deviations left of the mean when using DoubleTensor type (resp. FloatTensor).
    Hence if the first sampling produced values further away than 7,(4) standard deviations, these values are resampled using rejection sampling with rayleigh proposal and with truncation at the minimum between beta and -7,(-4).
    In other words in the rejection sampling we sample from the truncated normal, conditioned on the fact that the sampled variable is smaller than the minimum between beta and -7,(-4).
    This implementation uses an efficient rejection sampler to handle the samples in the tail, on the cpu it takes 3x as much time as 'sample_trunc_norm_neg'.
    The thresholds -7,-4 have been chosen looking at the performance of the functions torch.erf and torch. erfinv. They work well works when the input is greater than 3e-8 - 1 in single precision (corresponds to an output of about 5.3 standard deviations ), 6e-17 -1 for double precision (corresponds to 8.2 standard deviations)
    """
    tail_threshold=torch.tensor(-7) if (mean.type()=='torch.DoubleTensor' or mean.type()==torch.cuda.DoubleTensor) else torch.tensor(-4)  #defines the threshold for the rejection sampling.
    tail_threshold=torch.minimum(tail_threshold,beta)
    tail_threshold_sq=tail_threshold**2
    z=torch.empty(size=beta.shape)
    U=torch.rand(size=beta.shape)
    sqrt_2=torch.sqrt(torch.tensor(2))
    norm_cdf_beta=0.5+0.5*torch.erf(beta/sqrt_2)
    z = sqrt_2*torch.erfinv(2*U*norm_cdf_beta -1)
    mask_tail=(z<=tail_threshold) #-7 for DoubleTensor, -3.5 for FloatTensor. Below this value the direct inversion method is not precise, so we resample using a rejection sampler with Rayleigh proposal.
    while(mask_tail.any()): #when beta<<1 the acceptance fraction goes to 1.
        UV=torch.rand(size=(2,)+beta[mask_tail].shape)
        x=torch.sqrt(tail_threshold_sq[mask_tail]-2*torch.log(1-UV[0]))
        accepted=(UV[1]<=(-tail_threshold[mask_tail]/x))
        z_inf=z[mask_tail]
        z_inf[accepted] = -x[accepted]
        z[mask_tail]=z_inf
        mask_tail[mask_tail.clone()]=torch.logical_and(mask_tail[mask_tail],torch.logical_not(accepted))
    return mean+std*z


def sample_trunc_norm_pos(alpha, mean, std): 
    """Samples a truncated normal random variable, from a threshold a  up to a +infinity.

    'mean' is the expected value of the normal distribution, 'std' is the standard deviation of the normal distribution, alpha= (a-mean)/std is the parameter that sets the threshold.
    Notice that one should pass alpha as parameter instead of the threshold a.
    This version of the function only supports a scalar standard deviation (i.e. the standard deviation should be uniform across coordinates). Moreover beta and mean should have the same shape, which will also be the shape of the output.
    This function only samples when alpha < -35 more or less (i.e. in the right tail, the truncation is within 35 standard deviations from the position of the mean). This is due to the floating point precision limit of the gaussian inverse, which gives inf.
    When, inf would be returned, the function instead returns b, deterministically."""
    
    loc_min = -mean
    return -sample_trunc_norm_neg(beta=-alpha, mean=loc_min, std=std)


def sample_trunc_norm_pos_precise(alpha, mean, std): 
    """Samples a truncated normal random variable, from a threshold a  up to a +infinity.

    'mean' is the expected value of the normal distribution, 'std' is the standard deviation of the normal distribution, alpha= (a-mean)/std is the parameter that sets the threshold.
    Notice that one should pass alpha as parameter instead of the threshold a.
    This version of the function only supports a scalar standard deviation (i.e. the standard deviation should be uniform across coordinates). Moreover beta and mean should have the same shape, which will also be the shape of the output.
    This function only samples when alpha < -35 more or less (i.e. in the right tail, the truncation is within 35 standard deviations from the position of the mean). This is due to the floating point precision limit of the gaussian inverse, which gives inf.
    When, inf would be returned, the function instead returns b, deterministically."""
    
    loc_min = -mean
    return -sample_trunc_norm_neg_precise(beta=-alpha, mean=loc_min, std=std)

# List of available nonlinearities

def identity(x): 
    return x


def ReLU(x):
    return torch.maximum(torch.tensor(0),x)


def truncated_identity(x): #sampling not implemented
    """This is the identity between x=-1 and x=1, and outputs the sign of x outside this range."""
    mask=(torch.abs(x)<1)
    return torch.logical_not(mask)*torch.sign(x)+mask*x


#Update functions for fully connected networks

def sample_W_l_fcl(X_l,b_l,Z_lp1,lambda_W_l,Delta_Z_lp1):
    """Samples the weights W_l, in a fully connected layer

    This function samples W_l conditioned on X_l, Z_lp1, b_l
    The generative model is:
    Z_lp1=X_l@W_l.T+b_l[None,:]+ Normal(0,Delta_Z_lp1)
    All the noise coordinates are i.i.d. and the sum is done element wise.
    Each entry of W_l has prior distribution Normal(0, 1/lambda_W_l)
    Z_lp1 are the pre activations of layer l+1 (n,d_l+1), X_l are the post activations of layer l (n,d_l), b_l is the bias of layer l (d_lp1), W_l is the weight matrix of layer l (d_lp1,d_l) .The dimensions of each array are indicated between parentheses
    n,d_l,d_lp1 are respectively the number of samples, the width of layer l, and the width of layer l+1."""

    d_lp1=Z_lp1.shape[1]
    d_l=X_l.shape[1]
    Cov_W_resc=torch.linalg.inv((X_l.T)@X_l+Delta_Z_lp1*lambda_W_l*torch.eye(d_l)) #the true covariance of each row of W is Cov_W = Cov_W_resc*Delta_Z_lp1. We divide by Delta_Z_lp1 to be regularized when Delta_Z_lp1-->0 (provided that X.T@X is invertible)
    Cholesky_Cov_W_resc=torch.linalg.cholesky(Cov_W_resc)
    m_W=((Cov_W_resc@(X_l.T)@(Z_lp1-b_l[None,:])).T) #mean of the weights
    return torch.sqrt(Delta_Z_lp1)*torch.randn(size=[d_lp1,d_l])@Cholesky_Cov_W_resc.T+m_W


def sample_b_l_fcl(W_l,Z_lp1,X_l,Delta_Z_lp1,lambda_b_l):
    """Samples the biases in a fully conected layer.

    This function samples b_l conditioned of X_l,Z_lp1,W_l.
    The generative model is:
    Z_lp1=X_l@W_l.T+b_l[None,:]+ Normal(0,Delta_Z_lp1)
    All the noise coordinates are i.i.d. and the sum is done element wise.
    b_l has an element wise i.i.d. prior Normal(0,1/lambda_b_l).
    Z_lp1 are the pre activations of layer l+1 (n,d_l+1), X_l are the post activations of layer l (n,d_l), b_l is the bias of layer l (d_lp1), W_l is the weight matrix of layer l (d_lp1,d_l) .The dimensions of each array are indicated between parentheses
    n,d_l,d_lp1 are respectively the number of samples, the width of layer l, and the width of layer l+1."""

    n,d_lp1=Z_lp1.shape
    std_dev_b=torch.sqrt(Delta_Z_lp1/(n+lambda_b_l*Delta_Z_lp1)) #standard deviation of the bias
    m_b=torch.sum(Z_lp1-X_l@(W_l.T),axis=0)/(n+lambda_b_l*Delta_Z_lp1) #mean of the bias
    return std_dev_b*torch.randn(size=[d_lp1])+m_b


def sample_W_b_l_fcl(X_l,Z_lp1,lambda_W_l,lambda_b_l,Delta_Z_lp1):
    """Jointly samples the weights W_l and biases b_l, in a fully connected layer.

    This function samples W_l,b_l conditioned on X_l,Z_lp1
    The generative model is:
    Z_lp1=X_l@W_l.T+b_l[None,:]+ Normal(0,Delta_Z_lp1)
    All the noises coordinates are i.i.d. and the sum is done element wise.
    Each entry of W_l has prior distribution Normal(0, 1/lambda_W_l)
    Each entry of b_l has prior distribution Normal(0,1/lambda_b_l)
    Z_lp1 are the pre activations of layer l+1 (n,d_l+1), X_l are the post activations of layer l (n,d_l), b_l is the bias of layer l (d_lp1), W_l is the weight matrix of layer l (d_lp1,d_l). The dimensions of each array are indicated between parentheses
    n,d_l,d_lp1 are respectively the number of samples, the width of layer l, and the width of layer l+1.
    In the code the bias b_l is treated as the first coordinate of an extended weigth vector of dimension (d_lp1, d_l + 1). In other words the first column is the bias vector
    The function returns W_l, b_l.
    Sampling from the joint distribution should lead to faster mixing, however it's unclear whether this is as fast as the sepaate sampling, in fact the torch.block and torch.stack operations can be expensive."""

    d_lp1=Z_lp1.shape[1]
    n,d_l=X_l.shape
    sum_X_l=torch.sum(X_l,axis=0)[None,:]
    up_block=torch.cat((torch.tensor([[lambda_b_l*Delta_Z_lp1+n]]),sum_X_l),axis=1)
    down_block=torch.cat((sum_X_l.T,(X_l.T)@X_l+Delta_Z_lp1*lambda_W_l*torch.eye(d_l)),axis=1)
    Cov_W_b_resc=torch.linalg.inv(torch.cat((up_block,down_block),axis=0)) #not very elegant alternative to numpy.block
    Cholesky_Cov_W_b_resc=torch.linalg.cholesky(Cov_W_b_resc)
    sum_Z=torch.sum(Z_lp1,axis=0)
    m_W_b=(Cov_W_b_resc @ torch.vstack([sum_Z,X_l.T@Z_lp1])).T #mean of biases and weights
    W_b=torch.sqrt(Delta_Z_lp1)*torch.randn(size=[d_lp1,d_l+1])@(Cholesky_Cov_W_b_resc.T)+m_W_b 
    return W_b[:,1:],  W_b[:,0] #returns respectively the weight matrix and the bias vector


def sample_X_l_fcl(fwd_Z_l,W_l,b_l,Z_lp1,Delta_X_l,Delta_Z_lp1):
    """Samples the postactivations X_l of layer l in a fully connected layer

    This function samples X_l conditioned on fwd_Z_l,W_l,b_l,Z_lp1.
    The generative model is:
    X_l=fwd_Z_l+ Normal(0,Delta_X_l)     
    Z_lp1=X_l@W_l.T+b_l[None,:]+ Normal(0,Delta_Z_lp1)
    All the noises coordinates are i.i.d. and the sum is done element wise.
    In the following the dimensions of each array are indicated between parentheses.
    Z_lp1 are the pre activations of layer l+1 (n,d_l+1), X_l are the post activations of layer l (n,d_l), b_l is the bias of layer l (d_lp1), W_l is the weight matrix of layer l (d_lp1,d_l).
    fwd_Z_l (n,d_l) is the mean of X_l conditioned on Z_l. For example in the case of a non linearity sigma, one has fwd_Z_l=sigma(Z_l), so that X_l=sigma(Z_l)+ Normal(0,Delta_X_l).
    n,d_l,d_lp1 are respectively the number of samples, the width of layer l, and the width of layer l+1."""

    n,d_l=fwd_Z_l.shape
    Cov_X_resc=torch.linalg.inv((W_l.T)@W_l+torch.eye(d_l)*(Delta_Z_lp1/Delta_X_l)) #the covariance of X_l[mu] (it is the same across samples) is Cov_X = Cov_X_resc*Delta_Z_lp1, we divide by Delta_Z_lp1 to be regularized when Delta_Z_lp1-->0
    Cholesky_Cov_X_resc=torch.linalg.cholesky(Cov_X_resc)
    m_X=(fwd_Z_l*(Delta_Z_lp1/Delta_X_l)+(Z_lp1-b_l[None,:])@W_l)@Cov_X_resc
    return torch.sqrt(Delta_Z_lp1)*torch.randn(size=[n,d_l])@(Cholesky_Cov_X_resc.T)+m_X


def sample_Z_lp1_relu(fwd_X_l,X_lp1,Delta_Z_lp1,Delta_X_lp1,precise=True): 
    """Samples Z_lp1, the preactivations of layer l+1, in the case in which X_lp1 = ReLU(Z_lp1) (i.e. the activation function is a ReLU)

    This function samples Z_lp1 conditioned on fwd_X_l, X_lp1.
    The generative model is:
    Z_lp1= fwd_X_l + Normal(0,Delta_Z_lp1) 
    X_lp1= sigma(Z_lp1) + Normal(0,Delta_X_lp1)

    All the noises coordinates are i.i.d. and the sum is done element wise.
    This function can be used for every layer type (e.g. both convolutional and fully connected), in fact the shape of Z_lp1 is inherited from fwd_X_l.
    X_lp1 and fwd_X_l must have the same shape, as it is implied that the ReLU acts in an element wise manner.
    Z_lp1 are the pre activations of layer l+1, X_l are the post activations of layer l, fwd_X_l is the mean of Z_lp1 conditoned on X_l.
    For example in a fully connected layer with weights W_l and biases b_l one has fwd_X_l=X_l@(W_l.T)+b_l[None,:].
    From time to time the variable log_Z_plus_over_Z_minus can give overflow. The overflow should not cause any concern, as it only implies that p_minus=0
    the 'precise' argument specifies which truncated normal sampler to use. If precise==False it will use a sampler that is subject to approximation when the truncation of the normal happens too far from the mean. See the documentation of 'sample_trunc_norm_neg' for more details.
    """
    sample_tn_neg = sample_trunc_norm_neg_precise if precise else sample_trunc_norm_neg
    sample_tn_pos = sample_trunc_norm_pos_precise if precise else sample_trunc_norm_pos

    erf_arg_plus=(Delta_Z_lp1*X_lp1+Delta_X_lp1*fwd_X_l)/torch.sqrt(2*Delta_X_lp1*Delta_Z_lp1*(Delta_X_lp1+Delta_Z_lp1))
    erfc_arg_minus=fwd_X_l/torch.sqrt(2*Delta_Z_lp1)
    Delta_coeff=0.5*torch.log(Delta_X_lp1/(Delta_X_lp1+Delta_Z_lp1))
    p_minus=p_minus=1/(1+torch.exp(Delta_coeff+(erf_arg_plus*erf_arg_plus-erfc_arg_minus*erfc_arg_minus)+torch.log(torch.erf(erf_arg_plus)+1)-torch.log(torch.erfc(erfc_arg_minus))))
    r=torch.rand(p_minus.shape)<p_minus #if r=1 then I have to select the negative portion
    high = -fwd_X_l/torch.sqrt(Delta_Z_lp1)
    low = -(Delta_X_lp1*fwd_X_l+Delta_Z_lp1*X_lp1)/torch.sqrt(Delta_X_lp1*Delta_Z_lp1*(Delta_X_lp1+Delta_Z_lp1))
    Z_lp1 = torch.zeros(fwd_X_l.shape)
    Z_lp1[r==1] = sample_tn_neg(beta=high[r==1], mean = fwd_X_l[r==1], std = torch.sqrt(Delta_Z_lp1))
    Z_lp1[r==0] = sample_tn_pos(alpha=low[r==0], mean=(Delta_X_lp1*fwd_X_l[r==0]+Delta_Z_lp1*X_lp1[r==0])/(Delta_X_lp1+Delta_Z_lp1) , std = torch.sqrt(Delta_X_lp1*Delta_Z_lp1/(Delta_X_lp1+Delta_Z_lp1)))
    return Z_lp1 


def sample_Z_Lp1_multinomial_probit(fwd_X_L,Z_Lp1,y,Delta_Z_Lp1,precise=True): 
    """Samples the last layer's preactivations Z_Lp1, in the case of multiclass classification with the multinomial probit model

    This function samples Z_Lp1 conditioned on fwd_X_L, y.
    The generative model is:
    Z_Lp1 = fwd_X_L + Normal(0,Delta_Z_Lp1).
    y^\mu = argmax_i Z_Lp1^\mu_i. In other words the label is given by the coordinate where Z_Lp1 attains its maximum. (In code this would be written as y=torch.argmax(Z_Lp1,axis=1))
    All the noise coordinates are i.i.d. and the sum is done element wise.
    fwd_X_L is the mean of Z_Lp1 conditioned on X_L. For example in a fully connected layer with weigths W_L and biases b_L one has fwd_X_L=X_L@(W_L.T)+b_L[None,:].

    y= class label belonging to the set (0,1,..,C-1), with C= number of classes. y must be of integer type (use for example " y=y.type(torch.long) " to cast to integer).
    This function should only be used if y is fixed throughout the dynamics (i.e. I'm conditioning on y). The update order depends on y, so if y is a variable, this could prevent the MCMC from sampling the posterior, by introducing a bias.
    For example y represents the vector of training labels, then it is safe to use this functions, since the training labels don't change.
    fwd_X_L must have the same shape as Z_Lp1.
    The function modifies the array fwd_X_L internally.
    """
    sample_tn_neg = sample_trunc_norm_neg_precise if precise else sample_trunc_norm_neg
    sample_tn_pos = sample_trunc_norm_pos_precise if precise else sample_trunc_norm_pos

    n=Z_Lp1.shape[0]
    std_Z=torch.sqrt(Delta_Z_Lp1)  

    #swap the coordinate of the max with the first coordinate (repeat for every sample). We do this to better vectorize the operations.
    tmp_Z=Z_Lp1[:,0].clone()
    Z_Lp1[:,0]=Z_Lp1[torch.arange(n),y]
    Z_Lp1[torch.arange(n),y]=tmp_Z
    tmp_fwd=fwd_X_L[:,0].clone()
    fwd_X_L[:,0]=fwd_X_L[torch.arange(n),y]
    fwd_X_L[torch.arange(n),y]=tmp_fwd
    
    max_Z_excl_y,_=torch.max(Z_Lp1[:,1:],axis=1) #takes the maximum over all the array elements excluded y (which is now in the first coordinate)
    Z_Lp1[:,0]=sample_tn_pos(alpha=(max_Z_excl_y-fwd_X_L[:,0])/std_Z,mean=fwd_X_L[:,0],std=std_Z) #first sample the coordinate corresponding to the maximum, fixing all the other variables
    Z_Lp1[:,1:]=sample_tn_neg(beta=(Z_Lp1[:,0,None]-fwd_X_L[:,1:])/std_Z, mean=fwd_X_L[:,1:],std=std_Z) #then sample the other coordinates, fixing the maximum

    #swap again
    tmp_Z=Z_Lp1[:,0].clone()
    Z_Lp1[:,0]=Z_Lp1[torch.arange(n),y]
    Z_Lp1[torch.arange(n),y]=tmp_Z
    return Z_Lp1   


def sample_W_l_conv2d(X_l,b_l,Z_lp1,lambda_W_l,Delta_Z_lp1,stride_y,stride_x, H_W, W_W):
    """Samples the weights W_l of a convolutional layer.

    This function samples W_l conditioned on Z_lp1,X_l,b_l
    The generative model is:
    Z_lp1= conv2d_layer(X_l,W_l) + b_l + Normal(0,Delta_Z_lp1)
    The prior on each entry of W is i.i.d.  Normal(0, 1/lambda_W_l)
    All the noises coordinates are i.i.d. and the sum is done element wise.

    In the following we indicate the shape of each variable between parentheses
    b_l bias vector (C_lp1). There is one bias per output channel.
    X_l post activations of layer l (n,C_l,H_X_l,W_X_l), W_l weights of layer l (C_lp1,C_l,H_W,W_W), Z_lp1 is the preactivation of layer l+1 (n,C_lp1,H_Z,W_Z).
    For H_Z, W_Z to be valid one must have W_Z=(W_X_l-W_W)//stride_x+1, H_Z=(H_X_l-H_W)//stride_y+1
    n,C_l,C_lp1,H_X_l, W_X_l are respectively the number of samples, number of channels in layer l, number of channels in layer l+1, height of layer l, width of layer l.
    H_W,W_W are respectively the height and width of the convolutional filter W_l, so that W.shape=(C_lp1,C_l,H_W,W_W))
    stride_y, stride_x are respectively the vertical and horizontal strides of the convolution.
    This function possily requires a lot of memory to run. See 'sample_W_l_conv2d_low_mem' for a less memory intensive (but slower) version.
    """

    n,C_lp1,H_Z,W_Z=Z_lp1.shape
    C_l=X_l.shape[1]     

    jump_y,jump_x=X_l.stride()[-2:] #number of positions I have to move in memory to go to the next  3rd, 4th index respectively in X_l.
    X_strided_shape=(n,C_l,H_W,W_W,H_Z,W_Z) 
    X_strides=X_l.stride()+(stride_y*jump_y,stride_x*jump_x)

    X_strided=torch.as_strided(X_l,size=X_strided_shape, stride=X_strides) #this array can occupy a lot of memory
    A_tilde_resc=torch.tensordot(X_strided,X_strided, [[0,4,5],[0,4,5]])
            
    A_resc=A_tilde_resc.reshape([C_l*H_W*W_W,C_l*H_W*W_W]) #when reshaping the innermost index varies the fastest
    A_resc=A_resc+lambda_W_l*Delta_Z_lp1*torch.eye(C_l*H_W*W_W) #adding the weights' prior
    Cov_W_resc=torch.linalg.inv(A_resc) #Cov_W=Cov_W_resc * Delta_Z_lp1. We use this rescaling so that in the limit Delta_Z_lp1-->0 all the quantities are well behaved.
    #In the case l=1 everything up to here can be precomputed as it is constant during the dynamics.
   
    Z_lp1_minus_b_l=Z_lp1-b_l[None,:,None,None]#*torch.ones(Z_lp1.shape)#torch.einsum('abcd,b->abcd',torch.ones([n,C_lp1,H_Z,W_Z]),b_l)

    ZX_tilde=torch.tensordot(Z_lp1_minus_b_l,X_strided,[[0,2,3],[0,4,5]])

    ZX=ZX_tilde.reshape([C_lp1,C_l*H_W*W_W]) 
    m_W=ZX@(Cov_W_resc.T)
    cholesky_Cov_W_resc=torch.linalg.cholesky(Cov_W_resc)
    #torch.random.seed(0)
    return (torch.sqrt(Delta_Z_lp1)*torch.randn(size=[C_lp1,C_l*H_W*W_W])@(cholesky_Cov_W_resc.T)+m_W).reshape([C_lp1,C_l,H_W,W_W])  


def sample_W_l_conv2d_lowmem(X_l,b_l,Z_lp1,lambda_W_l,Delta_Z_lp1,stride_y,stride_x, H_W, W_W):
    """Samples the weights W_l of a convolutional layer.

    A faster version of this function is 'sample_W_l_conv2d',however it requires more memory
    This function samples W_l conditioned on Z_lp1,X_l,b_l.
    The generative model is:
    Z_lp1= conv2d_layer(X_l,W_l) + b_l + Normal(0,Delta_Z_lp1)
    All the noise coordinates are i.i.d. and the sum is done element wise.
    The prior on each entry of W is i.i.d.  Normal(0, 1/lambda_W_l)

    In the following we indicate the shape of each variable between parentheses
    b_l is the bias vector (C_lp1). There is one bias per output channel.
    X_l post activations of layer l (n,C_l,H_X_l,W_X_l), W_l weights of layer l (C_lp1,C_l,H_W,W_W), Z_lp1 is the preactivation of layer l+1 (n,C_lp1,H_Z,W_Z).
    For H_Z, W_Z to be valid one must have W_Z=(W_X_l-W_W)//stride_x+1, H_Z=(H_X_l-H_W)//stride_y+1
    n,C_l,C_lp1,H_X_l, W_X_l are respectively the number of samples, number of channels in layer l, number of channels in layer l+1, height of layer l, width of layer l.
    H_W,W_W are respectively the height and width of the convolutional filter W_l so that  W.shape=(C_lp1,C_l,H_W,W_W))
    stride_y, stride_x are respectively the vertical and horizontal strides of the convolution."""

    C_lp1,H_Z,W_Z=Z_lp1.shape[1:]
    C_l=X_l.shape[1]     
    A_tilde_resc=torch.empty([C_l,H_W,W_W,C_l,H_W,W_W])

    for r_y in range(H_W):
        for r_x in range(W_W):
            for r_y_p in range(H_W):
                for r_x_p in range(r_x+1):# can be made more efficient usind as_strides, but also more memory demanding. 
                    #the following operation is equivalent to einsum with the indices 'abcd,azcd->bz'
                    A_tilde_resc[:,r_y,r_x,:,r_y_p,r_x_p]=torch.tensordot(X_l[:, :, r_y:r_y+stride_y*(H_Z-1)+1:stride_y, r_x:r_x+stride_x*(W_Z-1)+1:stride_x],X_l[:,:, r_y_p:r_y_p+stride_y*(H_Z-1)+1:stride_y, r_x_p:r_x_p+stride_x*(W_Z-1)+1:stride_x],[[0,2,3],[0,2,3]])
                    A_tilde_resc[:,r_y_p,r_x_p,:,r_y,r_x]=(A_tilde_resc[:,r_y,r_x,:,r_y_p,r_x_p].T).clone()

    A_resc=A_tilde_resc.reshape([C_l*H_W*W_W,C_l*H_W*W_W]) #when reshaping the innermost index varies the fastest
    A_resc=A_resc+lambda_W_l*Delta_Z_lp1*torch.eye(C_l*H_W*W_W) #adding the weights' prior
    Cov_W_resc=torch.linalg.inv(A_resc) #Cov_W=Cov_W_resc * Delta_Z_lp1. We use this rescaling so that in the limit Delta_Z_lp1-->0 all the quantities are well behaved.
    #In the case l=1 everything up to here can be precomputed as it is constant during the dynamics.

    m_W=torch.empty([C_lp1,C_l,H_W,W_W])
    ZX_tilde=torch.empty([C_lp1,C_l,H_W,W_W])    
    Z_lp1_minus_b_l=Z_lp1-b_l[None,:,None,None]
    for r_y_p in range(H_W):
        for r_x_p in range(W_W):
            ZX_tilde[:,:,r_y_p,r_x_p]=torch.tensordot(Z_lp1_minus_b_l,X_l[:,:, r_y_p:r_y_p+stride_y*(H_Z-1)+1:stride_y, r_x_p:r_x_p+stride_x*(W_Z-1)+1:stride_x],[[0,2,3],[0,2,3]])
    ZX=ZX_tilde.reshape([C_lp1,C_l*H_W*W_W]) 
    m_W=ZX@(Cov_W_resc.T)
    cholesky_Cov_W_resc=torch.linalg.cholesky(Cov_W_resc)
    return (torch.sqrt(Delta_Z_lp1)*torch.randn(size=[C_lp1,C_l*H_W*W_W])@(cholesky_Cov_W_resc.T)+m_W).reshape([C_lp1,C_l,H_W,W_W])  


def sample_X_l_conv2d(fwd_Z_l,W_l,b_l,Z_lp1,Delta_X_l,Delta_Z_lp1,stride_y,stride_x):
    """Samples X_l, the post activations of layer l, when X_l is followed by a convolutional layer.

    This function samples X_l conditioned on fwd_Z_l, W_l, b_l, Z_lp1.
    The generative model is:
    X_l= fwd_Z_l + Normal(0,Delta_X_l)
    Z_lp1= 2d_convolution(X_l,W_l) + b_l + Normal(0,Delta_Z_lp1)
    All the noises coordinates are i.i.d. and the sum is done element wise.

    fwd_Z_l is the mean of X_l conditioned on Z_l. For example in the case of an element wise nonlinearity sigma, one has fwd_Z_l = sigma(Z_l). This gives X_l= sigma(Z_l) + Normal(0,Delta_X_l).
    X_l inherits the shape of fwd_Z_l.
    The convolution has a filter of respective height and width H_W, W_W, and it is applied with respective vertical and horizontal strides stride_y, stride_x.
    X_l has shape (n,C_l,H_X,W_X), with n being the number of samples, C_l the number of channels, H_X,W_X beig respectively the height and width of the layer.
    W_l is the weight filter of shape (C_lp1,C_l,H_W,W_W), b_l is the bias vector of shape (C_lp1).
    Z_lp1 are the l+1 layer pre activations with shape (n,C_lp1,H_Z,W_Z). H_Z=1+(H_X-H_W)//stride_y, W_Z=1+(W_X-W_X)//stride_x.
    n,C_l,C_lp1,H_X_l, W_X_l are respectively the number of samples, number of channels in layer l, number of channels in layer l+1, height of layer l, width of layer l.
    H_W,W_W are respectively the height and width of the convolutional filter W_l.
    stride_y, stride_x are respectively the vertical and horizontal strides of the convolution.
    """
    H_W,W_W=W_l.shape[-2:]
    H_Z_lp1,W_Z_lp1=Z_lp1.shape[-2:]
    n,C_l,H_X,W_X=fwd_Z_l.shape

    #precomputable stuff
    y_r0_coords=torch.arange(0,stride_y*(H_Z_lp1-1)+1,stride_y)
    x_r0_coords=torch.arange(0,stride_x*(W_Z_lp1-1)+1,stride_x)
    xx_r0,yy_r0=torch.meshgrid(x_r0_coords,y_r0_coords,indexing='ij')
    xx_r0=xx_r0.reshape(-1)
    yy_r0=yy_r0.reshape(-1)

    W_expanded=torch.tensordot(W_l,W_l,[[0],[0]])
    A_tilde=torch.zeros([C_l,H_X,W_X,C_l,H_X,W_X])
    #this must be computed at every iteration
    for r_y in range(H_W):
        for r_x in range(W_W):
            for r_y_p in range(H_W):
                for r_x_p in range(W_W):
                    A_tilde[:,yy_r0+r_y,xx_r0+r_x,:,yy_r0+r_y_p,xx_r0+r_x_p]+=W_expanded[None,:,r_y,r_x,:,r_y_p,r_x_p]
                    #the loops must be executed sequentially, as the views of A_tilde are overlapping (i.e. multiple pointers referring to the same element)
                    #in the setting of asynchronous GPU execution it might be necessary to call torch.cuda.synchronize() at every iteration.


    A_resc=A_tilde.reshape([C_l*H_X*W_X,C_l*H_X*W_X])+torch.eye(C_l*H_X*W_X)*(Delta_Z_lp1/Delta_X_l)

    Cov_X_resc=torch.linalg.inv(A_resc) #the true covariance is Cov_X=Delta_Z_lp1*Cov_X_resc. A_resc is a banded matrix (elements farther than H_W*W_W from the main diagonal are zero), there are probably efficient algorithms to invert it
    Cov_X_resc_part_tilde=Cov_X_resc.reshape([C_l*H_X*W_X,C_l,H_X,W_X])# unpacking only the second index

    fwd_Z_l_reshaped=fwd_Z_l.reshape([n,C_l*H_X*W_X])
    m_X=(Delta_Z_lp1/Delta_X_l)*fwd_Z_l_reshaped@Cov_X_resc
    ZW=torch.tensordot(Z_lp1-b_l[None,:,None,None],W_l,[[1],[0]])#big in memory (maybe not worth it)
    for r_y in range(H_W):
        for r_x in range(W_W):
            m_X+=torch.tensordot(ZW[:,:,:,:,r_y,r_x],Cov_X_resc_part_tilde[:,:,r_y:r_y+stride_y*(H_Z_lp1-1)+1:stride_y,r_x:r_x+stride_x*(W_Z_lp1-1)+1:stride_x],[[3,1,2],[1,2,3]])
    cholesky_Cov_X_resc=torch.linalg.cholesky(Cov_X_resc)
    return (torch.sqrt(Delta_Z_lp1)*torch.randn(size=[n,C_l*H_X*W_X])@(cholesky_Cov_X_resc.T)+m_X).reshape([n,C_l,H_X,W_X])


def sample_X_l_avg_pooling(fwd_Z_l,X_lp1,Delta_X_l,Delta_X_lp1):
    """samples X_l, the layer that gets pooled with average pooling. 

    This function samples X_l conditioned on fwd_Z_l, X_lp1.
    The generative model is:
    X_l = fwd_Z_l + Normal(0,Delta_X_l)
    X_lp1 = average_pooling(X_l) + Normal(0,Delta_X_lp1)
    All the noise coordinates are i.i.d. and the sum is done element wise.

    In the following we indicate array shapes between parentheses.
    X_l is the post activation of layer l (n,C_l,H_X_l,W_X_l), X_lp1 is the output of the pooling layer (n,C_l,H_X_lp1,W_X_lp1).
    fwd_Z_l is the mean of X_l conditioned on Z_l. For example in the case of a nonlinearity sigma one would have fwd_X_l=sigma(Z_l).
    The shape of X_l is inherited from fwd_Z_l.
    n,C_l,H_X_l, W_X_l are respectively the number of samples, number of channels in layer l, height of layer l, width of layer l.
    From X_l and X_lp1 we infer the height and width of the pooling filter respectively as H_X_l//H_X_lp1 and W_X_l//W_X_lp1.
    If the dimension of layer l+1 (H_X_lp1) is not a multiple of the dimension of layer l (H_X_l), then the pooling acts only on the first  (H_X_l//H_X_lp1)*H_X_lp1  pixels of layer l. Same for the horizontal dimension.
    The remaining H_X_l % H_X_lp1 pixels (those that don't take part in the pooling), are correctly sampled independently from Normal(fwd_Z_l,Delta_X_l).
    We only consider the case where the stride is equal to the filter dimension (i.e. each input pixel (in layer l) belongs to the receptive field of exactly one pixel in the output (layer l+1))."""

    n,C_l,H_X_lp1,W_X_lp1=X_lp1.shape #number of channels should match between Z_l and X_lp1
    H_X_l,W_X_l=fwd_Z_l.shape[2:]
    #these are the dimensions of the pooling filter (inferred from the dimensions of the inputs)
    H_W=torch.div(H_X_l,H_X_lp1,rounding_mode='floor')
    W_W=torch.div(W_X_l,W_X_lp1,rounding_mode='floor')
    
    q=(1-torch.sqrt(H_W*W_W*Delta_X_lp1/(Delta_X_l+Delta_X_lp1*H_W*W_W)))/(H_W*W_W)
    p1=Delta_X_l/(Delta_X_l+H_W*W_W*Delta_X_lp1)
    X_l=torch.sqrt(Delta_X_l)*torch.randn(size=[n,C_l,H_X_l,W_X_l])
    
    pooled_sigma_Z_l=average_pool2d(fwd_Z_l,H_W,W_W)
    m_X_l=p1*(X_lp1-pooled_sigma_Z_l) #m_X_l+sigma_Z_l_block is the block mean, however we prefer to add sigma(Z_l) afterwards to treat also those pixels that are left out of the pooling

    jump_y,jump_x=X_l.stride()[-2:]

    strided_shape_X=X_l.shape[:-2]+(H_X_lp1,W_X_lp1,H_W,W_W)

    strides_X=X_l.stride()[:-2]+(H_W*jump_y,W_W*jump_x,jump_y,jump_x)
    X_strided=torch.torch.as_strided(X_l,size=strided_shape_X,stride=strides_X)
    X_strided+=-q*torch.sum(X_strided,axis=(4,5))[:,:,:,:,None,None]+m_X_l[:,:,:,:,None,None] #one must not reassign X_strided, otherwise modifications to X_strided will not be passed to X_l
    return X_l+fwd_Z_l


def sample_b_l_conv2d(X_l,Z_lp1,W_l,Delta_Z_lp1,lambda_b_l,stride_y,stride_x):
    """Samples the bias vector b_l in a convolutional layer. 

    This function samples b_l conditioned on X_l,Z_lp1,W_l.
    The generative model is:
    Z_lp1= 2d_convolution(X_l,W_l) + b_l + Normal(0,Delta_Z_lp1). 
    All the noise coordinates are i.i.d. and the sum is done element wise.
    Each coordinate of the bias has i.i.d. prior Normal(0, 1/lambda_b_l)
    The convolution operator is applied with respective vertical and horizontal strides, stride_y, stride_x.

    In the following we indicate shapes between parentheses.
    There is one bias per output channel so b_l has shape (C_lp1).
    X_l is the post activation of layer l (n,C_l,H_X_l,W_X_l), W_l are the weights of layer l (C_lp1,C_l,H_W,W_W), Z_lp1 is the preactivation of layer l+1 (n,C_lp1,H_Z,W_Z).
    For shapes to be valid one must have  H_Z=1+(H_X-H_W)//stride_y, W_Z=1+(W_X-W_X)//stride_x.
    n,C_l,C_lp1,H_X_l, W_X_l are respectively the number of samples, number of channels in layer l, number of channels in layer l+1, height of layer l, width of layer l.
    H_W,W_W are respectively the height and width of the convolutional filter W_l."""
    n,C_lp1,H_Z,W_Z=Z_lp1.shape
    fwd_conv2d=conv2d_layer(X_l,W_l,stride_y,stride_x)
    m_b=torch.sum(Z_lp1-fwd_conv2d,axis=(0,2,3))/(n*H_Z*W_Z+Delta_Z_lp1*lambda_b_l)
    return torch.sqrt(Delta_Z_lp1/(Delta_Z_lp1*lambda_b_l+n*H_Z*W_Z))*torch.randn(size=[C_lp1])+m_b

# Update functions for special cases (e.g. faster implementations for first of last layers)

def sample_W_1_fcl(Z_2,b_1,Cholesky_Cov_W_resc,Cov_W_resc_XT,Delta_Z_2): 
    """Samples the first layer weights in a fully connected layer.

    This function samples W_1 conditioned on X, Z_2, b_1
    The generative model is:
    Z_2=X@W_1.T+b_1[None,:]+ Normal(0,Delta_Z_2)
    Each entry of W_1 has a prior Normal(0, 1/lambda_W_1).
    All the noise coordinates are i.i.d. and the sum is done element wise.

    We indicate the shapes of arrays between parentheses
    X is the input data matrix (n,d), W_1 is the matrix of first layer weights (d_2,d),b_1 is the bias of the first layer (d_2), Z_2 are the preactivatinos of the second layer (n,d_2)
    n,d,d_2 are respectiively the number of training samples, the input dimenson and the dimension of the second layer.
    Exploits the fact that the  covariance is fixed (since the input data X is constant) during the dynamics to speed up the sampling (which amounts in this case to a matrix multiplication).
    If Cov_W is the covariance of each row of W_1 (all the rows have thew same covariance), then the rescaled covariance is Cov_W_resc=Cov_W/Delta_Z_2
    The cholesky decomposition of Cov_W_resc should be passed as argument together with Cov_W_resc_XT= Cov_W_resc @ (X.T).

    In summary before running the code one should have executed the following lines of code:

    "
    Cov_W_resc=torch.linalg.inv(X.T@X+Delta_Z_2*lambda_W*torch.eye(d))
    Cov_W_resc_XT=Cov_W_resc@(X.T)
    Cholesky_Cov_W_resc= torch.linalg.cholesky(Cov_W_resc)
    "
    """
    d_2=Z_2.shape[1]
    d=Cholesky_Cov_W_resc.shape[0]
    return (torch.sqrt(Delta_Z_2)*Cholesky_Cov_W_resc@torch.randn(size=[d,d_2])+Cov_W_resc_XT@(Z_2-b_1[None,:])).T


def sample_X_L_fcl(fwd_Z_L,W_L,b_L,Z_Lp1,Delta_X_L,Delta_Z_Lp1): 
    """Samples X_L the post activations of the last layer (L), in a regression setting.

    Samples X_L conditioned on fwd_Z_L, W_L, b_L, Z_Lp1.
    The generative model is:
    X_L= fwd_Z_L+ Normal(0, Delta_X_L)
    Z_Lp1 = X @ W_L.T + b_L + Normal(0, Delta_y)
    All the noise coordinates are i.i.d. and the sum is done element wise.

    This is a particular case of the function 'sample_X_l_fcl', in which W_l has dimensions (1,d_l). This allows for much more efficient computation of the covariance.
    For example this function can be used in a regression setting, where Z_Lp1 has shape (n,1) and represents the labels.

    fwd_Z_L is the mean of X_L conditioned on Z_L.
    W_L has dimensions (1, d_L), fwd_Z_L has dimensions (n, d_L), b_L has dimensions (1,), Z_Lp1 has dimensions (n,1).
    n,d_L are respectively the number of training samples and the width of layer L.
    W_L cannot be zero when computing this function (otherwise it gives nan), hence it's preferable to use a small random initialization (or to update W_l before calling this function)."""
    a=W_L.reshape(-1)
    sq_norm_a=torch.inner(a,a)
    q=(1-torch.sqrt(1-sq_norm_a/(sq_norm_a+Delta_Z_Lp1/Delta_X_L)))/sq_norm_a
    m_Phi=torch.outer((Z_Lp1[:,0]-b_L-fwd_Z_L@a)/(sq_norm_a+Delta_Z_Lp1/Delta_X_L),a)+fwd_Z_L
    tmp_rand=torch.sqrt(Delta_X_L)*torch.randn(size=fwd_Z_L.shape)
    return tmp_rand-q*torch.outer(tmp_rand@a,a)+m_Phi


def sample_W_b_1_fcl(Cholesky_Cov_W_b_resc, Cov_W_b_resc_XT,Z_2,Delta_Z_2):
    """Jointly samples the first layer weights W_1 and biases b_1, in a fully connected layer.

    This function samples W_1,b_1 conditioned on X (the training set) and Z_2 (the second layer preactivations)
    The generative model is:
    Z_2=X@W_1.T+b_1[None,:]+ Normal(0,Delta_Z_2)
    All the noises coordinates are i.i.d. and the sum is done element wise.
    Each entry of W_1 has prior distribution Normal(0, 1/lambda_W_1)
    Each entry of b_1 has prior distribution Normal(0,1/lambda_b_1)

    Z_2 are the pre activations of layer 2 (n,d_2), X are the input data (n,d), b_1 is the bias of the first layer (d_2), W_1 is the weight matrix of layer 1 (d_2,d). The dimensions of each array are indicated between parentheses
    n, d, d_2 are respectively the number of samples, the input dimension, and the width of layer 2.
    In the code the bias b_1 is treated as the first coordinate of an extended weigth vector of dimension (d_2, d + 1). In other words the first column is the bias vector.
    The function returns W_1, b_1.

    This function is a special case of the function 'sample_W_b_l_fcl'. This implementation exploits the fact that the input is constant through training and therefore the covariance matrix of W_1,b_1 is constant. This allows a significant speedup.
    Cov_W_b_resc = Cov_W_b/Delta_Z_2 is the rescaled covariance matrix of the augmented weight vector (b_[i], W[i]) of dimension d+1. All the rows of (b,W) have in fact the same covariance. The true covariance is here indicated with Cov_W_b.
    This is related to the argument one must pass through Cov_W_b_resc_XT = Cov_W_b_resc @ torch.cat((torch.ones([n,1]),X),axis=1).T 
    One also needs to pass the cholesky decomposition of Cow_W_b_resc.

    In summary before running the dynamics one should execute the following code

    "
    n,d=X.shape
    sum_X=torch.sum(X,axis=0)[None,:]
    up_block=torch.cat((torch.tensor([[lambda_b_1*Delta_Z_2+n]]),sum_X),axis=1)
    down_block=torch.cat((sum_X.T,(X.T)@X+Delta_Z_2*lambda_W_1*torch.eye(d)),axis=1)
    Cov_W_b_resc=torch.linalg.inv(torch.cat((up_block,down_block),axis=0))
    Cholesky_Cov_W_b_resc=torch.linalg.cholesky(Cov_W_b_resc) #<---- must pass as first argument
    Cov_W_b_resc_XT = Cov_W_b_resc @ torch.cat((torch.ones([n,1]),X),axis=1).T #<---- must pass this as second argument
    "
    Sampling from the joint distribution should lead to faster mixing, however it's unclear whether this is as fast as the separate sampling, in fact the torch.cat and torch.stack operations can be expensive."""

    d_lp1=Z_2.shape[1]
    d_l=(Cholesky_Cov_W_b_resc.shape[0]) - 1
    m_W_b=(Cov_W_b_resc_XT@Z_2).T #mean of biases and weights
    W_b=torch.sqrt(Delta_Z_2)*torch.randn(size=[d_lp1,d_l+1])@(Cholesky_Cov_W_b_resc.T)+m_W_b 
    return W_b[:,1:],  W_b[:,0] #returns respectively the weight matrix and the bias vector


def sample_W_1_conv2d(Cholesky_Cov_W_resc, Cov_W_resc_XT,b_1,Z_2,Delta_Z_2, H_W, W_W, C_1):
    """Samples the first layer weights W_1 of a convolutional layer.

    Samples W_1 conditioned on X,Z_2,b_1.
    The generative model is:
    Z_2= conv2d_layer(X,W_1) + b_1 + Normal(0,Delta_Z_2)
    The prior on each entry of W_1 is i.i.d.  Normal(0, 1/lambda_W_1)
    All the noises coordinates are i.i.d. and the sum is done element wise.

    This function exploits the fact that X is constant throughout the dynamics (X is the input) to precompute some quantities, and is hence faster (about 10x) than 'sample_W_l_conv2d'.    

    Additional details about the model:
    In the following we indicate the shape of each variable between parentheses
    b_1 is the bias vector (C_2). There is one bias per output channel.
    X is the input data (n,C_1,H_X,W_X), W_1 weights of the first layer (C_2,C_1,H_W,W_W), Z_2 is the preactivation of layer 2 (n,C_2,H_Z,W_Z).
    For H_Z, W_Z to be valid one must have W_Z=(W_X-W_W)//stride_x+1, H_Z=(H_X-H_W)//stride_y+1
    n,C_1,C_2,H_X, W_X are respectively the number of samples, number of channels in the input, number of channels in layer 2, height of the input, width of the input.
    H_W,W_W are respectively the height and width of the convolutional filter W_1 (i.e. W_1[0,0].shape=(H_W,W_W))
    stride_y, stride_x are respectively the vertical and horizontal strides of the convolution.

    The precomputed quantities to pass can be computed once before starting the dynamics, by executing the following lines of code:

    "
    n,C_2,H_Z,W_Z=Z_2.shape
    C_1=X.shape[1]     

    jump_y,jump_x=X.stride()[-2:] #number of positions I have to move in memory to go to the next  3rd, 4th index respectively in X_l.
    X_strided_shape=(n,C_1,H_W,W_W,H_Z,W_Z) 
    X_strides=X.stride()+(stride_y*jump_y,stride_x*jump_x)

    X_strided=torch.as_strided(X,size=X_strided_shape, stride=X_strides)
    A_tilde_resc=torch.tensordot(X_strided,X_strided, [[0,4,5],[0,4,5]])
    
    A_resc=A_tilde_resc.reshape([C_1*H_W*W_W,C_1*H_W*W_W]) 
    A_resc=A_resc+lambda_W_1*Delta_Z_2*torch.eye(C_1*H_W*W_W)
    Cov_W_resc=torch.linalg.inv(A_resc)  
    Cholesky_Cov_W_resc=torch.linalg.cholesky(Cov_W_resc)  #<---should be passed as argument
    Cov_W_resc_XT=torch.tensordot(Cov_W_resc.reshape([C_1*H_W*W_W,C_1,H_W,W_W]),X_strided,[[1,2,3],[1,2,3]]) #<--- should be passed as argument
    "

    """
    C_2=Z_2.shape[1]
    Z_2_minus_b_1=Z_2-b_1[None,:,None,None]
    m_W=torch.tensordot(Z_2_minus_b_1,Cov_W_resc_XT,[[0,2,3],[1,2,3]])
    return (torch.sqrt(Delta_Z_2)*torch.randn(size=[C_2,C_1*H_W*W_W])@(Cholesky_Cov_W_resc.T)+m_W).reshape([C_2,C_1,H_W,W_W])  


####END MAIN FUNCTIONS####

# Notable variants which can give marginal speedups

def sample_X_l_conv2d_strides(sigma,Z_l,W_l,b_l,Z_lp1,Delta_X_l,Delta_Z_lp1,stride_y,stride_x):
    """Samples X_l, the post activations of layer l, when X_l is followed by a convolutional layer.

    This function samples X_l conditioned on fwd_Z_l, W_l, b_l, Z_lp1.
    The generative model is:
    X_l= fwd_Z_l + Normal(0,Delta_X_l)
    Z_lp1= 2d_convolution(X_l,W_l) + b_l + Normal(0,Delta_Z_lp1)
    All the noises coordinates are i.i.d. and the sum is done element wise.

    This function is an alternative to 'sample_X_l_conv2d'. 
    This variant employs torch.as_strided to marginally speed up some of the computation; however the matrix inversion (which is not sped up) is what dominates the overall computational time, hence the gains are limited.

    fwd_Z_l is the mean of X_l conditioned on Z_l. For example in the case of an element wise nonlinearity sigma, one has fwd_Z_l = sigma(Z_l). This gives X_l= sigma(Z_l) + Normal(0,Delta_X_l).
    X_l inherits the shape of fwd_Z_l.
    The convolution has a filter of respective height and width H_W, W_W, and it is applied with respective vertical and horizontal strides stride_y, stride_x.
    X_l has shape (n,C_l,H_X,W_X), with n being the number of samples, C_l the number of channels, H_X,W_X beig respectively the height and width of the layer.
    W_l is the weight filter of shape (C_lp1,C_l,H_W,W_W), b_l is the bias vector of shape (C_lp1).
    Z_lp1 are the l+1 layer pre activations with shape (n,C_lp1,H_Z,W_Z). H_Z=1+(H_X-H_W)//stride_y, W_Z=1+(W_X-W_X)//stride_x.
    n,C_l,C_lp1,H_X_l, W_X_l are respectively the number of samples, number of channels in layer l, number of channels in layer l+1, height of layer l, width of layer l.
    H_W,W_W are respectively the height and width of the convolutional filter W_l.
    stride_y, stride_x are respectively the vertical and horizontal strides of the convolution.
    """
    C_l,H_W,W_W=W_l.shape[1:]
    n,_,H_Z_lp1,W_Z_lp1=Z_lp1.shape
    H_X,W_X=Z_l.shape[-2:]
    
    W_expanded=torch.tensordot(W_l,W_l,[[0],[0]])
    A_tilde=torch.zeros([C_l,H_X,W_X,C_l,H_X,W_X])
    
    A_strided_shape=(H_W,W_W,H_W,W_W,H_Z_lp1,W_Z_lp1,C_l,C_l)
    s0,s1,s2,s3,s4,s5=A_tilde.stride()
    A_tilde_strides=(s1,s2,s4,s5,stride_y*(s1+s4),stride_x*(s2+s5),s0,s3)
    A_tilde_strided=torch.as_strided(A_tilde,size=A_strided_shape,stride=A_tilde_strides)
    for r_y in range(H_W): #do NOT attempt to vectorize this for loop! A_tilde_strided contains views so one would be several references to the same element in parallel.
        for r_x in range(W_W):
            for r_y_p in range(H_W):
                for r_x_p in range(W_W):
                    A_tilde_strided[r_y,r_x,r_y_p,r_x_p,:,:,:,:]+=W_expanded[None,None,:,r_y,r_x,:,r_y_p,r_x_p]
                    
    A_resc=A_tilde.reshape([C_l*H_X*W_X,C_l*H_X*W_X])+torch.eye(C_l*H_X*W_X)*(Delta_Z_lp1/Delta_X_l)
    Cov_X_resc=torch.linalg.inv(A_resc) #the true covariance is Cov_X=Delta_Z_lp1*Cov_X_resc. A_resc is a banded matrix (elements farther than H_W*W_W from the main diagonal are zero), there are probably efficient algorithms to invert it
    Cov_X_resc_part_tilde=Cov_X_resc.reshape([C_l*H_X*W_X,C_l,H_X,W_X])# unpacking only the second index
    
    Z_l_resh=Z_l.reshape([n,C_l*H_X*W_X])
    m_X=(Delta_Z_lp1/Delta_X_l)*sigma(Z_l_resh)@Cov_X_resc #no need to transpose because Cov_X_resc should be symmetric.
    ZW=torch.tensordot(Z_lp1-b_l[None,:,None,None],W_l,[[1],[0]]) #big in memory (maybe not worth it)
    for r_y in range(H_W):
        for r_x in range(W_W):
            m_X+=torch.tensordot(ZW[:,:,:,:,r_y,r_x],Cov_X_resc_part_tilde[:,:,r_y:r_y+stride_y*(H_Z_lp1-1)+1:stride_y,r_x:r_x+stride_x*(W_Z_lp1-1)+1:stride_x],[[3,1,2],[1,2,3]])
    cholesky_Cov_X_resc=torch.linalg.cholesky(Cov_X_resc)
    return (torch.sqrt(Delta_Z_lp1)*torch.randn(size=[n,C_l*H_X*W_X])@(cholesky_Cov_X_resc.T)+m_X).reshape([n,C_l,H_X,W_X])
