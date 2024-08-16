import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
import threading
import copy
device = "cuda"
model = None
loss_fn = None

def get_val_images(testset, num_points):
    images = testset[0]
    labels = testset[1]
    rand = torch.randint(images.shape[0], (num_points,))
    val_images, val_labels = images[rand], labels[rand]
    return val_images, val_labels

def get_val_images_batch(testset, batch_size):
    images = testset[0]
    labels = testset[1]
    num_batches = images.shape[0]//batch_size
    rand = torch.randint(num_batches, ())
    a = rand*batch_size
    b = rand*batch_size+batch_size
    val_images, val_labels = images[a:b], labels[a:b]
    return val_images, val_labels
            
def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    out = functional_call(model, (params, buffers), (batch,))
    print("out", sample.shape, batch.shape, out.shape, "targets", targets.shape, targets)

    loss = loss_fn(out, targets)
    return loss


def cat(x, batch_size):
    return  torch.cat([x[key].view(batch_size, -1) for key in x.keys()], dim=1)

def importance_sampling(prev_subset, current_batch, args):
    sampling_mode = args["sampling_mode"]
    
    images, prev_images = current_batch["images"], prev_subset["images"]
    labels, prev_labels = current_batch["labels"], prev_subset["labels"]
    features, prev_features = current_batch["features"], prev_subset["features"]
    size_at = 0 if prev_images is None else prev_images.shape[0]
    batch_size_curr = images.shape[0]
    len_opt = 0
    # change is here
    if(prev_images is None or size_at == 0): sampling_mode = None
    if(sampling_mode == "Uniform"):
        lamb = args["lamb"]
        frac_at = int(batch_size_curr*(1-lamb))
        frac_curr = batch_size_curr - frac_at
        prev_sample = torch.randint(size_at, (frac_at,))  
        curr_sample = torch.randint(batch_size_curr, (frac_curr,))  
        ret_images = torch.cat((prev_images[prev_sample], images[curr_sample]), dim=0)
        ret_labels = torch.cat((prev_labels[prev_sample], labels[curr_sample]), dim=0)
        ret_features = torch.cat((prev_features[prev_sample], features[curr_sample]), dim=0)
        len_opt = prev_sample.shape[0]
        # print("ret", ret_images.shape[0])
        return {"images": ret_images, "labels": ret_labels, "features": ret_features}, len_opt
    
    if(sampling_mode == "Binomial"):
        raise NotImplementedError
    
    if(sampling_mode == None):
        len_opt = current_batch["images"].shape[0]
        return {"images": images, "labels":labels, "features":features}, 0
    
def importance_sampling_v2(pmodel, ploss_fn, prev_subset, current_batch, testset, args):
    images, prev_images = current_batch["images"], prev_subset["images"]
    labels, prev_labels = current_batch["labels"], prev_subset["labels"]
    features, prev_features = current_batch["features"], prev_subset["features"]
    size_at = 0 if prev_images is None else prev_images.shape[0]
    batch_size_curr = images.shape[0]
    num_val_points = args["num_val_points"]
    eta_n = args["eta_n"]
    lamb = args["lamb"]
    sampling_mode = args["sampling_mode"]
    
    # max_size = 1024
    # curr_at_max = min(max_size, size_at+batch_size_curr)
    # print("curr max", curr_at_max)
    # selection = torch.rand(curr_at_max) <= lamb
    # at_select = selection[:size_at]
    # curr_select = selection[size_at+1:]
    
    # if(size_at == 0):
    #     cat_images = current_batch
    # else:
    #     cat_images = {
    #         k : torch.cat((prev_subset[k], current_batch[k]), dim=0) for k in current_batch.keys()
    #     }
    
    # len_opt = size_at
    
    # subset_grads, val_grads = calc_grads(pmodel, ploss_fn, cat_images, testset, num_val_points)
    # num = min(100, cat_images["images"].shape[0])
    # bucket_grads = None
    # bucket_idxs = []
    # for step in range(num):
    #     metricOver = [[i for i in range(0, len(cat_images["images"]))]]
    #     metricList = calc_metric(metricOver, eta_n, subset_grads, val_grads, bucket_grads, reduction=None)
    #     top_idx = torch.argmax(metricList, dim=1)
    #     bucket_idxs.append(top_idx.item())
    #     if bucket_grads is None:
    #         bucket_idxs.append(top_idx)
    #         bucket_grads = subset_grads[top_idx:top_idx+1]
            
    #     else:
    #         bucket_grads = torch.cat((bucket_grads, subset_grads[top_idx:top_idx+1]), dim=0)
        
    # print("new bach", bucket_idxs)
    # print("*************")
    # subset = {
    #     "images": cat_images["images"][bucket_idxs],
    #     "features": cat_images["features"][bucket_idxs],
    #     "labels": cat_images["labels"][bucket_idxs],
    # }
    
    len_opt = 0
    # change is here
    if(prev_images is None or size_at == 0): sampling_mode = None
    if(sampling_mode == "Uniform"):
        # lamb = args["lamb"]
        frac_at = int(batch_size_curr*lamb)
        frac_curr = batch_size_curr - frac_at
        # frac_at = int(curr_at_max*lamb)
        # frac_curr = curr_at_max - frac_at
        prev_sample = torch.randint(size_at, (frac_at,))  
        curr_sample = torch.randint(batch_size_curr, (frac_curr,))  
        ret_images = torch.cat((prev_images[prev_sample], images[curr_sample]), dim=0)
        ret_labels = torch.cat((prev_labels[prev_sample], labels[curr_sample]), dim=0)
        ret_features = torch.cat((prev_features[prev_sample], features[curr_sample]), dim=0)
        len_opt = prev_sample.shape[0]
        imp = {"images": ret_images, "labels": ret_labels, "features": ret_features}
    
    elif(sampling_mode == "Binomial"):
        raise NotImplementedError
    
    elif(sampling_mode == None):
        len_opt = current_batch["images"].shape[0]
        imp = {"images": images, "labels":labels, "features":features}
    
    
    # if(size_at == 0):
    #     cat_images = current_batch
    # else:
    #     cat_images = {
    #         k : torch.cat((prev_subset[k], current_batch[k]), dim=0) for k in current_batch.keys()
    #     }
    
    
    subset_grads, val_grads = calc_grads(pmodel, ploss_fn, imp, testset, num_val_points)
    num = min(50, imp["images"].shape[0])
    bucket_grads = None
    bucket_grads_idxs = []
    opt_grads_idxs = None
    bucket_idxs = []
    for step in range(num):
        metricOver = [[i for i in range(0, len(imp["images"]))]]
        metricList = calc_metric(metricOver, eta_n, subset_grads, val_grads, bucket_grads, reduction=None)
        top_idx = torch.argmax(metricList, dim=1)
        # _, idxs = torch.topk(metricList, metricList.shape[1], dim=1)
        # for val in idxs[0]:
        #     if val not in bucket_idxs:
        #         top_idx = val
        #         bucket_idxs.append(val.item())
        bucket_idxs.append(top_idx.item())
        if(top_idx.item() < size_at):
            if(opt_grads_idxs is None): opt_grads_idxs = [top_idx.item()]
            else: opt_grads_idxs.append(top_idx.item())
        if bucket_grads is None:
            bucket_grads = subset_grads[top_idx:top_idx+1]
        else:
            bucket_grads = torch.cat((bucket_grads, subset_grads[top_idx:top_idx+1]), dim=0)
        
    # print("new bach", bucket_idxs)
    # print("*************")
    # print("grads", subset_grads.shape, val_grads.shape)
    # print("opt idxs", opt_grads_idxs)
    imp = {
        "images": imp["images"][bucket_idxs],
        "features": imp["features"][bucket_idxs],
        "labels": imp["labels"][bucket_idxs],
    }
    opt_grads = subset_grads[opt_grads_idxs] if opt_grads_idxs is not None else None
    # print("imp shape", imp['images'].shape, subset_grads[opt_grads_idxs].shape)
    return imp, opt_grads
    
    
def get_greedy_list(funcs, submod_budget, multi_thread=False, optimizer="NaiveGreedy"):
    greedyList = {}
    def thread(x):
        # Maximize the function
        f = funcs[x]
        _greedy = submod_maximize(f, submod_budget, optimizer=optimizer)
        greedyList[x] = _greedy
    
    if multi_thread:
        '''Multi-threading'''
        threads = [threading.Thread(target=thread, args=(i,)) for i in range(len(funcs))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        
        '''Multi-processing'''
        # pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        # [pool.submit(thread, i) for i in range(len(funcs))]
        # pool.shutdown(wait=True)
    else:
        for i,f in enumerate(funcs):
            # Maximize the function
            _greedy = submod_maximize(f, submod_budget, optimizer=optimizer)
            greedyList[i] = _greedy
    return greedyList

def eps_greedy_composition(model, imp_sample, len_opt, testset, loss_fn, step, funcs, submod_budget, moment_sum, args, optimizer="NaiveGreedy", greedyOnly=False, opt_grads=None):
    lamb = args["lamb"]
    pi = args["pi"]
    thresh = step/((step+lamb)**pi)
    eps = torch.rand(1).item()
    greedyList = get_greedy_list(funcs, submod_budget, optimizer)
    # if(eps > thresh or greedyOnly):
    if(True):
        return best_submod_bandit(model, imp_sample, len_opt, greedyList, testset, loss_fn, args["eta_n"], moment_sum, args["num_val_points"], opt_grads)
    else:
        sample = torch.randint(len(greedyList), ()).item()
        return greedyList[sample], sample
    

def submod_maximize(f, budget, optimizer="NaiveGreedy"):
    return f.maximize(budget = budget, optimizer=optimizer, 
                stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, 
                verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
    
def best_submod_bandit__OLD__(pmodel, subset, len_opt, greedyList, testset, ploss_fn, eta_n, moment_sum, num_val_points):
    best_index = 0
    best_metric = -10000000000
    global model, loss_fn
    model = pmodel
    cached_state_dict = copy.deepcopy(model.state_dict())
    clone_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(clone_dict)
    model.eval()
    loss_fn = ploss_fn
    # Per-sample gradients
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    # print("keys", [k for k in params.keys()], [k for k in buffers.keys()])
    subset_grads = ft_compute_sample_grad(params, buffers, subset["images"], subset["labels"].to(device))
    val_images, val_labels = get_val_images(testset, num_val_points)
    val_grads = ft_compute_sample_grad(params, buffers, val_images, val_labels)
    
    torch.cuda.empty_cache()
    model.load_state_dict(cached_state_dict)
    model.train()
    
    subset_size = subset["images"].shape[0]
    subset_grads = cat(subset_grads, subset_size) # B,P
    val_grads = cat(val_grads, val_images.shape[0]) # Bv,P
    
    alpha = 0.7
    # moment_sum = alpha*(val_grads*val_grads) + (1-alpha)*moment_sum
    with torch.autocast(device_type=device):
        for i in range(len(greedyList)):
            submod_indices = [greedyList[i][j][0] for j in range(len(greedyList[i]))]
            t = torch.mean(val_grads, dim=0, keepdim=True)
            term1 = eta_n*subset_grads[submod_indices]@(t.transpose(0,1)) # B',1
            # print("term1", term1.shape)
            term2 = 0
            if(len_opt != 0):
                grad_sum = torch.sum(subset_grads[0:len_opt], dim=0, keepdim=True)
                # eps = 1e-5
                hessian = torch.ones(grad_sum.transpose(0,1).shape).to(device)
                # assert torch.equal(grad_sum.transpose(0,1), hessian*grad_sum.transpose(0,1)) == True
                # hessian = (moment_sum).transpose(0,1)
                term2 = eta_n*eta_n*subset_grads[submod_indices]@((hessian*(grad_sum.transpose(0,1)))) # B',1
            metric =  torch.sum(term1 - term2, dim=0)/subset_size
            metric = torch.mean(metric, dim=0)
            if(metric.item() > best_metric):
                best_metric = metric.item()
                best_index = i
    return greedyList[best_index], best_index


# CACHED only last layer grads
def calc_grads_cached(pmodel, ploss_fn, subset, testset, num_val_points, perBatch=False):
    global model, loss_fn
    model = pmodel
    # cached_state_dict = copy.deepcopy(model.state_dict())
    # model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    # Per-sample gradients
    def compute_gradients(model, loss_fn, inputs, targets,  perBatch=False):
        out, l1 = model(inputs, last=True, freeze=True)
        loss = loss_fn(out, targets).sum()
        l0_grads = torch.autograd.grad(loss, out)[0]
        if perBatch:
            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                
        torch.cuda.empty_cache()
        return l0_grads
    
    subset_grads = compute_gradients(model, loss_fn, subset["images"], subset["labels"])
    val_images, val_labels = get_val_images_batch(testset, num_val_points)
    val_grads = compute_gradients(model, loss_fn, val_images, val_labels)
    # print("labels size", subset["labels"].shape)
    # torch.cuda.empty_cache()
    # model.load_state_dict(cached_state_dict)
    # model.train()
    return subset_grads, val_grads

# only last layer grads
def calc_grads(pmodel, ploss_fn, subset, testset, num_val_points, perBatch=False):
    global model, loss_fn
    model = pmodel
    cached_state_dict = copy.deepcopy(model.state_dict())
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    # Per-sample gradients
    def compute_gradients(model, loss_fn, inputs, targets,  perBatch=False):
        out, l1 = model(inputs, last=True, freeze=True)
        loss = loss_fn(out, targets).sum()
        l0_grads = torch.autograd.grad(loss, out)[0]
        if perBatch:
            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                
        torch.cuda.empty_cache()
        return l0_grads
    
    subset_grads = compute_gradients(model, loss_fn, subset["images"], subset["labels"])
    val_images, val_labels = get_val_images_batch(testset, num_val_points)
    val_grads = compute_gradients(model, loss_fn, val_images, val_labels)
    # print("labels size", subset["labels"].shape)
    # torch.cuda.empty_cache()
    model.load_state_dict(cached_state_dict)
    model.train()
    return subset_grads, val_grads

def calc_grads_all_params(pmodel, ploss_fn, subset, testset, num_val_points):
    global model, loss_fn
    model = pmodel
    cached_state_dict = copy.deepcopy(model.state_dict())
    clone_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(clone_dict)
    model.eval()
    loss_fn = ploss_fn
    # Per-sample gradients
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    # print("keys", [k for k in params.keys()], [k for k in buffers.keys()])
    subset_grads = ft_compute_sample_grad(params, buffers, subset["images"], subset["labels"].to(device))
    val_images, val_labels = get_val_images_batch(testset, num_val_points)
    val_grads = ft_compute_sample_grad(params, buffers, val_images, val_labels)
    subset_size = subset["images"].shape[0]
    subset_grads = cat(subset_grads, subset_size) # B,P
    val_grads = cat(val_grads, val_images.shape[0]) # Bv,P
    
    # torch.cuda.empty_cache()
    model.load_state_dict(cached_state_dict)
    model.train()
    return subset_grads, val_grads

    
def best_submod_bandit(pmodel, imp_sample, len_opt, greedyList, testset, ploss_fn, eta_n, moment_sum, num_val_points, opt_grads=None):
    # print("calling best")
    best_index = 0
    imp_sample_grads, val_grads =  calc_grads(pmodel, ploss_fn, imp_sample, testset, num_val_points)
    alpha = 0.7
    indices_list = [[greedyList[i][j][0] for j in range(len(greedyList[i]))] for i in range(len(greedyList))]
    # _grads = opt_grads if opt_grads is not None else 
    print("Len Opt is", len_opt)
    optimal_grads = None
    if(opt_grads is not None): optimal_grads = opt_grads
    elif(len_opt > 0): optimal_grads = imp_sample_grads[0:len_opt]
    metric_list = calc_metric(indices_list , eta_n, imp_sample_grads, val_grads, optimal_grads)
    max_index = torch.argmax(metric_list)
    vmap_max = metric_list[max_index]
    best_metric = -100000 if torch.isnan(vmap_max) else vmap_max.item()
    best_index = max_index.item()
    # print("best metric", best_index, best_metric, metric_list.shape )
    
    return greedyList[best_index], best_index


def greedy_metric(imp_sample_grads,eta_n,opt_grads,val_grads_mean, index):
    # check again
    term1 = eta_n*imp_sample_grads[index]@(val_grads_mean.transpose(0,1)) # B',1
    if(opt_grads is None):
        term2 = 0
    else:
        grad_sum = torch.sum(opt_grads, dim=0, keepdim=True)
        term2 = eta_n*eta_n*imp_sample_grads[index]@((grad_sum.transpose(0,1))) # B',1
    metric =  term1 - term2
    return metric

def calc_metric(indices_list,eta_n, imp_sample_grads, val_grads, opt_grads, reduction="mean"):
    indices_list = torch.tensor(indices_list)
    val_grads_mean = torch.mean(val_grads, dim=0, keepdim=True)
    print("Indices list", indices_list)
    print("Importance sample grads", imp_sample_grads)
    final = torch.tensor([])
    final_indices = 
    for item in range(len(imp_sample_grads)):
        grad_sum = torch.sum(final)
        greedy_metric(
            imp_sample_grads,
            eta_n,
            opt_grads
        )
    def func(submod_indices):
        # check again
        term1 = eta_n*imp_sample_grads[submod_indices]@(val_grads_mean.transpose(0,1)) # B',1
        if(opt_grads is None):
            term2 = 0
        else:
            grad_sum = torch.sum(opt_grads, dim=0, keepdim=True)
            term2 = eta_n*eta_n*imp_sample_grads[submod_indices]@((grad_sum.transpose(0,1))) # B',1
        metric =  term1 - term2
        return metric
    
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    with torch.autocast(device_type=device):
        metric_list = vmap(func, in_dims=(0))(indices_list)
    # print("metric list", metric_list.shape)
    if(reduction == "mean"):
        metric_list = torch.mean(metric_list, dim=1)
    
    print("metric list after", metric_list)
    return metric_list




