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

    predictions = functional_call(model, (params, buffers), (batch,))
    loss = loss_fn(predictions, targets)
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
    if(prev_images is None): sampling_mode = None
    if(sampling_mode == "Uniform"):
        lamb = args["lamb"]
        frac_at = int(batch_size_curr*lamb)
        frac_curr = batch_size_curr - frac_at
        prev_sample = torch.randint(size_at, (frac_at,))  
        curr_sample = torch.randint(batch_size_curr, (frac_curr,))  
        ret_images = torch.cat((prev_images[prev_sample], images[curr_sample]), dim=0)
        ret_labels = torch.cat((prev_labels[prev_sample], labels[curr_sample]), dim=0)
        ret_features = torch.cat((prev_features[prev_sample], features[curr_sample]), dim=0)
        len_opt = prev_sample.shape[0]
        return {"images": ret_images, "labels": ret_labels, "features": ret_features}, len_opt
    
    if(sampling_mode == "Binomial"):
        raise NotImplementedError
    
    if(sampling_mode == None):
        len_opt = current_batch["images"].shape[0]
        return {"images": images, "labels":labels, "features":features}, len_opt
    
    
def get_greedy_list(funcs, submod_budget, multi_thread=False, optimizer="NaiveGreedy"):
    greedyList = {}
    def thread(x):
        # Maximize the function
        f = funcs[x]
        _greedy = submod_maximize(f, submod_budget)
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
            _greedy = submod_maximize(f, submod_budget)
            greedyList[i] = _greedy
    return greedyList

def eps_greedy_composition(model, subset, len_opt, testset, loss_fn, step, funcs, submod_budget, moment_sum, args, optimizer="StochasticGreedy"):
    lamb = args["lamb"]
    pi = args["pi"]
    thresh = step/((step+lamb)**pi)
    eps = torch.rand(1).item()
    greedyList = get_greedy_list(funcs, submod_budget, optimizer)
    if(eps > thresh):
        return best_submod_bandit(model, subset, len_opt, greedyList, testset, loss_fn, args["eta_n"], moment_sum, args["num_val_points"])
    else:
        sample = torch.randint(len(greedyList), ()).item()
        return greedyList[sample], sample
    

    
def submod_maximize_stochastic_greedy(f, budget):
    return f.maximize(budget = budget, optimizer='StochasticGreedy', 
                stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, 
                verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)

def submod_maximize(f, budget, optimizer="StochasticGreedy"):
    return f.maximize(budget = budget, optimizer=optimizer, 
                stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, 
                verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
    
def best_submod_bandit_OLD_(pmodel, subset, len_opt, greedyList, testset, ploss_fn, eta_n, moment_sum, num_val_points):
    best_index = 0
    best_metric = -10000000000
    best_t1 = -1000000
    best_t2 = -1000000
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
    
    # torch.cuda.empty_cache()
    model.load_state_dict(cached_state_dict)
    model.train()
    
    subset_size = subset["images"].shape[0]
    subset_grads = cat(subset_grads, subset_size) # B,P
    val_grads = cat(val_grads, val_images.shape[0]) # Bv,P
    
    alpha = 0.7
    # vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    indices_list = torch.tensor([[greedyList[i][j][0] for j in range(len(greedyList[i]))] for i in range(len(greedyList))])
    def func(submod_indices):
        # submod_indices = indices_list[index]
        t = torch.mean(val_grads, dim=0, keepdim=True)
        term1 = eta_n*subset_grads[submod_indices]@(t.transpose(0,1)) # B',1
        grad_sum = torch.sum(subset_grads[0:len_opt], dim=0, keepdim=True)
        term2 = eta_n*eta_n*subset_grads[submod_indices]@((grad_sum.transpose(0,1))) # B',1
        metric =  torch.mean(term1 - term2, dim=0)
        return metric
    # moment_sum = alpha*(val_grads*val_grads) + (1-alpha)*moment_sum
    with torch.autocast(device_type=device):
        metric_list = vmap(func, in_dims=(0))(indices_list)
        
    # print("metric list", metric_list)
    # print("mex", torch.argmax(metric_list).item())
    
    max_index = torch.argmax(metric_list)
    vmap_max = metric_list[max_index]
    best_metric = -100000 if torch.isnan(vmap_max) else vmap_max.item()
    best_index = max_index.item()
    
    # with torch.autocast(device_type=device):
    #     for i in range(len(greedyList)):
    #         submod_indices = [greedyList[i][j][0] for j in range(len(greedyList[i]))]
    #         t = torch.mean(val_grads, dim=0, keepdim=True)
    #         term1 = eta_n*subset_grads[submod_indices]@(t.transpose(0,1)) # B',1
    #         # print("term1", term1.shape)
    #         grad_sum = torch.sum(subset_grads[0:len_opt], dim=0, keepdim=True)
    #         # eps = 1e-5
    #         # hessian = torch.ones(grad_sum.transpose(0,1).shape).to(device)
    #         # assert torch.equal(grad_sum.transpose(0,1), hessian*grad_sum.transpose(0,1)) == True
    #         # hessian = (moment_sum).transpose(0,1)
    #         term2 = eta_n*eta_n*subset_grads[submod_indices]@((grad_sum.transpose(0,1))) # B',1
    #         metric =  torch.mean(term1 - term2, dim=0)
    #         # metric = torch.mean(metric, dim=0)
    #         print("sudmod", submod_indices)
    #         print("t", t)
    #         print("terms1", term1)
    #         print("term2", term2)
    #         print("*************8")
    #         if(metric.item() > best_metric):
    #             best_metric = metric.item()
    #             best_t1 = term1
    #             best_t2 = term2
    #             # print("Best t1", best_t1)
    #             # print("Best t2", best_t2)
    #             # print("Best metric", best_metric)
    #             # print("**************")
    #             best_index = i
    # print("bests", best_metric, vmap_max, best_index, torch.argmax(metric_list).item())
    # assert best_metric == vmap_max
    
    # print("Best", best_index, best_metric)
    return greedyList[best_index], best_index


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

    
def best_submod_bandit(pmodel, subset, len_opt, greedyList, testset, ploss_fn, eta_n, moment_sum, num_val_points):
    best_index = 0
    subset_grads, val_grads =  calc_grads(pmodel, ploss_fn, subset, testset, num_val_points)

    ## subset = importance sampling |(lambda*B_t + 1- lambda*A_{t-1})| <= |B_t| #80 when batch size 128
    ## subset grads for these 80 samples
    ## for each sample in subset:
        # calc term 1 (w.r.t average mean val)
    ## B^hat_t = {}
    ##


    alpha = 0.7
    indices_list = [[greedyList[i][j][0] for j in range(len(greedyList[i]))] for i in range(len(greedyList))]
    metric_list = calc_metric(indices_list , eta_n, subset_grads, val_grads, len_opt)
    max_index = torch.argmax(metric_list)
    vmap_max = metric_list[max_index]
    best_metric = -100000 if torch.isnan(vmap_max) else vmap_max.item()
    best_index = max_index.item()
    # print("best metric", best_index, best_metric, metric_list.shape )
    
    return greedyList[best_index], best_index


def calc_metric(indices_list,eta_n, subset_grads, val_grads, len_opt, reduction="mean"):
    indices_list = torch.tensor(indices_list)
    val_grads_mean = torch.mean(val_grads, dim=0, keepdim=True)
    def func(submod_indices):
        term1 = eta_n*subset_grads[submod_indices]@(val_grads_mean.transpose(0,1)) # B',1
        if(len_opt == 0):
            term2 = 0
        else:
            grad_sum = torch.sum(subset_grads[0:len_opt], dim=0, keepdim=True)
            term2 = eta_n*eta_n*subset_grads[submod_indices]@((grad_sum.transpose(0,1))) # B',1
        metric =  term1 - term2
        return metric
    
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    with torch.autocast(device_type=device):
        metric_list = vmap(func, in_dims=(0))(indices_list)
    # print("metric list", metric_list.shape)
    if(reduction == "mean"):
        metric_list = torch.mean(metric_list, dim=1)
    # print("metric list after", metric_list.shape)
    return metric_list
