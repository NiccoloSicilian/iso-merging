import torch
import math
def dm_layer_specific(task_vectors, config):
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]
            new_vector[key] = sum(tvs) / len(tvs)
            print("all ",key,task_vectors[0].vector[key].shape)
            if len(tensor.shape) == 4:
                print("CONV")
                new_vector[key] *= len(tvs)
                matrix = new_vector[key]  # [dout, din, k, k]
                dout, din, k, _ = matrix.shape
                
                # Compute scaling factor
                scaling_factor = 1.0 / (k**2 * torch.sqrt(torch.tensor(dout / din, dtype=matrix.dtype)))
                
                # Create output tensor
                transformed = torch.zeros_like(matrix)
                
                # For each spatial position (i, j) in the k×k grid
                for i in range(k):
                    for j in range(k):
                        # Extract the [dout, din] slice at position (i, j)
                        slice_matrix = matrix[:, :, i, j]  # [dout, din]
                        
                        # SVD decomposition
                        U, S, Vt = torch.linalg.svd(slice_matrix, full_matrices=False)
                        
                        # Reconstruct with U @ V^T (ignoring singular values)
                        reconstructed = U @ Vt
                        
                        # Apply scaling factor
                        transformed[:, :, i, j] = scaling_factor * reconstructed
                new_vector[key] = transformed
            elif 'embedding' in key.lower() and len(tensor.shape) == 2:
                print("EMBEDDING")
                new_vector[key] *= len(tvs)
                rms_norm = torch.sqrt(torch.mean(new_vector[key] ** 2, dim=0, keepdim=True))
                
                # Normalize each column by its RMS norm
                new_vector[key] = new_vector[key] / rms_norm
            elif len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
                new_vector[key] *= len(tvs)
                dout, din = new_vector[key].shape
                dinDoutRatio = torch.sqrt(torch.tensor(dout / din, dtype=torch.float32))
                U, S, V = torch.linalg.svd(new_vector[key], full_matrices=False)
                S_dm = torch.full_like(S,dinDoutRatio)
                print(key,new_vector[key].shape,S_dm.shape, "USING DM")
                new_vector[key] = torch.linalg.multi_dot(
                  (
                      U,
                      torch.diag(S_dm),
                      V,
                  )
                )
            else:
                print("Skipped")
        return new_vector
def dm_per_task(task_vectors, config):
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
      new_vector = {}
      S_mean = None
      S_dm = None
      for key in task_vectors[0].vector:
          tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]
          new_vector[key] = sum(tvs) / len(tvs)
      
          
          print("all ",key,task_vectors[0].vector[key].shape)
          
              
          if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
              
              new_vector[key] = torch.full_like(new_vector[key],0.0)
              dout, din = new_vector[key].shape
              dinDoutRatio = torch.sqrt(torch.tensor(dout / din, dtype=torch.float32))
              print("USING NESTED DM")
              for v in tvs:
                  U, S, V = torch.linalg.svd(v, full_matrices=False)
                  S_dm = torch.full_like(S,dinDoutRatio)
                  new_vector[key] += torch.linalg.multi_dot((U,torch.diag(S_dm), V,))
              new_vector[key]=new_vector[key]/len(tvs)
              dout, din = new_vector[key].shape
              U, S, V = torch.linalg.svd(new_vector[key], full_matrices=False)
              S_mean = torch.ones_like(S) * S.mean()
              S_dm = torch.full_like(S,dinDoutRatio)
              I = torch.full_like(S, 1.0)
              print(key,new_vector[key].shape,S_mean.shape,S_dm.shape, "USING Mean")
              new_vector[key] = torch.linalg.multi_dot(
                  (
                      U,
                      torch.diag(S_mean),
                      V,
                  )
              )
          else:
              print("Skipped")
      print(S_mean, S_dm)
    return new_vector
def dm_whole_vec(task_vectors, config):
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
      new_vector = {}
      for key in task_vectors[0].vector:
          tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]
          new_vector[key] = sum(tvs) / len(tvs)
          print("all ",key,task_vectors[0].vector[key].shape)
          if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
              new_vector[key] *= len(tvs)
              dout, din = new_vector[key].shape
              dinDoutRatio = torch.sqrt(torch.tensor(dout / din, dtype=torch.float32))
              U, S, V = torch.linalg.svd(new_vector[key], full_matrices=False)
              S_dm = torch.full_like(S,dinDoutRatio)
              print(key,new_vector[key].shape,S_dm.shape, "USING DM")
              new_vector[key] = torch.linalg.multi_dot(
                  (
                      U,
                      torch.diag(S_dm),
                      V,
                  )
              )
          else:
              print("Skipped")
    return new_vector

def iso_c(task_vectors, config):
    
    return dm_layer_specific(task_vectors, config)
'''
def new_method(task_vectors, config):
    pretrained_model = ImageEncoder.load(self.model_name, checkpoint)
    pretrained_model = pretrained_model.to(args.device)
    dict = dict of model
    
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]
            if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
                U0,S0,V0 = SVD(dict[key])
                for tv in tvs:
                    
                dout, din = new_vector[key].shape
                dinDoutRatio = torch.sqrt(torch.tensor(dout / din, dtype=torch.float32))
                new_vector[key] *= len(tvs)
                U, S, V = torch.linalg.svd(new_vector[key], full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()
                S_dm = torch.full_like(S,dinDoutRatio)
                I = torch.full_like(S, 1.0)
                print(new_vector[key].shape,S_mean.shape,S_dm.shape, "USING DM")
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        U,
                        torch.diag(S_dm),
                        V,
                    )
                )
'''

@torch.no_grad()
def iso_cts(task_vectors, config):
    device = config.device
    new_vector = {}

    print("Computing SVD...")
    for key in task_vectors[0].vector:
        shape_ = task_vectors[0].vector[key].shape

        is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
        if not is_2d_matrix:
            print(f"Combining by avg {key}...")
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
                vec = task_vector.vector[key].to(device)
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)
            continue
        
        print(f"Computing common space using sum for {key}...")
        combined_w = sum([task_vector.vector[key].to(device) for task_vector in task_vectors])

        ### Calculate the common space size (making sure that task specific space is equally divisible) ###
        common_space_index_s = int(min(shape_) * config.method.common_space_fraction)
        _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / len(config.DATASETS)) * len(config.DATASETS)
        common_space_index_s = min(shape_) - _task_specific_total_space_index_s

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]
        ###################################################################
        
        ### Calculate task specific space ###
        n_dims_per_task = int((min(shape_) - common_space_index_s) / len(config.DATASETS))
        for i, task_vector in enumerate(task_vectors):
            w = task_vector.vector[key].to(device)

            # calculate the projection onto task specific space to remove the common space
            w_ts = w - common_space_u @ common_space_u.T @ w
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)            
            
            if i == 0:
                combined_space_u = torch.zeros_like(u_ts, device=device)
                combined_space_s = torch.zeros_like(s_ts, device=device)
                combined_space_v = torch.zeros_like(v_ts, device=device)
                
            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]
        ###################################################################
        
        combined_space_u[:, len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = common_space_u
        combined_space_s[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = common_space_s
        combined_space_v[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s, :] = common_space_v
        
        ### Orthogonalize combined_space_u and combined_space_v ###
        u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
        u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
        combined_space_u = u_combined_space_u @ v_combined_space_u
        combined_space_v = u_combined_space_v @ v_combined_space_v
        ###################################################################
        
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
                
        new_vector[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        )
    
    return new_vector
