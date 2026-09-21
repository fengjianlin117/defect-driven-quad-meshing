import os
import torch
import torch.nn as nn

import utils.utils as utils


def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()
    # eikonal_term = (-torch.log(all_grads.norm(2, dim=2))).mean()
    return eikonal_term

class MorseLoss_quad_mesh(nn.Module):
    def __init__(self, weights=None, loss_type='siren_wo_n_w_morse', div_decay='none',
                 div_type='l1', vertex_neighbors_list=None,
                 vertex_neighbors=None, axis_angle_R_mat_list=None, device=None,
                 use_batched_hessian=True):
        super().__init__()
        if weights is None:
            weights = [7e3, 6e2, 10, 5e1, 30, 3]
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_morse = True if 'morse' in self.loss_type else False
        self.vertex_neighbors_list = vertex_neighbors_list
        self.vertex_neighbors = vertex_neighbors
        self.axis_angle_R_mat_list = axis_angle_R_mat_list
        self.device = device
        self.use_batched_hessian = use_batched_hessian

        # These tensors are topology constants.  The original implementation
        # rebuilt them on the CPU and copied them to the GPU on every training
        # step.  Register them once so the loss math and group-wise reduction
        # stay unchanged while removing that per-step Python/PCIe overhead.
        self._neighbor_cache_names = []
        if self.vertex_neighbors_list is not None and self.vertex_neighbors is not None:
            cache_device = torch.device(self.device) if self.device is not None else torch.device('cpu')
            for group_id, group_faces in enumerate(self.vertex_neighbors_list):
                idx_name = '_neighbor_group_idx_{}'.format(group_id)
                neighbors_name = '_neighbor_group_neighbors_{}'.format(group_id)
                rotation_name = None

                idx = torch.as_tensor(group_faces, dtype=torch.long, device=cache_device)
                neighbors = torch.as_tensor(
                    [self.vertex_neighbors[face_idx] for face_idx in group_faces],
                    dtype=torch.long,
                    device=cache_device,
                )
                self.register_buffer(idx_name, idx, persistent=False)
                self.register_buffer(neighbors_name, neighbors, persistent=False)

                if self.axis_angle_R_mat_list is not None:
                    rotation_name = '_neighbor_group_rotation_{}'.format(group_id)
                    rotations = torch.as_tensor(
                        self.axis_angle_R_mat_list[group_id],
                        dtype=torch.float32,
                        device=cache_device,
                    )
                    self.register_buffer(rotation_name, rotations, persistent=False)

                self._neighbor_cache_names.append((idx_name, neighbors_name, rotation_name))



    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None, near_points=None, batch_idx=0,
                logdir=None, filename=None, save_best=False, mnfld_pts_theta_output_pred=None, local_coord_u=None,
                local_coord_v=None):

        dims = mnfld_points.shape[-1]

        #########################################
        # Compute required terms
        #########################################

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]

        morse_loss = torch.tensor([0.0], device=self.device)
        normal_term = torch.tensor([0.0], device=self.device)
        eikonal_term = torch.tensor([0.0], device=self.device)
        mnfld_hessian_term = torch.tensor([0.0], device=self.device)

        # Phase 2: merged first-order gradients (3 -> 1 call)
        all_pred_direct = output_pred.get('all_pred', None)
        decoder_all_pts = output_pred.get('all_pts_decoder', None)
        merged_grad_available = all_pred_direct is not None and decoder_all_pts is not None and manifold_pred is not None
        near_grad = None
        if merged_grad_available:
            M_mnfld = mnfld_points.shape[1]
            N_nonmnfld = nonmnfld_points.shape[1]
            total_used = M_mnfld + N_nonmnfld
            # Gradient on FULL tensors (all points used → no 'unused' error)
            all_grad = utils.gradient(decoder_all_pts, all_pred_direct)
            # Split: decoder order is [mnfld, nonmnfld, near]
            mnfld_grad = all_grad[:, :M_mnfld, :]
            nonmnfld_grad = all_grad[:, M_mnfld:total_used, :]
            if near_points is not None:
                near_grad = all_grad[:, total_used:total_used + near_points.shape[1], :]
        else:
            # Fallback to old method
            if manifold_pred is not None:
                mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
            else:
                mnfld_grad = None
            nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)
            M_mnfld = mnfld_points.shape[1] if manifold_pred is not None else 0
            N_nonmnfld = nonmnfld_points.shape[1]
            total_used = M_mnfld + N_nonmnfld
        
        morse_nonmnfld_grad = None
        if self.use_morse and near_points is not None:
            if near_grad is not None:
                morse_nonmnfld_grad = near_grad
            else:
                morse_nonmnfld_grad = utils.gradient(near_points, output_pred['near_points_pred'])
        elif self.use_morse and near_points is None:
            morse_nonmnfld_grad = nonmnfld_grad

        # Hessian (use decoder_all_pts to stay in the merged computation graph)
        if self.use_morse:
            mnfld_pts_for_hess = output_pred.get('all_pts_decoder', None)
            if mnfld_pts_for_hess is not None and self.use_batched_hessian:
                # Compute the three Hessian rows in one vectorized autograd
                # traversal.  Each leading basis vector is an independent VJP;
                # permuting the result matches stack((dx, dy, dz), dim=-1).
                hessian_basis = torch.eye(
                    dims, dtype=mnfld_grad.dtype, device=mnfld_grad.device
                ).view(dims, 1, 1, dims).expand(dims, *mnfld_grad.shape)
                batched_hessian = torch.autograd.grad(
                    outputs=mnfld_grad,
                    inputs=mnfld_pts_for_hess,
                    grad_outputs=hessian_basis,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    is_grads_batched=True,
                )[0]
                mnfld_hessian_term = batched_hessian[:, :, :M_mnfld, :].permute(1, 2, 3, 0)
            else:
                if mnfld_pts_for_hess is None:
                    mnfld_pts_for_hess = mnfld_points
                    mnfld_dx = utils.gradient(mnfld_pts_for_hess, mnfld_grad[:, :, 0])
                    mnfld_dy = utils.gradient(mnfld_pts_for_hess, mnfld_grad[:, :, 1])
                else:
                    def hessian_row(component):
                        row = torch.autograd.grad(
                            outputs=mnfld_grad[:, :, component],
                            inputs=mnfld_pts_for_hess,
                            grad_outputs=torch.ones_like(mnfld_grad[:, :, component]),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        return row[:, :M_mnfld, :]

                    mnfld_dx = hessian_row(0)
                    mnfld_dy = hessian_row(1)
                if dims == 3:
                    if output_pred.get('all_pts_decoder', None) is None:
                        mnfld_dz = utils.gradient(mnfld_pts_for_hess, mnfld_grad[:, :, 2])
                    else:
                        mnfld_dz = hessian_row(2)
                    mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
                else:
                    mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy), dim=-1)

            morse_mnfld = torch.tensor([0.0], device=self.device)
            if self.div_type == 'l1':
                mnfld_n_gt = mnfld_n_gt.permute(1, 0, 2)
                mnfld_hessian_term = mnfld_hessian_term.squeeze(0)

                morse_mnfld = torch.bmm(mnfld_n_gt, mnfld_hessian_term)
                morse_mnfld = morse_mnfld.abs().mean()

            morse_loss = 0.5 * morse_mnfld

        sdf_term = torch.abs(manifold_pred).mean()

        # eikonal term
        eikonal_term = eikonal_loss(morse_nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

        # theta_term
        local_coord_u = local_coord_u.squeeze(0)
        local_coord_v = local_coord_v.squeeze(0)

        mnfld_pts_theta_output_pred = mnfld_pts_theta_output_pred.squeeze(0)

        vector_alpha = local_coord_u * torch.cos(mnfld_pts_theta_output_pred) + local_coord_v * torch.sin(
            mnfld_pts_theta_output_pred)
        vector_alpha = vector_alpha / (vector_alpha.norm(dim=-1, keepdim=True) + 1e-12)
        vector_alpha = vector_alpha.unsqueeze(1)

        vector_beta = -local_coord_u * torch.sin(mnfld_pts_theta_output_pred) + local_coord_v * torch.cos(
            mnfld_pts_theta_output_pred)
        vector_beta = vector_beta / (vector_beta.norm(dim=-1, keepdim=True) + 1e-12)
        vector_beta = vector_beta.unsqueeze(1)

        theta_hessian_term = torch.tensor([0.0], device=self.device)
        theta_neighbors_term = torch.tensor([0.0], device=self.device)

        for idx_name, neighbors_name, rotation_name in self._neighbor_cache_names:
            idx = getattr(self, idx_name)

            # theta_hessian_term
            hessian_i = mnfld_hessian_term[idx].unsqueeze(1)  # n x 1 x 3 x 3
            vector_alpha_i = vector_alpha[idx].unsqueeze(1)  # n x 1 x 1 x 3
            vector_beta_i = vector_beta[idx].unsqueeze(1)  # n x 1 x 1 x 3

            vertex_h_term_alpha = torch.matmul(vector_alpha_i, hessian_i)
            vertex_h_term_alpha = torch.linalg.cross(vertex_h_term_alpha, vector_alpha_i)
            vertex_h_term_alpha = vertex_h_term_alpha.abs().mean()

            vertex_h_term_beta = torch.matmul(vector_beta_i, hessian_i)
            vertex_h_term_beta = torch.linalg.cross(vertex_h_term_beta, vector_beta_i)
            vertex_h_term_beta = vertex_h_term_beta.abs().mean()

            vertex_h_term = 0.5 * (vertex_h_term_alpha + vertex_h_term_beta)
            theta_hessian_term += vertex_h_term

            # theta_neighbors_term
            vertex_neighbors_i = getattr(self, neighbors_name)
            vector_alpha_j = vector_alpha[vertex_neighbors_i]  # n x neighbors_size x 1 x 3
            vector_beta_j = vector_beta[vertex_neighbors_i]  # n x neighbors_size x 1 x 3

            if rotation_name is not None:
                axis_angle_R_mat_i = getattr(self, rotation_name)
                vector_alpha_j = utils.transform_vectors_only_rotation(vector_alpha_j, axis_angle_R_mat_i)
                vector_beta_j = utils.transform_vectors_only_rotation(vector_beta_j, axis_angle_R_mat_i)

            neighbors_term_alpha_alpha = torch.matmul(vector_alpha_i, vector_alpha_j.permute(0, 1, 3, 2)).abs()
            neighbors_term_alpha_beta = torch.matmul(vector_alpha_i, vector_beta_j.permute(0, 1, 3, 2)).abs()
            neighbors_term_beta_alpha = torch.matmul(vector_beta_i, vector_alpha_j.permute(0, 1, 3, 2)).abs()
            neighbors_term_beta_beta = torch.matmul(vector_beta_i, vector_beta_j.permute(0, 1, 3, 2)).abs()
            neighbors_term = (neighbors_term_alpha_alpha + neighbors_term_alpha_beta + neighbors_term_beta_alpha + \
                              neighbors_term_beta_beta - 2).mean()  # the sum value is greater than 2

            theta_neighbors_term += neighbors_term

        num_vert_neigh = len(self._neighbor_cache_names)
        theta_hessian_term = theta_hessian_term / num_vert_neigh
        theta_neighbors_term = theta_neighbors_term / num_vert_neigh

        # get the grad, curvature of the points
        if save_best:
            output_dir = os.path.join(logdir, 'save_crossField')
            utils.save_only_crossField(vector_alpha, vector_beta, batch_idx=batch_idx, output_dir=output_dir,
                                       shapename=filename)

        # losses used in the paper
        if self.loss_type == 'siren_wo_n_w_morse_w_theta':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + \
                   self.weights[5] * morse_loss + self.weights[2] * theta_hessian_term + self.weights[
                       4] * theta_neighbors_term
        else:
            print(self.loss_type)
            raise Warning("unrecognized loss type")


        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term,
                'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'morse_term': morse_loss,
                'theta_hessian_term': theta_hessian_term, 'theta_neighbors_term': theta_neighbors_term}

    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.7, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we
        elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we
        elif self.div_decay == 'step':  # change weight at s
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")
