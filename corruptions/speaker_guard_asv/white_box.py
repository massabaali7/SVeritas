

import torch
import numpy as np
from abc import ABCMeta, abstractmethod

from .EOT import EOT
from .utils import resolve_loss, resolve_prediction, SEC4SR_MarginLoss

class Attack(metaclass=ABCMeta):

    @abstractmethod
    def attack(self, x, y, verbose=0, EOT_size=1, EOT_batch_size=1):
        pass

    def compare(self, y, y_pred, targeted):
        if targeted:
            return (y_pred == y).tolist()
        else:
            return (y_pred != y).tolist()


class FGSM(Attack):
    def __init__(self, model, task='CSI', epsilon=0.002, loss='Entropy', targeted=False, 
                batch_size=1, EOT_size=1, EOT_batch_size=1, 
                verbose=0, threshold=0.):

        self.model = model # remember to call model.eval()
        self.task = task
        self.epsilon = epsilon
        self.loss_name = loss
        self.targeted = targeted
        self.batch_size = batch_size
        EOT_size = max(1, EOT_size)
        EOT_batch_size = max(1, EOT_batch_size)
        assert EOT_size % EOT_batch_size == 0, 'EOT size should be divisible by EOT batch size'
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        self.threshold = threshold
        if self.task in ['SV', 'OSI']:
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))
        self.loss, self.grad_sign = resolve_loss(loss_name=self.loss_name, targeted=self.targeted,
                                    task=self.task, threshold=self.threshold, clip_max=False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, True)

        self.max_iter = 1 # FGSM is single step attack, keep in consistency with PGD
        self.step_size = epsilon # keep in consistency with PGD
    
    def attack_batch(self, x_batch, x_tgt_batch, y_batch, lower, upper, batch_id):
        
        x_batch = x_batch.clone() # avoid influcing
        # x_batch.retain_grad()
        x_batch.requires_grad = True
        success = None
        for iter in range(self.max_iter + 1):
            EOT_num_batches = int(self.EOT_size // self.EOT_batch_size) if iter < self.max_iter else 1
            real_EOT_batch_size = self.EOT_batch_size if iter < self.max_iter else 1
            use_grad = True if iter < self.max_iter else False
            # scores, loss, grad = EOT_wrapper(x_batch, y_batch, EOT_num_batches, real_EOT_batch_size, use_grad)
            scores, loss, grad, decisions = self.EOT_wrapper(x_batch, x_tgt_batch, y_batch, EOT_num_batches, real_EOT_batch_size, use_grad)
            scores.data = scores / EOT_num_batches
            loss.data = loss / EOT_num_batches
            if iter < self.max_iter:
                grad.data = grad / EOT_num_batches
            # predict = torch.argmax(scores.data, dim=1).detach().cpu().numpy()
            predict = resolve_prediction(decisions)
            target = y_batch.detach().cpu().numpy()
            success = self.compare(target, predict, self.targeted)
            if self.verbose:
                print("batch:{} iter:{} loss: {} predict: {}, target: {}".format(batch_id, iter, loss.detach().cpu().numpy().tolist(), predict, target))
            
            if iter < self.max_iter:
                x_batch.grad = grad
                # x_batch.data += self.epsilon * torch.sign(x_batch.grad) * self.grad_sign
                ## x_batch.data += self.epsilon * torch.sign(grad) * self.grad_sign
                x_batch.data += self.step_size * torch.sign(x_batch.grad) * self.grad_sign
                x_batch.grad.zero_()
                # x_batch.data = torch.clamp(x_batch.data, min=lower, max=upper)
                x_batch.data = torch.min(torch.max(x_batch.data, lower), upper)
            
        return x_batch, success

    def attack(self, x, x_tgt, y):

        lower = -1
        upper = 1 
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
        lower = torch.tensor(lower, device=x.device, dtype=x.dtype).expand_as(x)
        upper = torch.tensor(upper, device=x.device, dtype=x.dtype).expand_as(x)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))
        for batch_id in range(n_batches):
            x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
            x_tgt_batch = x_tgt[batch_id*batch_size:(batch_id+1)*batch_size]
            lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
            upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
            adver_x_batch, success_batch = self.attack_batch(x_batch, x_tgt_batch, y_batch, lower_batch, upper_batch, batch_id)
            if batch_id == 0:
                adver_x = adver_x_batch
                success = success_batch
            else:
                adver_x = torch.cat((adver_x, adver_x_batch), 0)
                success += success_batch

        return adver_x, success

class PGD(FGSM):
    
    def __init__(self, model, task='CSI', epsilon=0.002, step_size=0.0004, max_iter=10, num_random_init=0, 
                loss='Entropy', targeted=False, 
                batch_size=1, EOT_size=1, EOT_batch_size=1, 
                verbose=0, threshold=0.):

        self.model = model # remember to call model.eval()
        self.task = task
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_iter = max_iter
        self.num_random_init = num_random_init
        self.loss_name = loss
        self.targeted = targeted
        self.batch_size = batch_size
        EOT_size = max(1, EOT_size)
        EOT_batch_size = max(1, EOT_batch_size)
        assert EOT_size % EOT_batch_size == 0, 'EOT size should be divisible by EOT batch size'
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        self.threshold = threshold
        if self.task in ['SV', 'OSI']:
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))
        self.loss, self.grad_sign = resolve_loss(loss_name=self.loss_name, targeted=self.targeted,
                                    task=self.task, threshold=self.threshold, clip_max=False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, True)

    def attack(self, x, x_tgt, y):

        lower = -1
        upper = 1
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
        n_audios, n_channels, max_len = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
        upper = torch.clamp(x+self.epsilon, max=upper)
        lower = torch.clamp(x-self.epsilon, min=lower)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))

        x_ori = x.clone()
        best_success_rate = -1
        best_success = None
        best_adver_x = None
        for init in range(max(1, self.num_random_init)):
            if self.num_random_init > 0:
                x = x_ori + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, \
                                (n_audios, n_channels, max_len)), device=x.device, dtype=x.dtype) 
            for batch_id in range(n_batches):
                x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
                x_tgt_batch = x_tgt[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
                y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
                lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
                upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
                adver_x_batch, success_batch = self.attack_batch(x_batch, x_tgt_batch, y_batch, lower_batch, upper_batch, '{}-{}'.format(init, batch_id))
                if batch_id == 0:
                    adver_x = adver_x_batch
                    success = success_batch
                else:
                    adver_x = torch.cat((adver_x, adver_x_batch), 0)
                    success += success_batch
            if sum(success) / len(success) > best_success_rate:
                best_success_rate = sum(success) / len(success)
                best_success = success
                best_adver_x = adver_x

        return best_adver_x, best_success


class CW2(FGSM):

    def __init__(self, model, task='CSI',
                targeted=False,
                confidence=0.,
                initial_const=1e-3, 
                binary_search_steps=9,
                max_iter=10000,
                stop_early=True,
                stop_early_iter=1000,
                lr=1e-2,
                batch_size=1,
                verbose=0,
				threshold=0.):

        self.model = model
        self.task = task
        self.targeted = targeted
        self.confidence = confidence
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.stop_early = stop_early
        self.stop_early_iter = stop_early_iter
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose

        self.threshold = threshold
        if self.task in ['SV', 'OSI']:
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))

        self.loss = SEC4SR_MarginLoss(targeted=self.targeted, confidence=self.confidence, task=self.task, threshold=self.threshold, clip_max=True)
    
    def attack_batch(self, x_batch, x_tgt_batch, y_batch, lower, upper, batch_id):

        n_audios, _, _ = x_batch.shape

        const = torch.tensor([self.initial_const] * n_audios, dtype=torch.float, device=x_batch.device)
        lower_bound = torch.tensor([0] * n_audios, dtype=torch.float, device=x_batch.device)
        upper_bound = torch.tensor([1e10] * n_audios, dtype=torch.float, device=x_batch.device)

        global_best_l2 = [np.infty] * n_audios
        global_best_adver_x = x_batch.clone()
        global_best_score = [-2] * n_audios # do not use [-1] * n_audios since -1 is within the decision space of SV and OSI tasks 

        for _ in range(self.binary_search_steps):

            self.modifier = torch.zeros_like(x_batch, dtype=torch.float, requires_grad=True, device=x_batch.device)
            self.optimizer = torch.optim.Adam([self.modifier], lr=self.lr)

            best_l2 = [np.infty] * n_audios
            # best_score = [-1] * n_audios
            best_score = [-2] * n_audios

            continue_flag = True
            prev_loss = np.infty
            # we need to perform the gradient descent max_iter times; 
            # the additional one iteration is used to to evaluate the final updated examples
            for n_iter in range(self.max_iter+1): 
                if not continue_flag:
                    break
                # deal with box constraint, [-1, 1], different from image
                input_x = torch.tanh(self.modifier + torch.atanh(x_batch * 0.999999))
                decisions, scores = self.model(input_x, x_tgt_batch) # (n_audios, n_spks)
                loss1 = self.loss(scores, y_batch)
                loss2 = torch.sum(torch.square(input_x - x_batch), dim=(1,2))
                loss = const * loss1 + loss2

                if n_iter < self.max_iter: # we only perform gradient descent max_iter times
                    loss.backward(torch.ones_like(loss))
                    # update modifier
                    self.optimizer.step()
                    self.modifier.grad.zero_()

                # predict = torch.argmax(scores.data, dim=1).detach().cpu().numpy() # not suitable for SV and OSI tasks which will reject
                predict = decisions.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy().tolist()
                loss1 = loss1.detach().cpu().numpy().tolist()
                loss2 = loss2.detach().cpu().numpy().tolist()
                if self.verbose:
                    print("batch: {}, c: {}, iter: {}, loss: {}, loss1: {}, loss2: {}, y_pred: {}, y: {}".format(
                        batch_id, const.detach().cpu().numpy(), n_iter, 
                        loss, loss1, loss2, predict, y_batch.detach().cpu().numpy()))
                
                if self.stop_early and n_iter % self.stop_early_iter == 0:
                    if np.mean(loss) > 0.9999 * prev_loss:
                        continue_flag = False
                    prev_loss = np.mean(loss)

                for ii, (l2, y_pred, adver_x, l1) in enumerate(zip(loss2, predict, input_x, loss1)):
                    # IF-BRANCH-1
                    if l1 <= 0 and l2 < best_l2[ii]: # l1 <= 0 indicates the attack succeed with at least kappa confidence
                        best_l2[ii] = l2
                        best_score[ii] = y_pred
                    # IF-BRANCH-2
                    if l1 <= 0 and l2 < global_best_l2[ii]: # l1 <= 0 indicates the attack succeed with at least kappa confidence
                        global_best_l2[ii] = l2
                        global_best_score[ii] = y_pred
                        global_best_adver_x[ii] = adver_x

            for jj, y_pred in enumerate(best_score):
                if y_pred != -2: # y_pred != -2 infers that IF-BRANCH-1 is entered at least one time, thus the attack succeeds
                    upper_bound[jj] = min(upper_bound[jj], const[jj])
                    if upper_bound[jj] < 1e9:
                        const[jj] = (lower_bound[jj] + upper_bound[jj]) / 2
                else:
                    lower_bound[jj] = max(lower_bound[jj], const[jj])
                    if upper_bound[jj] < 1e9:
                        const[jj] = (lower_bound[jj] + upper_bound[jj]) / 2
                    else:
                        const[jj] *= 10
            
            #print(const.detach().cpu().numpy(), best_l2, global_best_l2)
        
        success = [False] * n_audios
        for kk, y_pred in enumerate(global_best_score):
            if y_pred != -2: # y_pred != -2 infers that IF-BRANCH-2 is entered at least one time, thus the attack succeeds
                success[kk] = True 

        return global_best_adver_x, success

    def attack(self, x, x_tgt, y):
        return super().attack(x, x_tgt, y)

class CWinf(PGD):
    
    def __init__(self, model, task='CSI', epsilon=0.002, step_size=0.0004, max_iter=10, num_random_init=0, 
                loss='Margin', targeted=False, threshold=0.,
                batch_size=1, EOT_size=1, EOT_batch_size=1, 
                verbose=0):

        loss = 'Margin' # hard coding: using Margin Loss
        super().__init__(model, task=task, epsilon=epsilon, step_size=step_size, max_iter=max_iter, num_random_init=num_random_init, 
                        loss=loss, targeted=targeted,
                        batch_size=batch_size, EOT_size=EOT_size, EOT_batch_size=EOT_batch_size, 
                        verbose=verbose, threshold=threshold)


