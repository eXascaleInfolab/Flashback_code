
import torch
import numpy as np

class Evaluation:
    
    '''
    Handles evaluation on a given POI dataset and loader.
    
    The two metrics are MAP and recall@n. Our model predicts sequencse of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.
    
    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.
    
    Using the --report_user argument one can access the statistics per user.
    '''
    
    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting

    def evaluate(self):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)
        
        with torch.no_grad():        
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.
            
            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)        
            reset_count = torch.zeros(self.user_count)
            
            for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1
                
                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)            
                y = y.squeeze()
                y_t = y_t.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)
                
                active_users = active_users.to(self.setting.device)            
            
                # evaluate:
                out, h = self.trainer.evaluate(x, t, s, y_t, y_s, h, active_users)
                
                for j in range(self.setting.batch_size):  
                    # o contains a per user list of votes for all locations for each sequence entry
                    o = out[j]                        
                                        
                    # partition elements
                    o_n = o.cpu().detach().numpy()
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:] # top 10 elements
                                       
                    y_j = y[:, j]
                    
                    for k in range(len(y_j)):                    
                        if (reset_count[active_users[j]] > 1):
                            continue # skip already evaluated users.
                                                                                                            
                        # resort indices for k:
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)] # sort top 10 elements descending
                                                    
                        r = torch.tensor(r)
                        t = y_j[k]
                        
                        # compute MAP:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1+len(upper))
                        
                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t in r[:1]
                        u_recall5[active_users[j]] += t in r[:5]
                        u_recall10[active_users[j]] += t in r[:10]
                        u_average_precision[active_users[j]] += precision
            
            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                average_precision += u_average_precision[j]
    
                if (self.setting.report_user > 0 and (j+1) % self.setting.report_user == 0):
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1', formatter.format(u_recall1[j]/u_iter_cnt[j]), 'MAP', formatter.format(u_average_precision[j]/u_iter_cnt[j]), sep='\t')
            
            print('recall@1:', formatter.format(recall1/iter_cnt))
            print('recall@5:', formatter.format(recall5/iter_cnt))
            print('recall@10:', formatter.format(recall10/iter_cnt))
            print('MAP', formatter.format(average_precision/iter_cnt))
            print('predictions:', iter_cnt)