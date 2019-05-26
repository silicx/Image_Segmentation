import torch
import numpy as np

# SR : Segmentation Result
# GT : Ground Truth

class Metrics:
    def __init__(self, SR=None, GT=None, threshold=0.5):
        self.acc = []		# Accuracy
        self.SE  = []		# Sensitivity (Recall)
        self.SP  = []		# Specificity
        self.PC  = [] 		# Precision
        self.F1  = []		# F1 Score
        self.JS  = []		# Jaccard Similarity
        self.DC  = []		# Dice Coefficient
        self.class_acc = [] # classification acc

        if SR is not None and GT is not None:
            SR = torch.split(SR, 1, dim=0)
            GT = torch.split(GT, 1, dim=0)
            for i in range(len(SR)):
                sr, gt = SR[i], GT[i]
                sr, gt = sr.reshape(sr.shape[1:]), gt.reshape(gt.shape[1:])
                sr, gt = torch.argmax(sr, dim=0), torch.argmax(gt, dim=0)
                self.class_acc.append(get_classfication_acc(sr, gt))

                sr, gt = sr>0, gt>0

                self.acc.append(get_accuracy(   sr, gt, threshold) )
                self.SE.append( get_sensitivity(sr, gt, threshold) )
                self.SP.append( get_specificity(sr, gt, threshold) )
                self.PC.append( get_precision(  sr, gt, threshold) )
                self.F1.append( get_F1(         sr, gt, threshold) )
                self.JS.append( get_JS(         sr, gt, threshold) )
                self.DC.append( get_DC(         sr, gt, threshold) )


    def add(self, met):
        self.acc += met.acc
        self.SE  += met.SE 
        self.SP  += met.SP 
        self.PC  += met.PC 
        self.F1  += met.F1 
        self.JS  += met.JS 
        self.DC  += met.DC 

    def __str__(self):
        return 'Acc={:.4f}, SE={:.4f}, PC={:.4f}, JS={:.4f}, DC={:.4f}, C_acc={:.4f}'.format(
                np.nanmean(np.array(self.acc)),
                np.nanmean(np.array(self.SE)), 
                np.nanmean(np.array(self.PC)), 
                np.nanmean(np.array(self.JS)), 
                np.nanmean(np.array(self.DC)),
                np.nanmean(np.array(self.class_acc)))


def get_classfication_acc(SR, GT):
    corr = torch.sum((SR==GT)&(GT>0))
    size = torch.sum(GT!=0)
    if size==0:
        return np.nan
    else:
        return float(corr)/float(size)


def get_accuracy(SR,GT,threshold=0.5):
    #SR = (SR > threshold)
    corr = torch.sum(SR==GT)
    tensor_size = SR.nelement()
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    #SR = SR > threshold

    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    tp_fn = torch.sum(TP+FN)
    if tp_fn == 0:
        return np.nan
    else:
        return float(torch.sum(TP))/float(tp_fn) 
        

def get_specificity(SR,GT,threshold=0.5):
    #SR = SR > threshold

    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    tn_fp = torch.sum(TN+FP)
    if tn_fp == 0:
        return np.nan
    else:
        return float(torch.sum(TN))/float(tn_fp)
        

def get_precision(SR,GT,threshold=0.5):
    #SR = SR > threshold

    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    tp_fp = torch.sum(TP+FP)
    if tp_fp == 0:
        return np.nan
    else:
        return float(torch.sum(TP))/float(tp_fp)


def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    if SE+PC == 0:
        return np.nan
    else:
        return 2*SE*PC/(SE+PC)



def get_JS(SR,GT,threshold=0.5):
    #SR = SR > threshold
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    if Union == 0:
        return np.nan
    else:
        return float(Inter)/float(Union)


def get_DC(SR,GT,threshold=0.5):
    #SR = SR > threshold

    Inter = torch.sum((SR+GT)==2)
    Sum   = torch.sum(SR)+torch.sum(GT)

    if Sum == 0:
        return np.nan
    else:
        return float(2*Inter)/(float(Sum))




def evaluate_3D_image(pred, gt):
    res = dict()
    
    classwise_acc = []
    for i in range(min(gt), max(gt)+1):
        tot = np.sum(gt==i)
        if tot>0:
            classwise_acc.append(np.sum((pred==gt)&(gt==i))/tot)
    
    res['classification'] = {
        'accuracy': np.sum((pred==gt)&(gt>0))/np.sum(gt>0),
        'mean_accuracy': np.array(classwise_acc).mean(),
    }
    
    pred, gt = pred>0, gt>0
    
    TP = np.sum(pred   & gt)
    FN = np.sum(pred   & (~gt))
    FP = np.sum((~pred)& gt)
    
    res['segmentation'] = {
        'accuracy' : np.sum(pred==gt)/gt.size,
        'precision': TP/(TP+FP),
        'recall'   : TP/(TP+FN),
        'iou'      : np.sum(pred&gt)/np.sum(pred|gt),
        'dice'     : np.sum(pred&gt)*2/(np.sum(pred)+np.sum(gt))
    }
    
    return res


def evaluate_3D_path(pred_path, gt_path):
    with h5py.File(pred_path, 'r') as fp:
        pred = np.array(fp['data'])
        pred = pred>0.5
    
    with h5py.File(gt_path, 'r') as fp:
        gt = np.array(fp['annot'])
        gt = gt[gt.shape[0]%16:, ...]
        
    return evaluate_3D_image(pred, gt)