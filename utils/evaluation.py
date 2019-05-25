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

        if SR is not None and GT is not None:
            SR = torch.split(SR, 1, dim=0)
            GT = torch.split(GT, 1, dim=0)
            for i in range(len(SR)):
                self.acc.append( get_accuracy(   SR[i], GT[i], threshold) )
                self.SE.append(  get_sensitivity(SR[i], GT[i], threshold) )
                self.SP.append(  get_specificity(SR[i], GT[i], threshold) )
                self.PC.append(  get_precision(  SR[i], GT[i], threshold) )
                self.F1.append(  get_F1(         SR[i], GT[i], threshold) )
                self.JS.append(  get_JS(         SR[i], GT[i], threshold) )
                self.DC.append(  get_DC(         SR[i], GT[i], threshold) )


    def add(self, met):
        self.acc += met.acc
        self.SE  += met.SE 
        self.SP  += met.SP 
        self.PC  += met.PC 
        self.F1  += met.F1 
        self.JS  += met.JS 
        self.DC  += met.DC 

    def __str__(self):
        return 'Acc={:.4f}, SE={:.4f}, PC={:.4f}, JS={:.4f}, DC={:.4f}'.format(
				np.nanmean(np.array(self.acc)),
                np.nanmean(np.array(self.SE)), 
                np.nanmean(np.array(self.PC)), 
                np.nanmean(np.array(self.JS)), 
                np.nanmean(np.array(self.DC)))



def get_accuracy(SR,GT,threshold=0.5):
    SR = (SR > threshold)
    GT = (GT == torch.max(GT))
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    tp_fn = torch.sum(TP+FN)
    if tp_fn == 0:
        return np.nan
    else:
        return float(torch.sum(TP))/float(tp_fn) 
        

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    tn_fp = torch.sum(TN+FP)
    if tn_fp == 0:
        return np.nan
    else:
        return float(torch.sum(TN))/float(tn_fp)
        

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
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
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    if Union == 0:
        return np.nan
    else:
        return float(Inter)/float(Union)


def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    Sum   = torch.sum(SR)+torch.sum(GT)

    if Sum == 0:
        return np.nan
    else:
        return float(2*Inter)/(float(Sum))