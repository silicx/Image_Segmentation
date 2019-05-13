import torch

# SR : Segmentation Result
# GT : Ground Truth

class Metrics:
    def __init__(self, SR=None, GT=None, threshold=0.5):
        self.acc = 0.		# Accuracy
        self.SE  = 0.		# Sensitivity (Recall)
        self.SP  = 0.		# Specificity
        self.PC  = 0. 		# Precision
        self.F1  = 0.		# F1 Score
        self.JS  = 0.		# Jaccard Similarity
        self.DC  = 0.		# Dice Coefficient

        if SR is not None and GT is not None:
            SR = torch.split(SR, 1, dim=0)
            GT = torch.split(GT, 1, dim=0)
            for i in range(len(SR)):
                self.acc += get_accuracy(   SR[i], GT[i], threshold)
                self.SE  += get_sensitivity(SR[i], GT[i], threshold)
                self.SP  += get_specificity(SR[i], GT[i], threshold)
                self.PC  += get_precision(  SR[i], GT[i], threshold)
                self.F1  += get_F1(         SR[i], GT[i], threshold)
                self.JS  += get_JS(         SR[i], GT[i], threshold)
                self.DC  += get_DC(         SR[i], GT[i], threshold)


    def add(self, met):
        self.acc += met.acc
        self.SE  += met.SE 
        self.SP  += met.SP 
        self.PC  += met.PC 
        self.F1  += met.F1 
        self.JS  += met.JS 
        self.DC  += met.DC 

    
    def div(self, num):
        self.acc /=  num
        self.SE  /=  num
        self.SP  /=  num
        self.PC  /=  num
        self.F1  /=  num
        self.JS  /=  num
        self.DC  /=  num

    def __str__(self):
        return 'Acc={:.4f}, SE={:.4f}, PC={:.4f}, JS={:.4f}, DC={:.4f}'.format(
				self.acc, self.SE, self.PC, self.JS, self.DC)



def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
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

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC
