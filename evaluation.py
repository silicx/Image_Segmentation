import torch

# SR : Segmentation Result
# GT : Ground Truth

class Metrics(object):
    acc = 0.		# Accuracy
    SE  = 0.		# Sensitivity (Recall)
    SP  = 0.		# Specificity
    PC  = 0. 		# Precision
    F1  = 0.		# F1 Score
    JS  = 0.		# Jaccard Similarity
    DC  = 0.		# Dice Coefficient

    def __init__(self, SR=None, GT=None):
        if SR is not None and GT is not None:
            self.acc = self.get_accuracy(SR,GT)
            self.SE  = self.get_sensitivity(SR,GT)
            self.SP  = self.get_specificity(SR,GT)
            self.PC  = self.get_precision(SR,GT)
            self.F1  = self.get_F1(SR,GT)
            self.JS  = self.get_JS(SR,GT)
            self.DC  = self.get_DC(SR,GT)


    def add(self, metrics):
        acc += metrics.acc
        SE  += metrics.SE 
        SP  += metrics.SP 
        PC  += metrics.PC 
        F1  += metrics.F1 
        JS  += metrics.JS 
        DC  += metrics.DC 

    
    def div(self, num):
        acc /=  num
        SE  /=  num
        SP  /=  num
        PC  /=  num
        F1  /=  num
        JS  /=  num
        DC  /=  num



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
