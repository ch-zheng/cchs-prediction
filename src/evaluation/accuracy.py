class accuracy():
    def __init__(self, m, cnf_matrix):
        self.model = m
        self.sensitivity = self.sensitivity(cnf_matrix)
        self.specificity = self.specificity(cnf_matrix)
    def sensitivity(self, cnf_matrix):
        tp = cnf_matrix[1][1]
        fn = cnf_matrix[1][0]
        return tp / (tp + fn)
    def specificity(self, cnf_matrix):
        tn = cnf_matrix[0][0]
        fp = cnf_matrix[0][1]
        return tn / (tn + fp)
    def __str__(self):
        return "model: "+type(self.model).__name__+" sensitivity: "+str(self.sensitivity)+" specificity: "+str(self.specificity)
    def csv_str(self):
        return [type(self.model).__name__, str(self.sensitivity), str(self.specificity)]

def evaluate(cnf_matrices):
    evaluations = {}
    for model_name, cnf in cnf_matrices.items():
        evaluations[model_name] = accuracy(cnf)
    
    print("____________________")
    [print(model_name, e) for model_name, e in evaluations.items()]
    print("____________________")