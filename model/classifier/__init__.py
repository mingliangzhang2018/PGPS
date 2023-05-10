from .classifier_ops import *
from config import classifier_list

       
def get_classifier(args):

    bias_flag = args.classifier_bias
    num_features = args.num_features
    num_classes = args.num_classes

    if not args.classifier in classifier_list:
        raise NotImplementedError("Unsupported Classifier: {}".format(args.classifier))

    if args.classifier == "FCNorm":
        classifier = FCNorm(num_features, num_classes)
    elif args.classifier == "CosNorm":
        classifier = CosNorm(num_features, num_classes)
    elif args.classifier == "DotProduct":
        classifier = DotProduct(num_classes, num_features, bias_flag)
    elif args.classifier == "DistFC":
        classifier = DistFC(num_features, num_classes)
 
    return classifier