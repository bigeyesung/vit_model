#0.Packages
    Python==3.6
    cudatoolkit==11.0.221 
    keras==2.0.8
    tensorflow-gpu==1.15.0

#1.How to run code
    $conda create -n py36 python=3.6
    $python main.py

#2.Concept
    I came across one vision transformer paper and it mentioned it has 
    better performance than other CNN models although it takes longer training time.
    This assignment has smaller dataset so the training time is not an issue. I also 
    have one current working vision transformer model on hand. So I decide directly move it 
    to this assignment. 

    At first I assigned train/valid/test data ratio to 300:15:15. The initial valid data accuracy is around 91%.
    Walking through the dataset, I found animal group has the most diverse species and I infer it makes accuracy lower.
    Also, I think original dataset is too small, so I decided to seperate animal group into 8 groups(E.g. birds, lions...etc),
    and added additional images to make these 8 groups balanced. 

    Then,I trained the 8-group animals, buildings and landscapes again and received a promising result.
    I decided to merge them into original 3 groups. After we have these augmented data, the valid data accuracy arrived 96%
    The accuracy value has ups and downs, which causes the model could not find the best weights. So I changed the 
    learning rate dynamically during training. After I fine-tuned it, the model achieved 97~98% accuracy.

    Given we have a higher accuracy, I think the best way to test a model is through real data.
    Thus I prepared some images from the internet for tested data. The tested data accuracy is 96%.
    In section 6, it is listed detailed information regaring the tested data.

#3.Input data:
    #3-1.model folder: The result model.
    #3-2.screenshots folder: validation loss and accuracy.
    #3-3.input/test folder: additional pictures from internet

#4.Key components
    #4-1.Classfier class: Mainly to working on model classification. 
    #4-2.Utils class: Mainly to dealing with input, output data and other GPU things.

#5.Enum 
    #5-1.common.py: Mainly for enum and common variables

#6.Evaluation
    I prepared 40 pictures equally for each group, and below are their results
    #6-1.accuracy(major)
        In terms of accuracy, building groups has the lowest number and Landscapes are the highest.
        I infer it is due to building pictures ususally have common features with landscapes, such as mountains, sunset..etc.
        However, landscapes pictures have unique features among the groups.
    #6-2.precision(major)
        In terms of precision, It shows that Animal group has the lowest number. One potential 
        reason is I test with "Girafe toy" pictures and "3D rendered frog", which is confusing to judge if they are real animals.
        Perhaps, we could also try some "tricky" test data with the rest of two groups.
    #6-3.Recall(less important)
        In terms of recall, animal group has the highest value, then is landscapes, 
        followed by building group. I assume precision has higher importance than recall in this assignment.
        As eventually what we want to know is how accurate this classifier distinguishes these three groups.
        So I would mark recall as "less important" in this case.
            | Animal |Building | Landscapes |
TP          | 38     | 36      | 39         |
TN          | 0      | 0       | 0          |
FP          | 2      | 0       | 0          |
FN          | 0      | 4       | 1          |
Accuracy    | 95%    | 90 %    | 97.5%      |
Precision   | 95%    | 100 %   | 100%       |
Recall      | 100%   | 90 %    | 97.5%      |

TP:true positive
TN:true negative
FP:false positive
FN:false negative
Accuracy: (TP+TN)/(TP+TN+FP+FN)
Precision:TP/(TP+FP)
Recall:TP/(TP+FN)

#7.Improvement and future work:
#7-1.[Finish]
    Augmented the dataset.
    Learning rate dynamically changes during traing. It controls updated weights.
    
#7-2.[Future work] 
    #Try different models(Retnet,RetinalNet...etc) and compare them with vision transformer.
    #Finding more metrics, such as using sklearn.classification_report


