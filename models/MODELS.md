
** model_dl_unbalanced.h5 **
trained on an unbalanced set of lowercase letters and digits
3 pairs of conv & maxpool + 2 dense layers

** model_dl_balanced.h5 **
trained on an balanced set of lowercase letters and digits
3 pairs of conv & maxpool + 2 dense layers
0.9222 accuracy

** model_dl_balanced_2conv.h5 **
trained on an balanced set of lowercase letters and digits
conv -> maxpool -> conv -> maxpool  -> dense1 -> dense2
0.9392

** model_dl_balanced_2conv1dense.h5 **
trained on an balanced set of lowercase letters and digits
conv -> maxpool -> conv -> maxpool  -> dense1
0.9511 
