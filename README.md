# test-de-memoire-master-UY1
# Ce travail permet de faire des simulations sur la reconnaissance de race en utilisaant CSLBP de Heikilla et DSLBP que nous avons proposé.
# On utilise des images de taille 100x100.
# On utilise les blocs de taille 50x50, 40x40, 30x30, 25x25, 20x20, 15x15, 10x10
# Pour changer la taille des blocs il suffit de changer la variable "size_block" ligne 41
# On utilise une notion de contiguïté de blocs qui est mise en oeuvre lors de l'incrémentation pour passer au bloc suivant
# 1-contiguïté : i  = i + size_block et j = j + size_block
# 1/2-contiguïté : i  = i + size_block//2 et j = j + size_block//2
# 1/4-contiguïté : i  = i + size_block//4 et j = j + size_block//4
