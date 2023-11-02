import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib import rc
import seaborn as sns
palette = sns.color_palette("tab10")
#rc('text', usetex=True)
rc('font', family='Times New Roman')



# Esempio di probabilità previste per tre classi
predicted_probabilities = np.array([
[95, 0, 5],
   [10, 67, 23 ],
   [8 ,7 ,85]
])
class_labels = ["natural", "sketch", "comic"]
# Crea un grafico a colori con Seaborn
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(predicted_probabilities, yticklabels=class_labels, annot=True, cmap='Blues', fmt=".2f",annot_kws={"size": 25})
plt.setp(heatmap.get_yticklabels(), rotation=90, fontsize=22) 
# Aggiungi etichette per l'asse x e y (opzionale)
#plt.xlabel('Classe Prevista')
#plt.ylabel('Target domains',fontsize =22)
plt.xticks([])
plt.title(r'$\mathbf{v}\times100$',fontsize = 20)
#plt.show()#
plt.savefig("prob.pdf", dpi=200, bbox_inches='tight', pad_inches=0.01)