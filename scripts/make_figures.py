"""Rebuild the lab's scientific figures from actual saved measurements."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from coursenav.data import ROOT,LABELS,encode
from coursenav.training import load_reference,seed_everything

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.facecolor':'#f7f7f1','figure.facecolor':'#f7f7f1','axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
colors=['#738478','#3f6a54','#6590a0','#b18b52','#92778a','#284d39','#b6553d']
def save(fig,name):fig.savefig(ROOT/'figures'/name,dpi=160);plt.close(fig)
seed_everything()
results=json.loads((ROOT/'artifacts/results.json').read_text())['models']
fig,ax=plt.subplots(figsize=(12,3.4));ax.axis('off')
steps=['Real course\npassages','IDs + masks\nTokenizer','Learned vectors\nEncoder','Six scores\nTopic head','Real excerpts\nLesson index']
for i,label in enumerate(steps):
 ax.text(.08+i*.21,.56,label,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.9',fc='#e8edde',ec='#738478'))
 if i<4:ax.annotate('',xy=(.08+i*.21+.145,.56),xytext=(.08+i*.21+.075,.56),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#284d39',lw=2))
ax.text(.50,.12,'Training: logits → cross-entropy → gradients → parameter updates\nDeployment: saved tokenizer + adapted encoder + head; source excerpts are retrieved separately',ha='center',transform=ax.transAxes,color='#3c5145')
save(fig,'01_text_pipeline.png')

fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for i,row in enumerate(results[1:]):
 h=row['history'];axes[0].plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label=row['name'],color=colors[i]);axes[1].plot([x['epoch'] for x in h],[x['validation_macro_f1'] for x in h],label=row['name'],color=colors[i])
axes[0].set(title='Fitting the training passages',xlabel='Epoch',ylabel='Weighted training cross-entropy');axes[1].set(title='Generalizing to held-out sections',xlabel='Epoch',ylabel='Validation macro-F1',ylim=(0,1));axes[1].legend(fontsize=9);fig.tight_layout();save(fig,'02_learning_curves.png')

fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(len(results));width=.36
ax.bar(x-width/2,[r['test']['macro_f1'] for r in results],width,label='Held-out passage macro-F1',color='#3f6a54');ax.bar(x+width/2,[r['questions']['accuracy'] for r in results],width,label='Authored question accuracy (36)',color='#b18b52');ax.set_xticks(x,['TF-IDF','RNN','LSTM','GRU','Our encoder','Frozen BERT','Tuned BERT']);ax.set(ylim=(0,1),ylabel='Score',title='One dataset and one reference run: a larger model does not guarantee a win');ax.legend(loc='upper left',fontsize=9);fig.tight_layout();save(fig,'03_model_comparison.png')

fig,ax=plt.subplots(figsize=(8.5,6.5));cm=np.array(results[-1]['test']['confusion_matrix']);im=ax.imshow(cm,cmap='Greens')
short=['Foundations','Training','Representations','Transformers','Pretrained','Adaptation'];ax.set_xticks(range(6),short,rotation=40,ha='right');ax.set_yticks(range(6),short)
for i in range(6):
 for j in range(6):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#20362c')
ax.set(xlabel='Predicted topic',ylabel='Actual topic',title='Fine-tuned BERT: every held-out passage, including mistakes');fig.colorbar(im,ax=ax,label='Passages');fig.tight_layout();save(fig,'04_confusion_matrix.png')

fig,ax=plt.subplots(figsize=(12,4));ax.axis('off')
blocks=[('RNN',r'$h_t=\tanh(W_x x_t+W_h h_{t-1}+b)$','One repeatedly updated state'),('LSTM',r'$c_t=f_t\odot c_{t-1}+i_t\odot g_t$','Separate cell and hidden states'),('GRU',r'$h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}$','Reset and update gates')]
for i,(title,equation,description) in enumerate(blocks):
 ax.text(.02+i*.34,.82,title,transform=ax.transAxes,fontsize=18,fontweight='bold',color='#284d39');ax.text(.02+i*.34,.53,equation,transform=ax.transAxes,fontsize=13);ax.text(.02+i*.34,.28,description,transform=ax.transAxes,fontsize=11)
ax.text(.02,.04,'Gates use sigmoid values in [0,1]. ⊙ means element-by-element multiplication. These equations illustrate mechanisms, not measured scores.',transform=ax.transAxes,fontsize=10)
save(fig,'05_memory_mechanisms.png')

model=load_reference('transformer');vocabulary=json.loads((ROOT/'artifacts/vocabulary.json').read_text());sentence='attention lets each token read information from other tokens';ids,length=encode(sentence,vocabulary);tensor=torch.tensor([ids]);mask=tensor.ne(0)
with torch.inference_mode():
 positions=torch.arange(tensor.shape[1]);x=model.embedding(tensor)+model.position(positions)[None,:,:]
 _,weights=model.blocks[0].attention(model.blocks[0].norm1(x),mask,inspect=True)
fig,ax=plt.subplots(figsize=(8,6));labels=sentence.split();im=ax.imshow(weights[0,0,:length,:length],cmap='YlGnBu',vmin=0);ax.set_xticks(range(length),labels,rotation=55,ha='right');ax.set_yticks(range(length),labels);ax.set(xlabel='Key token being read',ylabel='Query token gathering information',title='Actual first-layer attention, head 1 (scratch encoder)');fig.colorbar(im,ax=ax,label='Attention weight');fig.tight_layout();save(fig,'06_attention.png')

selected=[w for w in ['attention','token','loss','gradient','batch','image','kernel','memory','hidden','weights','fine','training','model','encoder'] if w in vocabulary]
model=load_reference('gru');vectors=model.embedding.weight.detach().numpy()[[vocabulary[w] for w in selected]];xy=PCA(n_components=2).fit_transform(vectors)
fig,ax=plt.subplots(figsize=(9,5));ax.scatter(xy[:,0],xy[:,1],color='#3f6a54')
for label,point in zip(selected,xy):ax.annotate(label,point,xytext=(5,5),textcoords='offset points')
ax.set(xlabel='Principal component 1',ylabel='Principal component 2',title='Actual GRU embedding rows projected to two dimensions');fig.text(.1,.01,'Projection loses information. Distances in this small task-trained vocabulary are not proof of general semantic understanding.',fontsize=9);save(fig,'07_embeddings.png')
print('Saved seven figures from the actual reference artifacts.')
