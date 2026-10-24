'use strict';
const $=selector=>document.querySelector(selector);
const status=$('#status'),button=$('#submit');
let config,library,worker,ready,pending=new Map(),requestId=0;
const names={tfidf:'TF-IDF + linear model',rnn:'RNN',lstm:'LSTM',gru:'GRU',transformer:'Our Transformer',frozen:'Frozen BERT + head',finetuned:'Fine-tuned BERT · deployed'};
const notebooks=[
 ['01_rnn.ipynb','Learning from text with an RNN','Build the dataset, follow token IDs into embeddings and trace loss, gradients and updates.'],
 ['02_lstm_gru.ipynb','Managing memory with LSTM and GRU','Work through the gates, compare the models and investigate their errors.'],
 ['03_transformer.ipynb','Building our small Transformer','Implement attention, positions, padding masks and an encoder classifier.'],
 ['04_pretrained.ipynb','Adapting a pretrained Transformer','Inspect the checkpoint, freeze features, fine-tune and evaluate honestly.'],
 ['05_deployment.ipynb','From trained model to deployed app','Export, verify prediction parity, retrieve real excerpts and publish the application.']
];
function element(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
async function readJSON(path){const response=await fetch(path);if(!response.ok)throw Error(`Could not load ${path}`);return response.json();}
function say(text,error=false){status.textContent=text;status.classList.toggle('error',error);}
const initial=Promise.all([readJSON('config.json'),readJSON('library.json')]).then(([c,l])=>{
  config=c;library=l;
  for(const record of c.results){const row=element('tr');if(record.name==='finetuned')row.className='deployed';
    [names[record.name],record.test_f1.toFixed(3),(100*record.question_accuracy).toFixed(1)+'%'].forEach(value=>row.append(element('td',value)));$('#metrics').append(row);}
  $('#dataset-summary').textContent='1,752 passages · 337 source sections · 1,201 train / 286 validation / 265 test. Complete source sections stay in one split. The question probe set is separate.';
}).catch(error=>{say('Project assets could not load. Refresh the page to try again.',true);throw error;});
for(let i=0;i<notebooks.length;i++){
  const [file,title,description]=notebooks[i],row=element('article',undefined,'notebook');row.append(element('span',String(i+1).padStart(2,'0'),'number'));
  const body=element('div');body.append(element('h3',title),element('p',description));row.append(body);
  const nav=element('nav');nav.setAttribute('aria-label',title);
  const download=element('a','Notebook ↓');download.href='notebooks/'+file;download.download=file;
  const colab=element('a','Run in Colab ↗');colab.href='https://colab.research.google.com/github/solomonpromise/course-navigator/blob/main/'+file;colab.target='_blank';colab.rel='noopener';
  nav.append(download,colab);row.append(nav);$('#notebooks').append(row);
}
function request(type,text,timeout=300000){return new Promise((resolve,reject)=>{
  const id=++requestId,timer=setTimeout(()=>{pending.delete(id);reject(Error('The model took too long to respond. Please retry.'));},timeout);
  pending.set(id,{resolve,reject,timer});worker.postMessage({id,type,text});
});}
function resetWorker(){if(worker)worker.terminate();worker=null;ready=null;for(const item of pending.values()){clearTimeout(item.timer);item.reject(Error('Model worker stopped'));}pending.clear();}
async function loadModel(){
  if(!ready){worker=new Worker('inference-worker.js?v=2');worker.onmessage=({data})=>{const item=pending.get(data.id);if(!item)return;if(data.progress){say(data.progress);return;}clearTimeout(item.timer);pending.delete(data.id);if(data.error)item.reject(Error(data.error));else item.resolve(data);};
    worker.onerror=()=>resetWorker();ready=request('init');}
  return ready;
}
function retrieve(text,topic){
  // Mirror the exported TF-IDF index: stop words, logarithmic TF, IDF and L2 norm.
  const terms=text.toLowerCase().normalize('NFKD').replace(/\p{M}/gu,'').match(/[\p{L}\p{N}_]{2,}/gu)||[];
  const stop=new Set(library.stop_words),counts=new Map();
  for(const term of terms)if(!stop.has(term)&&Object.hasOwn(library.vocabulary,term)){const id=library.vocabulary[term];counts.set(id,(counts.get(id)||0)+1);}
  let norm=0;const query=new Map();for(const [id,count]of counts){const value=(1+Math.log(count))*library.idf[id];query.set(id,value);norm+=value*value;}norm=Math.sqrt(norm)||1;
  return library.sections.map(section=>{let score=0;section.indices.forEach((id,i)=>{score+=(query.get(id)||0)/norm*section.values[i];});return {...section,score,rank:score*(section.label===topic?1.15:1)};}).filter(row=>row.score>0).sort((a,b)=>b.rank-a.rank).slice(0,4);
}
function excerpt(text,question){
  const terms=new Set((question.toLowerCase().match(/[a-z]{3,}/g)||[]).filter(t=>!library.stop_words.includes(t)));
  // Split at sentence boundaries without breaking decimals or lesson IDs (8.1).
  const sentences=text.split(/(?<=[.!?])\s+(?=[A-Z“"‘])/u);
  let index=0,best=-1;sentences.forEach((sentence,i)=>{const score=[...terms].filter(t=>sentence.toLowerCase().includes(t)).length;if(score>best){best=score;index=i;}});
  const chosen=sentences.slice(index,index+2).join(' ').trim();return chosen.length>460?chosen.slice(0,457)+'…':chosen;
}
function render(result,text){
  const ordered=result.probabilities.map((score,id)=>({score,id})).sort((a,b)=>b.score-a.score),top=ordered[0],sections=retrieve(text,top.id);
  const weak=!sections.length||sections[0].score<config.policy.minimum_query_similarity;
  const tentative=weak||top.score<config.policy.confidence_threshold;
  $('#certainty').textContent=weak?'Little course overlap':tentative?'Tentative suggestions':'Suggested starting point';
  $('#topic').textContent=weak?'Try a more specific course question':config.labels[top.id];
  $('#result-note').textContent=(tentative?'The model is uncertain or found little matching course material. Check the excerpts carefully. ':'Start here, then check whether the excerpt fits your question. ')+(result.truncated?'Your input exceeded 94 WordPiece tokens; the model used its beginning.':'');
  $('#lessons').replaceChildren();
  for(const section of sections){const item=element('li'),link=element('a',section.title,'lesson-link');
    // Course URLs originate in our fixed index, never from the learner's input.
    const url=new URL(section.url);if(url.origin!=='https://solomonpromise.github.io')continue;
    link.href=url.href;link.target='_blank';link.rel='noopener';item.append(element('p','Lesson '+section.lesson+' · '+section.topic,'lesson-meta'),link,element('p',excerpt(section.text,text),'excerpt'));$('#lessons').append(item);}
  $('#probabilities').replaceChildren();for(const row of ordered){const div=element('div',undefined,'probability'),bar=element('progress');bar.max=1;bar.value=row.score;bar.setAttribute('aria-label',config.labels[row.id]);div.append(element('span',config.labels[row.id]),element('span',(row.score*100).toFixed(1)+'%'),bar);$('#probabilities').append(div);}
  $('#tokens').replaceChildren(...result.tokens.map(token=>element('span',token)));
  $('#results').hidden=false;say(`Prediction completed on your device in ${Math.round(result.milliseconds)} ms. The model can still make mistakes.`);
}
$('#question-form').addEventListener('submit',async event=>{
  event.preventDefault();const text=$('#question').value.trim();if(!text){say('Enter a question or a short passage.',true);return;}
  button.disabled=true;$('#results').hidden=true;
  try{await initial;say(ready?'Finding relevant course sections…':'Loading the pretrained model on your device. The first search takes longer…');await loadModel();render(await request('predict',text,30000),text);}
  catch(error){resetWorker();say('Could not run the model. Check your connection, then try again. '+error.message,true);}
  finally{button.disabled=false;}
});
document.querySelectorAll('[data-example]').forEach(button=>button.addEventListener('click',()=>{$('#question').value=button.dataset.example;$('#question').focus();}));
