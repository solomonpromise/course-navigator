"""Seal final metrics and export the exact adapted model for browser inference."""
from pathlib import Path
import json,sys,hashlib,time,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from coursenav.data import ROOT,LABELS,MAX_LENGTH,load_records
from coursenav.training import seed_everything,load_reference,dataset_for,logits_for,summarize,pretrained_tokenizer,MODEL_ID,MODEL_REVISION
from coursenav.retrieval import LessonIndex


def main():
    seed_everything()
    train,valid,test=load_records('train'),load_records('validation'),load_records('test')
    probes=json.loads((ROOT/'data/question_probes.json').read_text())
    questions=[dict(text=text,label=label) for text,label in probes['examples']]
    results=[]
    baseline=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=12000,sublinear_tf=True),
                           LogisticRegression(C=2.,class_weight='balanced',max_iter=1000,random_state=42))
    started=time.perf_counter();baseline.fit([r['text'] for r in train],[r['label'] for r in train])
    elapsed=time.perf_counter()-started
    results.append(dict(name='tfidf',parameters=None,train_seconds=elapsed,
                        validation=summarize([r['label'] for r in valid],baseline.predict_proba([r['text'] for r in valid])),
                        test=summarize([r['label'] for r in test],baseline.predict_proba([r['text'] for r in test])),
                        questions=summarize([r['label'] for r in questions],baseline.predict_proba([r['text'] for r in questions]))))
    for name in ['rnn','lstm','gru','transformer','frozen','finetuned']:
        print('EVALUATING',name,flush=True)
        model=load_reference(name)
        record=json.loads((ROOT/f'artifacts/{name}-run.json').read_text())
        for split,rows in [('validation',valid),('test',test),('questions',questions)]:
            logits=logits_for(model,dataset_for(name,rows)).numpy()
            record[split]=summarize([row['label'] for row in rows],logits)
            if name=='finetuned':
                np.save(ROOT/f'artifacts/{split}-logits.npy',logits)
        results.append(record)
    report=dict(protocol='Frozen settings, section-group split, one reference seed. Test/probes not used for model, epoch, temperature or threshold selection.',
                runtime=dict(python=platform.python_version(),torch=torch.__version__,device='CPU',threads=4),models=results)
    (ROOT/'artifacts/results.json').write_text(json.dumps(report,indent=2))

    # Temperature scaling changes confidence, not the winning class. Fit on validation only.
    model=load_reference('finetuned')
    valid_logits=torch.tensor(np.load(ROOT/'artifacts/validation-logits.npy'))
    labels=torch.tensor([r['label'] for r in valid])
    log_temperature=torch.zeros((),requires_grad=True)
    opt=torch.optim.LBFGS([log_temperature],lr=.1,max_iter=50)
    def closure():
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(valid_logits/log_temperature.exp().clamp(.5,5.),labels)
        loss.backward();return loss
    opt.step(closure)
    temperature=float(log_temperature.detach().exp().clamp(.5,5.))
    probabilities=(valid_logits/temperature).softmax(-1).numpy();confidence=probabilities.max(1);correct=probabilities.argmax(1)==labels.numpy()
    candidates=[]
    for threshold in np.arange(.35,.951,.025):
        keep=confidence>=threshold
        if keep.sum()>=20 and correct[keep].mean()>=.80:
            candidates.append((float(keep.mean()),float(threshold),float(correct[keep].mean())))
    # No promise of 80% reliability if validation cannot support it.
    coverage,threshold,accepted_accuracy=max(candidates) if candidates else (0.,.95,None)
    policy=dict(temperature=temperature,confidence_threshold=threshold,validation_coverage=coverage,
                validation_accepted_accuracy=accepted_accuracy,minimum_query_similarity=.02,
                note='Heuristic tentative-result handling; softmax confidence is not a correctness guarantee or a validated out-of-domain detector.')
    (ROOT/'artifacts/policy.json').write_text(json.dumps(policy,indent=2))
    # Fixed 96-token input keeps browser export and timing easy to reproduce.
    tokenizer=pretrained_tokenizer();tokenizer.save_pretrained(ROOT/'artifacts/tokenizer')
    dummy=tokenizer('How does attention work?',padding='max_length',truncation=True,max_length=MAX_LENGTH,return_tensors='pt')
    target=ROOT/'app/model.onnx'
    torch.onnx.export(model,(dummy['input_ids'],dummy['attention_mask'],dummy['token_type_ids']),str(target),
                      input_names=['input_ids','attention_mask','token_type_ids'],output_names=['logits'],opset_version=17,dynamo=False)
    import onnx,onnxruntime as ort
    onnx.checker.check_model(onnx.load(target))
    session=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
    fixtures=[]
    texts=[r['text'] for r in questions]+probes['outside_scope']+['café naïve résumé','中文 attention','attention\n\tmask','',"[MASK] isn't a word",'x'*110]
    worst=0.
    for text in texts:
        batch=tokenizer(text,padding='max_length',truncation=True,max_length=MAX_LENGTH,return_tensors='pt')
        inputs={k:v.numpy().astype(np.int64) for k,v in batch.items()}
        with torch.inference_mode():
            expected=model(batch['input_ids'],batch['attention_mask'],batch['token_type_ids']).numpy()
        actual=session.run(None,inputs)[0];worst=max(worst,float(np.abs(expected-actual).max()))
        fixtures.append(dict(text=text,ids=inputs['input_ids'][0].tolist(),mask=inputs['attention_mask'][0].tolist(),logits=expected[0].tolist()))
    if worst>1e-4:raise ValueError(f'ONNX mismatch: {worst}')
    (ROOT/'tests/browser-fixtures.json').write_text(json.dumps(fixtures,ensure_ascii=False))
    index=LessonIndex();(ROOT/'app/library.json').write_text(json.dumps(index.export_browser(),ensure_ascii=False,separators=(',',':')))
    vocabulary=tokenizer.get_vocab()
    config=dict(labels=LABELS,max_length=MAX_LENGTH,vocabulary=vocabulary,policy=policy,
                model=MODEL_ID,revision=MODEL_REVISION,source_revision=json.loads((ROOT/'data/manifest.json').read_text())['source_revision'],
                model_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),onnx_max_logit_error=worst,
                results=[dict(name=r['name'],test_f1=r['test']['macro_f1'],question_accuracy=r['questions']['accuracy'],parameters=r.get('parameters')) for r in results])
    (ROOT/'app/config.json').write_text(json.dumps(config,ensure_ascii=False,separators=(',',':')))
    print('ONNX maximum logit error',worst,'policy',policy,flush=True)
    print([(r['name'],round(r['test']['macro_f1'],3),round(r['questions']['accuracy'],3)) for r in results],flush=True)


if __name__=='__main__':main()
