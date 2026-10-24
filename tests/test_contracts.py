"""Meaningful regressions for data isolation, masks and exported inference."""
from pathlib import Path
import json,sys,unittest,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from coursenav.data import ROOT,load_records,fit_vocabulary,encode,TextDataset
from coursenav.models import RecurrentClassifier,TinyTransformer,SelfAttention
from coursenav.training import seed_everything,pretrained_tokenizer


class DataContract(unittest.TestCase):
    def test_whole_sections_stay_in_one_split(self):
        assignments={}
        records=load_records()
        for row in records:
            self.assertEqual(assignments.setdefault(row['group'],row['split']),row['split'])
        self.assertEqual(len({row['text_sha256'] for row in records}),len(records))
        self.assertEqual(len(records),1752)

    def test_vocabulary_cannot_fit_test_text(self):
        with self.assertRaises(ValueError):fit_vocabulary(load_records('test'))
        vocabulary=fit_vocabulary(load_records('train'))
        self.assertEqual(vocabulary,json.loads((ROOT/'artifacts/vocabulary.json').read_text()))
        ids,length=encode('',vocabulary)
        self.assertEqual((ids[0],length),(1,1))


class ModelContract(unittest.TestCase):
    def setUp(self):seed_everything()

    def test_trailing_padding_does_not_change_prediction(self):
        ids=torch.tensor([[2,3,4,5]])
        padded=torch.tensor([[2,3,4,5,0,0,0]])
        length=torch.tensor([4])
        for model in [RecurrentClassifier(20,kind) for kind in ['RNN','LSTM','GRU']]+[TinyTransformer(20)]:
            model.eval()
            with torch.inference_mode():
                self.assertTrue(torch.allclose(model(ids,length),model(padded,length),atol=1e-6))

    def test_attention_matches_torch_with_masked_keys(self):
        attention=SelfAttention().eval();x=torch.randn(2,5,64)
        mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
        actual,weights=attention(x,mask,inspect=True)
        q,k,v=[part.reshape(2,5,4,16).transpose(1,2) for part in attention.qkv(x).chunk(3,-1)]
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:])
        expected=attention.output(expected.transpose(1,2).reshape(2,5,64))
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        self.assertEqual(float(weights[0,:,:,3:].abs().max().detach()),0.)

    def test_export_logits_and_tokenizer_match_reference(self):
        import onnxruntime as ort
        session=ort.InferenceSession(str(ROOT/'app/model.onnx'),providers=['CPUExecutionProvider'])
        tokenizer=pretrained_tokenizer()
        for case in json.loads((ROOT/'tests/browser-fixtures.json').read_text()):
            batch=tokenizer(case['text'],padding='max_length',truncation=True,max_length=96,return_tensors='np')
            np.testing.assert_array_equal(batch['input_ids'][0],case['ids'])
            actual=session.run(None,{key:value.astype(np.int64) for key,value in batch.items()})[0][0]
            np.testing.assert_allclose(actual,case['logits'],atol=1e-4,rtol=1e-5)
        config=json.loads((ROOT/'app/config.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'app/model.onnx').read_bytes()).hexdigest(),config['model_sha256'])


if __name__=='__main__':unittest.main()
