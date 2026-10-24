/* Keep model loading and matrix operations off the page's UI thread. */
importScripts('vendor/ort.min.js','tokenizer.js');
let session,config;
onmessage=async({data})=>{
  const {id,type,text}=data;
  try {
    if(type==='init') {
      config=await fetch('config.json').then(r=>{if(!r.ok)throw Error('Model settings unavailable');return r.json();});
      ort.env.wasm.numThreads=1;
      ort.env.wasm.wasmPaths=new URL('vendor/',self.location.href).href;
      session=await ort.InferenceSession.create('model.onnx',{executionProviders:['wasm']});
      postMessage({id,ready:true});return;
    }
    if(!session)throw Error('Load the model before predicting');
    const started=performance.now(),encoded=CourseTokenizer.tokenize(text,config);
    const tensor=values=>new ort.Tensor('int64',BigInt64Array.from(values,BigInt),[1,config.max_length]);
    const output=await session.run({input_ids:tensor(encoded.ids),attention_mask:tensor(encoded.mask),token_type_ids:tensor(encoded.ids.map(()=>0))});
    const logits=Array.from(output.logits.data);
    postMessage({id,probabilities:CourseTokenizer.softmax(logits,config.policy.temperature),
                 tokens:encoded.tokens,truncated:encoded.truncated,milliseconds:performance.now()-started});
  }catch(error){postMessage({id,error:error.message||String(error)});}
};
