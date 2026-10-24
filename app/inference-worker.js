/* Keep model loading and matrix operations off the page's UI thread. */
importScripts('vendor/ort.min.js','tokenizer.js');
let session,config;
async function download(path,id,label){
  const response=await fetch(path);
  if(!response.ok)throw Error(label+' could not download (HTTP '+response.status+')');
  const total=Number(response.headers.get('content-length'))||0;
  if(!response.body){const bytes=new Uint8Array(await response.arrayBuffer());return bytes;}
  const reader=response.body.getReader(),chunks=[];let received=0,last=0;
  for(;;){const {done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.length;
    if(performance.now()-last>500){postMessage({id,progress:`Downloading ${label}: ${(received/1e6).toFixed(1)}${total?' / '+(total/1e6).toFixed(1):''} MB`});last=performance.now();}}
  const bytes=new Uint8Array(received);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}return bytes;
}
onmessage=async({data})=>{
  const {id,type,text}=data;
  try {
    if(type==='init') {
      config=await fetch('config.json').then(r=>{if(!r.ok)throw Error('Model settings unavailable');return r.json();});
      ort.env.wasm.numThreads=1;
      ort.env.wasm.wasmPaths=new URL('vendor/',self.location.href).href;
      // Fetch the two large assets together, with visible progress. Supplying
      // wasmBinary prevents the runtime from starting another sequential fetch.
      const [modelBytes,wasmBytes]=await Promise.all([
        download('model.onnx',id,'model'),
        download('vendor/ort-wasm-simd-threaded.wasm',id,'runtime')]);
      ort.env.wasm.wasmBinary=wasmBytes;
      postMessage({id,progress:'Downloads complete. Preparing the model on your device…'});
      session=await ort.InferenceSession.create(modelBytes,{executionProviders:['wasm']});
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
