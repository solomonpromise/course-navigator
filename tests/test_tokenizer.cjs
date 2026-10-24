'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'..');
const {tokenize,softmax}=require('../app/tokenizer.js');
const config=JSON.parse(fs.readFileSync(path.join(root,'app/config.json')));
const fixtures=JSON.parse(fs.readFileSync(path.join(root,'tests/browser-fixtures.json')));
for(const fixture of fixtures){
  const encoded=tokenize(fixture.text,config);
  assert.deepStrictEqual(encoded.ids,fixture.ids,fixture.text);
  assert.deepStrictEqual(encoded.mask,fixture.mask,fixture.text);
  assert.equal(encoded.ids.length,96);
}
assert(Math.abs(softmax([1000,1001,999]).reduce((a,b)=>a+b,0)-1)<1e-12);
const long=tokenize('attention '.repeat(150),config);assert(long.truncated);assert.equal(long.tokens.length,96);
console.log(`${fixtures.length} full BERT token/mask arrays match Python; truncation and stable softmax pass.`);
