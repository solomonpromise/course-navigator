/* The inference contract, shared with tests and the browser worker.
 * This is uncased BERT WordPiece, not the word tokenizer used by scratch models.
 * Fixtures compare complete 96-token arrays with Hugging Face's tokenizer.
 */
(function (root) {
  'use strict';
  const owns = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  function isChinese(code) {
    return [[0x4E00,0x9FFF],[0x3400,0x4DBF],[0x20000,0x2A6DF],[0x2A700,0x2B73F],
      [0x2B740,0x2B81F],[0x2B820,0x2CEAF],[0xF900,0xFAFF],[0x2F800,0x2FA1F]].some(([a,b])=>code>=a&&code<=b);
  }
  function basic(text) {
    let cleaned='';
    for(const character of text) {
      const code=character.codePointAt(0);
      if (/\s/u.test(character)) cleaned+=' ';
      else if(code===0 || code===0xFFFD || /\p{C}/u.test(character)) continue;
      else cleaned+=isChinese(code)?` ${character} `:character;
    }
    cleaned=cleaned.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu,'');
    // ASCII symbols count as punctuation in BERT, as do Unicode punctuation characters.
    let separated='';
    for(const c of cleaned) {
      const n=c.codePointAt(0);
      const punctuation=(n>=33&&n<=47)||(n>=58&&n<=64)||(n>=91&&n<=96)||(n>=123&&n<=126)||/\p{P}/u.test(c);
      separated+=punctuation?` ${c} `:c;
    }
    return separated.trim().split(/\s+/u).filter(Boolean);
  }
  function tokenize(text, config) {
    const vocabulary=config.vocabulary, tokens=[];
    for(const part of text.split(/(\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\])/g)) {
      if(/^\[(PAD|UNK|CLS|SEP|MASK)\]$/.test(part)) { tokens.push(part);continue; }
      for(const word of basic(part)) {
        const chars=Array.from(word);
        if(chars.length>100) {tokens.push('[UNK]');continue;}
        let start=0,pieces=[],failed=false;
        while(start<chars.length) {
          let end=chars.length,match=null;
          while(end>start) {
            const candidate=(start?'##':'')+chars.slice(start,end).join('');
            if(owns(vocabulary,candidate)) {match=candidate;break;}
            end--;
          }
          if(match===null){failed=true;break;}
          pieces.push(match);start=end;
        }
        tokens.push(...(failed?['[UNK]']:pieces));
      }
    }
    const truncated=tokens.length>config.max_length-2;
    const kept=['[CLS]',...tokens.slice(0,config.max_length-2),'[SEP]'];
    const ids=kept.map(token=>vocabulary[token]),mask=ids.map(()=>1);
    while(ids.length<config.max_length){ids.push(vocabulary['[PAD]']);mask.push(0);}
    return {ids,mask,tokens:kept,truncated};
  }
  function softmax(logits,temperature=1) {
    const scores=Array.from(logits,x=>x/temperature),max=Math.max(...scores);
    const exp=scores.map(x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0);
    return exp.map(x=>x/sum);
  }
  const api={tokenize,softmax};
  root.CourseTokenizer=api;
  if(typeof module!=='undefined')module.exports=api;
})(globalThis);
