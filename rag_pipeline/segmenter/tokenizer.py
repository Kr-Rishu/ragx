import numpy as np
import tiktoken

class Tokenizer:
    
    def __init__(self):
        pass
    
    def get_tokenizer(self, model_name):
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return encoding.encode
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode
        
    def count_tokens(self, text, model_name = 'gpt-4'):
        tokenizer = self.get_tokenizer(model_name)
        if tokenizer:
            return len(tokenizer(text))
        else:
            return max(1, len(text) // 4)
        
    def get_token_context(self, tokens, words, token_count):
        till_index = np.argwhere((np.cumsum(tokens)<=token_count)== True)[-1][0]
        return till_index, " ".join(words[0:till_index])
        
        
    
        
            

                
    
