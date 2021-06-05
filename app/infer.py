import torch

def predict(text, question):
    model = torch.load('app/model.pb', map_location='cpu')

    tokenizer = torch.load('app/tokenizer.pb', map_location='cpu')

    inputs = tokenizer(question, text, return_tensors='pt', max_length=512)
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])
    outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)
    sep_idx = tokens.index('[SEP]')

    start_idx, end_idx = start_idx - sep_idx - 1, end_idx - sep_idx - 1
    tokens = tokens[sep_idx+1:-1]

    s_idx = 0
    for i in range(start_idx):
        if '##' in tokens[i]:
            s_idx += len(tokens[i]) - 3
        else:
            s_idx += len(tokens[i])
        s_idx += 1

    e_idx = 0
    for i in range(start_idx, end_idx+1):
        if '##' in tokens[i]:
            e_idx += len(tokens[i]) - 3
        else:
            e_idx += len(tokens[i])
        e_idx += 1

    e_idx += s_idx
    tokens = ' '.join(tokens).replace(' ##', '')
    answer = tokens[s_idx:e_idx]
    
    results = {
        "answer": answer,
        "context": tokens,
        "answer_before": tokens[:s_idx],
        "answer_after": tokens[e_idx:]
    }
    
    return results
